"""Per-client live analysis state machine.

All computer-vision operations are injected so the exact same session contract
can be backed by the current server pipeline, Expo Go frame uploads, or a future
native/on-device implementation. No source or model files are modified at runtime.
"""
from collections import deque
from datetime import datetime, timezone
from enum import Enum
import json
import logging
import math
import time
import uuid

import cv2
import numpy as np
from baseline_state import BaselineLock


logger = logging.getLogger(__name__)


class AnalysisState(str, Enum):
    BASELINE_SEARCH = "BASELINE_SEARCH"
    BASELINE_CONFIRMATION = "BASELINE_CONFIRMATION"
    BASELINE_LOCKED = "BASELINE_LOCKED"
    LIVE_LINE_JUDGE = "LIVE_LINE_JUDGE"
    SHUTTLE_TRACKING = "SHUTTLE_TRACKING"
    LANDING_EVALUATION = "LANDING_EVALUATION"
    TEMPORARY_SHUTTLE_LOSS = "TEMPORARY_SHUTTLE_LOSS"
    BASELINE_REACQUISITION = "BASELINE_REACQUISITION"
    SEARCHING_FOR_COURT = "SEARCHING_FOR_COURT"
    CALIBRATING = "CALIBRATING"
    COURT_READY = "COURT_READY"
    TRACKING_SHUTTLE = "TRACKING_SHUTTLE"
    LANDING_ANALYSIS = "LANDING_ANALYSIS"
    IN = "IN"
    OUT = "OUT"
    UNKNOWN = "UNKNOWN"


class LiveFrameAnalyzer:
    """Analyze sequential camera frames and recalibrate geometry continuously."""

    def __init__(self, court_analyzer, candidate_extractor, tracker,
                 landing_estimator, decision_maker, static_mask_builder,
                 stable_frames=3, history_frames=60, track_confidence=0.45,
                 decision_confidence=0.65, lost_frame_limit=6,
                 baseline_validator=None, baseline_decision_maker=None, baseline_loss_limit=8):
        self.court_analyzer = court_analyzer
        self.candidate_extractor = candidate_extractor
        self.tracker = tracker
        self.landing_estimator = landing_estimator
        self.decision_maker = decision_maker
        self.static_mask_builder = static_mask_builder
        self.baseline_validator = baseline_validator
        self.baseline_decision_maker = baseline_decision_maker
        self.baseline_loss_limit = max(2, int(baseline_loss_limit))
        self.stable_frames = max(2, int(stable_frames))
        self.history_frames = max(8, int(history_frames))
        self.track_confidence_threshold = float(track_confidence)
        self.decision_confidence_threshold = float(decision_confidence)
        self.lost_frame_limit = max(1, int(lost_frame_limit))
        self.state = AnalysisState.SEARCHING_FOR_COURT
        self.frame_index = 0
        self.geometry = None
        self._anchor_polygon = None
        self._stable_count = 0
        self._previous_gray = None
        self._previous_previous_gray = None
        self._candidate_frames = deque(maxlen=self.history_frames)
        self._candidate_timestamps = deque(maxlen=self.history_frames)
        self._trajectory = []
        self._tracking_confidence = 0.0
        self._active_track = False
        self._lost_frames = 0
        self._reset_after_verdict = False
        self._timestamps = deque(maxlen=12)
        self._last_verdict = None
        self._calibration_valid = False
        self._current_geometry_valid = False
        self._current_court_confidence = 0.0
        self._current_court_reason = "court_not_observed"
        self._calibration_status = "searching"
        self._candidate_polygon = None
        self._candidate_polygon_count = 0
        self._last_transition = {
            "from": None, "to": self.state.value, "at": None,
            "at_utc": datetime.now(timezone.utc).isoformat(),
            "reason": "session_started", "court_confidence": 0.0,
            "geometry_confidence": 0.0, "geometry_valid": False,
            "calibration_valid": False, "frame_index": self.frame_index,
        }
        self._last_frame_timestamp = None
        self._last_candidate_frame = None
        self._max_missed_frame_gap = 0
        self._missed_frame_gap = 0
        self._track_reset_count = 0
        self._frame_metrics = {}
        self._session_id = uuid.uuid4().hex
        self._track_sequence = 0
        self._active_track_id = None
        self._baseline_config = None
        self._baseline_locked = False
        self._baseline_valid = False
        self._baseline_confidence = 0.0
        self._baseline_status = "not_configured"
        self._baseline_stable_count = 0
        self._baseline_missed_frames = 0
        self._baseline_machine = BaselineLock(self.stable_frames, self.baseline_loss_limit)

    def configure_baseline(self, config):
        """Accept explicit line endpoints plus the court-side direction from the client."""
        if self.baseline_validator is None or not isinstance(config, dict):
            return False
        points = config.get("points")
        if config.get("type") != "baseline_calibration" or config.get("court_type") not in ("singles", "doubles"):
            return False
        if not isinstance(points, (list, tuple)) or len(points) != 3:
            return False
        try:
            values = [(float(point[0]), float(point[1])) for point in points]
            if any(not math.isfinite(v) for point in values for v in point):
                return False
            if any(x < 0 or x > 1 or y < 0 or y > 1 for x, y in values):
                return False
            if math.dist(values[0], values[1]) < 0.15:
                return False
            if abs((values[1][0] - values[0][0]) * (values[2][1] - values[0][1])
                   - (values[1][1] - values[0][1]) * (values[2][0] - values[0][0])) < 0.005:
                return False
        except (TypeError, ValueError, IndexError):
            return False
        self._baseline_config = {"points": values, "court_type": config["court_type"]}
        self._baseline_locked = False
        self._baseline_valid = False
        self._baseline_status = "confirming_user_marked_line"
        self._baseline_stable_count = 0
        self._baseline_missed_frames = 0
        if config.get("previously_locked") is True:
            self._baseline_machine.restore_locked()
            self._baseline_locked = True
            self._baseline_status = "retained_waiting_for_frame_revalidation"
        else:
            self._baseline_machine.configure()
        self._candidate_polygon = None
        self._candidate_polygon_count = 0
        self._reset_tracking()
        initial_state = AnalysisState.BASELINE_LOCKED if self._baseline_locked else AnalysisState.BASELINE_CONFIRMATION
        reason = "previous_baseline_retained_waiting_for_frame_revalidation" if self._baseline_locked else "user_confirmed_baseline_candidate"
        self._set_state(initial_state, reason)
        return True

    def _set_state(self, state, reason, confidence=None, timestamp=None):
        state = AnalysisState(state)
        if state != self.state:
            self._log_transition(self.state, state, reason, confidence, timestamp)
        self.state = state

    def _log_transition(self, previous, next_state, reason, confidence=None, timestamp=None, event="state"):
        court_confidence = self._current_court_confidence if confidence is None else float(confidence)
        geometry_valid = bool(self._baseline_valid) if self._baseline_config is not None else bool(self._current_geometry_valid)
        calibration_valid = bool(self._baseline_locked) if self._baseline_config is not None else bool(self._calibration_valid)
        calibration_status = self._baseline_status if self._baseline_config is not None else self._calibration_status
        record = {
            "event": event,
            "previous_state": previous.value if isinstance(previous, AnalysisState) else str(previous),
            "next_state": next_state.value if isinstance(next_state, AnalysisState) else str(next_state),
            "reason": str(reason),
            "timestamp": float(timestamp if timestamp is not None else time.monotonic()),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "court_confidence": round(court_confidence, 3),
            "geometry_valid": geometry_valid,
            "calibration_valid": calibration_valid,
            "calibration_status": calibration_status,
            "baseline_valid": bool(getattr(self, "_baseline_valid", False)),
            "frame_index": self.frame_index,
        }
        self._last_transition = {
            "from": record["previous_state"],
            "to": record["next_state"],
            "at": record["timestamp"],
            "at_utc": record["timestamp_utc"],
            "reason": record["reason"],
            "court_confidence": record["court_confidence"],
            "geometry_confidence": round(float((self.geometry or {}).get("confidence", 0.0)), 3),
            "geometry_valid": record["geometry_valid"],
            "calibration_valid": record["calibration_valid"],
            "calibration_status": record["calibration_status"],
            "baseline_valid": record["baseline_valid"],
            "frame_index": self.frame_index,
        }
        logger.info("live_state_transition %s", json.dumps(record, separators=(",", ":")))

    def _baseline_snapshot(self, decision=None, reason=None):
        return {
            "mode": "baseline_live",
            "state": self.state.value,
            "frame_index": self.frame_index,
            "geometry_valid": self._baseline_valid,
            "calibration_valid": self._baseline_locked,
            "baseline_valid": self._baseline_valid,
            "baseline_locked": self._baseline_locked,
            "baseline_confidence": round(self._baseline_confidence, 3),
            "baseline_status": self._baseline_status,
            "court_type": (self._baseline_config or {}).get("court_type"),
            "baseline": {"points": (self._baseline_config or {}).get("points"), "locked": self._baseline_locked,
                         "valid": self._baseline_valid, "confidence": round(self._baseline_confidence, 3),
                         "status": self._baseline_status},
            "tracking_active": self._active_track,
            "tracked_points": len(self._trajectory),
            "tracking_confidence": round(float(self._tracking_confidence), 3),
            "trajectory": list(self._trajectory[-30:]),
            "track_id": self._active_track_id,
            "decision": decision or {"result": "UNKNOWN", "reason": reason or "awaiting_baseline_or_landing"},
            "state_transition": dict(self._last_transition),
            **self._frame_metrics,
        }

    def _process_baseline_frame(self, frame, now):
        height, width = frame.shape[:2]
        observation = self.baseline_validator(frame, self._baseline_config)
        self._frame_metrics["court_inference_ms"] = round(float(observation.get("processing_ms", 0.0)), 2)
        self._current_court_confidence = float(observation.get("confidence", 0.0))
        self._baseline_confidence = self._current_court_confidence
        self._baseline_valid = bool(observation.get("valid"))
        baseline_lock = self._baseline_machine.observe(self._baseline_valid, str(observation.get("reason") or ""))
        was_locked = self._baseline_locked
        self._baseline_locked = baseline_lock["locked"]
        self._baseline_status = baseline_lock["status"]
        self._baseline_stable_count = baseline_lock["stable_count"]
        self._baseline_missed_frames = baseline_lock["missed_frames"]
        lines = observation.get("lines", [])

        if not self._baseline_valid:
            self._current_geometry_valid = False
            if self._active_track or self._candidate_frames or self._previous_gray is not None:
                self._reset_tracking()
            if self._baseline_locked and self._baseline_machine.state == "BASELINE_LOCKED":
                self._baseline_status = self._baseline_machine.status
                self._set_state(AnalysisState.BASELINE_LOCKED, "baseline_temporarily_unverified", self._baseline_confidence, now)
            elif self._baseline_machine.state == "BASELINE_REACQUISITION":
                self._set_state(AnalysisState.BASELINE_REACQUISITION, "baseline_not_visible_for_configured_frames", self._baseline_confidence, now)
            else:
                self._set_state(AnalysisState.BASELINE_CONFIRMATION, self._baseline_status, self._baseline_confidence, now)
            return self._baseline_snapshot(reason=self._baseline_status)

        if not self._baseline_locked:
            next_state = AnalysisState.BASELINE_REACQUISITION if self._baseline_machine.state == "BASELINE_REACQUISITION" else AnalysisState.BASELINE_CONFIRMATION
            self._set_state(next_state, "marked_baseline_supported", self._baseline_confidence, now)
            return self._baseline_snapshot(reason="baseline_confirmation_in_progress")
        if not was_locked:
            self._set_state(AnalysisState.BASELINE_LOCKED, "baseline_supported_across_consistent_frames", self._baseline_confidence, now)
            self._set_state(AnalysisState.LIVE_LINE_JUDGE, "baseline_lock_confirmed_live_judging_started", self._baseline_confidence, now)

        if self._reset_after_verdict:
            self._reset_tracking()
            self._reset_after_verdict = False
            self._last_verdict = None

        stage_started = time.perf_counter()
        static_mask = self.static_mask_builder(frame, lines)
        gray, candidates = self.candidate_extractor(frame, self._previous_gray, self._previous_previous_gray, static_line_mask=static_mask)
        self._frame_metrics["candidate_inference_ms"] = round((time.perf_counter() - stage_started) * 1000.0, 2)
        self._frame_metrics["detections_per_frame"] = len(candidates)
        self._frame_metrics["detection_confidence"] = round(max((float(c.get("score", 0.0)) for c in candidates), default=0.0), 3)
        self._candidate_frames.append(candidates)
        self._candidate_timestamps.append(now)
        self._previous_previous_gray = self._previous_gray
        self._previous_gray = gray
        self._lost_frames = self._lost_frames + 1 if not candidates else 0

        stage_started = time.perf_counter()
        self._trajectory, self._tracking_confidence = self.tracker(list(self._candidate_frames))
        self._frame_metrics["tracker_inference_ms"] = round((time.perf_counter() - stage_started) * 1000.0, 2)
        if (len(self._trajectory) >= 3 and self._tracking_confidence >= self.track_confidence_threshold
                and (bool(candidates) or self._lost_frames < self.lost_frame_limit)):
            if not self._active_track:
                self._track_sequence += 1
                self._active_track_id = f"{self._session_id}:{self._track_sequence}"
            self._active_track = True
            if not candidates:
                self._set_state(AnalysisState.TEMPORARY_SHUTTLE_LOSS, "validated_track_has_short_detection_gap", self._baseline_confidence, now)
                return self._baseline_snapshot(reason="waiting_for_shuttle_reacquisition")
            self._set_state(AnalysisState.SHUTTLE_TRACKING, "shuttle_track_confident_baseline_locked", self._baseline_confidence, now)
            intervals = [b - a for a, b in zip(list(self._timestamps)[-12:-1], list(self._timestamps)[-11:]) if b > a]
            fps = 1.0 / float(np.median(intervals)) if intervals else 30.0
            landing = self.landing_estimator(self._trajectory, fps, None)
            if landing is not None:
                self._set_state(AnalysisState.LANDING_EVALUATION, "landing_candidate_detected", self._baseline_confidence, now)
                decision = (self.baseline_decision_maker(landing, self._baseline_config, width, height, self._tracking_confidence)
                            if self.baseline_decision_maker else {"result": "UNKNOWN", "inside": None, "reason": "baseline_decision_unavailable"})
                result = decision.get("result") if decision.get("result") in ("IN", "OUT") else "UNKNOWN"
                self._set_state(AnalysisState(result), decision.get("reason", "baseline_landing_evaluated"), self._baseline_confidence, now)
                decision = {**decision, "result": result,
                            "confidence": round(min(self._baseline_confidence, self._tracking_confidence), 3),
                            "event_id": f"{self._active_track_id or self._session_id}:landing",
                            "track_id": self._active_track_id}
                self._reset_after_verdict = True
                return self._baseline_snapshot(decision=decision)
        elif self._active_track and not candidates and self._lost_frames < self.lost_frame_limit:
            self._set_state(AnalysisState.TEMPORARY_SHUTTLE_LOSS, "shuttle_temporarily_missing_baseline_retained", self._baseline_confidence, now)
            return self._baseline_snapshot(reason="waiting_for_shuttle_reacquisition")
        elif self._active_track and self._lost_frames >= self.lost_frame_limit:
            self._reset_tracking()
            self._set_state(AnalysisState.LIVE_LINE_JUDGE, "shuttle_track_expired_no_landing_verdict", self._baseline_confidence, now)
            return self._baseline_snapshot(reason="shuttle_lost_without_landing_evidence")

        if not self._active_track:
            self._set_state(AnalysisState.LIVE_LINE_JUDGE, "baseline_locked_waiting_for_shuttle", self._baseline_confidence, now)
        return self._baseline_snapshot(reason="tracking_shuttle" if self._active_track else "baseline_locked_waiting_for_shuttle")

    def _set_calibration_status(self, status, reason, timestamp=None):
        if status == self._calibration_status:
            return
        self._calibration_status = status
        self._log_transition(self.state, self.state, reason, timestamp=timestamp, event="calibration_status")

    @staticmethod
    def _normalize_polygon(points):
        polygon = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if len(polygon) != 4 or not np.isfinite(polygon).all():
            return None
        hull = cv2.convexHull(polygon).reshape(-1, 2)
        if len(hull) != 4:
            return None
        center = hull.mean(axis=0)
        angles = np.arctan2(hull[:, 1] - center[1], hull[:, 0] - center[0])
        hull = hull[np.argsort(angles)]
        start = int(np.argmin(hull[:, 0] + hull[:, 1]))
        return np.roll(hull, -start, axis=0)

    def _reset_tracking(self):
        self._track_reset_count += 1
        self._last_candidate_frame = None
        self._missed_frame_gap = 0
        self._previous_gray = None
        self._previous_previous_gray = None
        self._candidate_frames.clear()
        self._candidate_timestamps.clear()
        self._trajectory = []
        self._tracking_confidence = 0.0
        self._active_track = False
        self._active_track_id = None
        self._lost_frames = 0

    def _geometry_payload(self):
        if self.geometry is None:
            return None
        diagnostics = dict(self.geometry.get("diagnostics", {}))
        diagnostics.update({
            "current_detection_reason": self._current_court_reason,
            "calibration_status": self._calibration_status,
        })
        return {
            **self.geometry,
            "ready": bool(self._calibration_valid and self._current_geometry_valid),
            "calibration_valid": self._calibration_valid,
            "current_frame_valid": self._current_geometry_valid,
            "calibration_status": self._calibration_status,
            "observed_confidence": round(self._current_court_confidence, 3),
            "diagnostics": diagnostics,
        }

    def _geometry_changed(self, reference, current, frame_shape):
        """Detect meaningful pose changes using translation and shape cues.

        Mean corner displacement alone can hide camera rotation, scale, or
        perspective changes when opposite corners move in different ways.
        Keep calibration invalidation aligned with the previous analyzer's
        tested material-change thresholds.
        """
        if reference is None or current is None:
            return True
        reference = np.asarray(reference, dtype=np.float32)
        current = np.asarray(current, dtype=np.float32)
        if reference.shape != (4, 2) or current.shape != (4, 2):
            return True
        height, width = frame_shape[:2]
        diagonal = max(1.0, math.hypot(width, height))
        ref_center, cur_center = np.mean(reference, axis=0), np.mean(current, axis=0)
        translation = float(np.linalg.norm(cur_center - ref_center)) / diagonal
        ref_edges = np.roll(reference, -1, axis=0) - reference
        cur_edges = np.roll(current, -1, axis=0) - current
        ref_lengths = np.linalg.norm(ref_edges, axis=1)
        cur_lengths = np.linalg.norm(cur_edges, axis=1)
        if np.min(ref_lengths) < 1 or np.min(cur_lengths) < 1:
            return True
        scale_ratios = cur_lengths / ref_lengths
        scale_change = float(np.max(np.abs(scale_ratios - 1.0)))
        ref_angle = math.atan2(float(ref_edges[0, 1]), float(ref_edges[0, 0]))
        cur_angle = math.atan2(float(cur_edges[0, 1]), float(cur_edges[0, 0]))
        angle_delta = abs((cur_angle - ref_angle + math.pi) % (2.0 * math.pi) - math.pi)
        rotation = cur_angle - ref_angle
        cosine, sine = math.cos(rotation), math.sin(rotation)
        rotation_matrix = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float32)
        ref_centered, cur_centered = reference - ref_center, current - cur_center
        aligned = ref_centered @ rotation_matrix.T * float(np.mean(scale_ratios))
        shape_change = float(np.sqrt(np.mean(np.sum((cur_centered - aligned) ** 2, axis=1)))) / diagonal
        return (
            translation > 0.02
            or angle_delta > math.radians(2.0)
            or scale_change > 0.05
            or shape_change > 0.02
        )

    def _reject_geometry_observation(self, reason, now):
        self._current_geometry_valid = False
        self._current_court_reason = str(reason)
        self._candidate_polygon = None
        self._candidate_polygon_count = 0
        if not self._calibration_valid:
            self._set_calibration_status("searching", reason, now)
            self._set_state(AnalysisState.SEARCHING_FOR_COURT, reason, self._current_court_confidence, now)
            return self._snapshot(reason=reason)

        # Keep the accepted court calibration, but stop this shuttle trajectory
        # because its image-coordinate history crosses an unobserved frame.
        if self._active_track or self._candidate_frames or self._previous_gray is not None:
            self._reset_tracking()
        self._set_calibration_status("temporarily_unobserved", reason, now)
        self._set_state(AnalysisState.COURT_READY, reason, self.geometry.get("confidence", 0.0), now)
        return self._snapshot(reason=reason)

    def _snapshot(self, decision=None, landing=None, reason=None):
        decision = decision or {"result": "UNKNOWN", "reason": reason or "no_landing_decision"}
        return {
            "state": self.state.value,
            "frame_index": self.frame_index,
            "geometry_confidence": round(self._current_court_confidence, 3),
            "court_confidence": round(self._current_court_confidence, 3),
            "geometry_valid": self._current_geometry_valid,
            "calibration_valid": self._calibration_valid,
            "calibration_status": self._calibration_status,
            "court_detection_reason": self._current_court_reason,
            "tracking_active": self._active_track,
            "geometry": self._geometry_payload(),
            "candidate_points": sum(len(items) for items in self._candidate_frames),
            "tracked_points": len(self._trajectory),
            "tracking_confidence": round(float(self._tracking_confidence), 3),
            "trajectory": list(self._trajectory[-30:]),
            "track_id": self._active_track_id,
            "estimated_landing_point": landing,
            "decision": decision,
            "state_transition": dict(self._last_transition),
            **self._frame_metrics,
        }

    def _emit_verdict(self, decision, landing):
        result = decision.get("result", "UNKNOWN")
        if result not in ("IN", "OUT"):
            result = "UNKNOWN"
        self._set_state(result, "landing_decision", timestamp=self._timestamps[-1] if self._timestamps else None)
        event_id = f"{self._active_track_id or self._session_id}:landing"
        decision = {**decision, "result": result,
                    "confidence": round(min(float(self.geometry["confidence"]), self._tracking_confidence), 3),
                    "event_id": event_id, "track_id": self._active_track_id}
        if landing is not None:
            landing_frame = landing.get("frame")
            if isinstance(landing_frame, int) and 0 <= landing_frame < len(self._candidate_timestamps):
                event_at = list(self._candidate_timestamps)[landing_frame]
                self._frame_metrics["decision_latency_ms"] = round(max(0.0, (self._timestamps[-1] - event_at) * 1000.0), 2)
            else:
                self._frame_metrics["decision_latency_ms"] = None
        else:
            self._frame_metrics["decision_latency_ms"] = None
        self._last_verdict = decision
        self._reset_after_verdict = True
        return self._snapshot(decision, landing)

    def process_frame(self, frame, timestamp=None):
        """Measure and consume one BGR frame; metrics are per received frame."""
        now = float(timestamp if timestamp is not None else time.monotonic())
        started = time.perf_counter()
        interval_ms = None if self._last_frame_timestamp is None else max(0.0, (now - self._last_frame_timestamp) * 1000.0)
        self._last_frame_timestamp = now
        self._frame_metrics = {"frame_interval_ms": round(interval_ms, 2) if interval_ms is not None else None,
                               "court_inference_ms": None, "candidate_inference_ms": None,
                               "tracker_inference_ms": None, "processing_ms": None,
                               "detections_per_frame": 0, "detection_confidence": 0.0,
                               "missed_frame_gap": 0, "max_missed_frame_gap": self._max_missed_frame_gap,
                               "track_resets": self._track_reset_count,
                               "decision_latency_ms": None}
        result = self._process_frame(frame, now)
        self._frame_metrics["processing_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
        self._frame_metrics["track_resets"] = self._track_reset_count
        result.update(self._frame_metrics)
        return result

    def _process_frame(self, frame, timestamp=None):
        """Run the deterministic live analysis state machine for one frame."""
        self.frame_index += 1
        now = float(timestamp if timestamp is not None else time.monotonic())
        self._timestamps.append(now)
        if self._baseline_config is not None:
            return self._process_baseline_frame(frame, now)
        stage_started = time.perf_counter()
        court = self.court_analyzer(frame)
        self._frame_metrics["court_inference_ms"] = round((time.perf_counter() - stage_started) * 1000.0, 2)
        region = court.get("region") if court else None
        try:
            confidence = float((court or {}).get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        self._current_court_confidence = confidence if math.isfinite(confidence) else 0.0
        self._current_court_reason = str((court or {}).get("reason") or (court or {}).get("mode") or "court_geometry_not_observed")
        reliable = bool(court and court.get("reliable") and region is not None
                        and self._current_court_confidence >= self.decision_confidence_threshold)
        if not reliable:
            return self._reject_geometry_observation(self._current_court_reason, now)

        polygon = self._normalize_polygon(region)
        if polygon is None:
            return self._reject_geometry_observation("invalid_playable_region", now)

        if not self._calibration_valid:
            if self._candidate_polygon is None:
                self._candidate_polygon = polygon.copy()
                self._candidate_polygon_count = 1
                reason = "court_geometry_detected"
            elif self._geometry_changed(self._candidate_polygon, polygon, frame.shape):
                self._candidate_polygon = polygon.copy()
                self._candidate_polygon_count = 1
                reason = "court_geometry_candidate_changed"
                self._log_transition(self.state, self.state, reason, self._current_court_confidence, now, event="calibration_candidate")
            else:
                self._candidate_polygon_count += 1
                reason = "geometry_stabilizing"
            self._stable_count = self._candidate_polygon_count
            self.geometry = {
                "region": polygon.astype(float).tolist(),
                "mode": court.get("mode", "court_geometry"),
                "confidence": self._current_court_confidence,
                "diagnostics": court.get("diagnostics", {}),
            }
            self._current_geometry_valid = True
            self._current_court_reason = reason
            self._set_calibration_status("stabilizing", reason, now)
            self._set_state(AnalysisState.CALIBRATING, reason, self._current_court_confidence, now)
            if self._candidate_polygon_count < self.stable_frames:
                return self._snapshot(reason=reason)

            self._anchor_polygon = polygon.copy()
            self._calibration_valid = True
            self._stable_count = self.stable_frames
            self._candidate_polygon = None
            self._candidate_polygon_count = 0
            self._calibration_status = "valid"
            self._log_transition(self.state, self.state, "court_geometry_stable", self._current_court_confidence, now, event="calibration_status")
            self._set_state(AnalysisState.COURT_READY, "court_geometry_stable", self._current_court_confidence, now)
        else:
            moved = self._geometry_changed(self._anchor_polygon, polygon, frame.shape)
            if moved:
                if self._candidate_polygon is None or self._geometry_changed(self._candidate_polygon, polygon, frame.shape):
                    self._candidate_polygon = polygon.copy()
                    self._candidate_polygon_count = 1
                else:
                    self._candidate_polygon_count += 1
                self._current_geometry_valid = False
                self._current_court_reason = "court_geometry_change_candidate"
                self._set_calibration_status("change_candidate", "court_geometry_change_candidate", now)
                if self._active_track or self._candidate_frames or self._previous_gray is not None:
                    self._reset_tracking()
                self._set_state(AnalysisState.COURT_READY, "court_geometry_change_candidate", self.geometry.get("confidence", 0.0), now)
                if self._candidate_polygon_count < self.stable_frames:
                    return self._snapshot(reason="court_geometry_change_candidate")

                self._anchor_polygon = polygon.copy()
                self._candidate_polygon = None
                self._candidate_polygon_count = 0
                self._stable_count = self.stable_frames
                self.geometry = {
                    "region": polygon.astype(float).tolist(),
                    "mode": court.get("mode", "court_geometry"),
                    "confidence": self._current_court_confidence,
                    "diagnostics": court.get("diagnostics", {}),
                }
                self._current_geometry_valid = True
                self._current_court_reason = "court_geometry_replaced_after_consensus"
                self._calibration_status = "valid"
                self._log_transition(self.state, self.state, "court_geometry_replaced_after_consensus", self._current_court_confidence, now, event="calibration_status")
            else:
                self._current_geometry_valid = True
                if self._calibration_status == "change_candidate":
                    self._candidate_polygon = None
                    self._candidate_polygon_count = 0
                    self._set_calibration_status("valid", "previous_calibration_revalidated", now)
                elif self._calibration_status == "temporarily_unobserved":
                    self._set_calibration_status("valid", "calibrated_geometry_reacquired", now)
                self._current_court_reason = "calibrated_geometry_reacquired"
                self._stable_count = self.stable_frames
                self.geometry = {
                    "region": polygon.astype(float).tolist(),
                    "mode": court.get("mode", "court_geometry"),
                    "confidence": self._current_court_confidence,
                    "diagnostics": court.get("diagnostics", {}),
                }


        if self._reset_after_verdict:
            self._reset_tracking()
            self._reset_after_verdict = False
            self._last_verdict = None
        if self.state in (AnalysisState.SEARCHING_FOR_COURT, AnalysisState.CALIBRATING,
                          AnalysisState.IN, AnalysisState.OUT, AnalysisState.UNKNOWN):
            self._set_state(AnalysisState.COURT_READY, "court_geometry_stable", self._current_court_confidence, now)

        stage_started = time.perf_counter()
        static_mask = self.static_mask_builder(frame, court.get("lines", []))
        gray, candidates = self.candidate_extractor(
            frame,
            self._previous_gray,
            self._previous_previous_gray,
            static_line_mask=static_mask,
        )
        self._frame_metrics["candidate_inference_ms"] = round((time.perf_counter() - stage_started) * 1000.0, 2)
        self._frame_metrics["detections_per_frame"] = len(candidates)
        self._frame_metrics["detection_confidence"] = round(max((float(c.get("score", 0.0)) for c in candidates), default=0.0), 3)
        if candidates:
            self._last_candidate_frame = self.frame_index
            self._missed_frame_gap = 0
        elif self._last_candidate_frame is not None:
            self._missed_frame_gap = self.frame_index - self._last_candidate_frame
            self._max_missed_frame_gap = max(self._max_missed_frame_gap, self._missed_frame_gap)
        self._frame_metrics["missed_frame_gap"] = self._missed_frame_gap
        self._frame_metrics["max_missed_frame_gap"] = self._max_missed_frame_gap
        self._candidate_frames.append(candidates)
        self._candidate_timestamps.append(now)
        self._previous_previous_gray = self._previous_gray
        self._previous_gray = gray
        self._lost_frames = self._lost_frames + 1 if not candidates else 0

        stage_started = time.perf_counter()
        self._trajectory, self._tracking_confidence = self.tracker(list(self._candidate_frames))
        self._frame_metrics["tracker_inference_ms"] = round((time.perf_counter() - stage_started) * 1000.0, 2)
        if len(self._trajectory) >= 3 and self._tracking_confidence >= self.track_confidence_threshold:
            if not self._active_track:
                self._track_sequence += 1
                self._active_track_id = f"{self._session_id}:{self._track_sequence}"
            self._active_track = True
            self._set_state(AnalysisState.TRACKING_SHUTTLE, "shuttle_track_confident", self.geometry["confidence"], now)
            times = list(self._timestamps)
            intervals = [b - a for a, b in zip(times[:-1], times[1:]) if b > a]
            fps = 1.0 / float(np.median(intervals)) if intervals else 30.0
            landing = self.landing_estimator(self._trajectory, fps, self.geometry["region"])
            if landing is not None:
                self._set_state(AnalysisState.LANDING_ANALYSIS, "landing_candidate_detected", self.geometry["confidence"], now)
                if (self._tracking_confidence < self.decision_confidence_threshold
                        or self.geometry["confidence"] < self.decision_confidence_threshold
                        or landing.get("method") not in ("downward_phase_bounce", "downward_phase_settle")):
                    decision = {"result": "UNKNOWN", "inside": None,
                                "reason": "landing_or_geometry_confidence_insufficient"}
                else:
                    decision = self.decision_maker(landing, self.geometry["region"])
                return self._emit_verdict(decision, landing)
        if self._active_track and self._lost_frames >= self.lost_frame_limit:
            self._set_state(AnalysisState.LANDING_ANALYSIS, "shuttle_track_lost", self.geometry["confidence"], now)
            return self._emit_verdict({"result": "UNKNOWN", "inside": None,
                                       "reason": "landing_not_confidently_observed"}, None)
        if not self._active_track:
            self._set_state(AnalysisState.COURT_READY, "monitoring_for_shuttle", self.geometry["confidence"], now)

        return self._snapshot(reason="tracking_shuttle" if self.state == AnalysisState.TRACKING_SHUTTLE else "monitoring_for_shuttle")
