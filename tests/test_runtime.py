import unittest
import math

import numpy as np

from main import _index_live_frame_update, _transition_score, check_landing_inside_court, choose_landing_point, track_shuttle
from runtime_analyzer import AnalysisState, LiveFrameAnalyzer
from baseline_geometry import classify_baseline_landing


BASE_POLYGON = [(10, 10), (90, 10), (90, 90), (10, 90)]


def reliable_court(region=None, mode="geometry_polygon", confidence=0.9):
    return {"region": region or BASE_POLYGON, "mode": mode, "confidence": confidence,
            "reliable": True, "diagnostics": {"reliable": True}, "lines": []}


def make_analyzer(court_provider=None, candidate_provider=None, landing_provider=None,
                  stable_frames=3, lost_frame_limit=4, baseline_validator=None,
                  tracker_provider=None):
    court_calls = {"count": 0}
    candidate_calls = {"count": 0}

    def court(frame):
        court_calls["count"] += 1
        return (court_provider(court_calls["count"]) if court_provider else reliable_court())

    def extract(frame, previous, previous_previous, static_line_mask=None):
        candidate_calls["count"] += 1
        candidates = candidate_provider(candidate_calls["count"]) if candidate_provider else []
        return frame[:, :, 0].copy(), candidates

    def landing(trajectory, fps, region):
        if landing_provider:
            return landing_provider(trajectory, fps, region)
        return choose_landing_point(trajectory, fps, region)

    analyzer = LiveFrameAnalyzer(
        court_analyzer=court,
        candidate_extractor=extract,
        tracker=tracker_provider or track_shuttle,
        landing_estimator=landing,
        decision_maker=lambda point, region: check_landing_inside_court(point, region, margin=3),
        static_mask_builder=lambda frame, lines: None,
        stable_frames=stable_frames,
        lost_frame_limit=lost_frame_limit,
        baseline_validator=baseline_validator,
        baseline_decision_maker=classify_baseline_landing,
    )
    return analyzer


def frame():
    return np.zeros((100, 100, 3), dtype=np.uint8)


def moving_candidate(step):
    return {"x": 20 + step * 12.0, "y": 15 + step * 18.0,
            "score": 0.95, "motion_ratio": 0.9, "yellow_ratio": 0.2,
            "appearance_ratio": 0.8, "area": 18, "w": 5, "h": 5}


class LiveRuntimeStateMachineTests(unittest.TestCase):
    def test_wire_frame_index_stays_aligned_across_rejected_jpeg_messages(self):
        rejected = _index_live_frame_update({"state": "LIVE_LINE_JUDGE", "decision": {"result": "UNKNOWN"}}, 2)
        analyzed = _index_live_frame_update({"state": "SHUTTLE_TRACKING", "frame_index": 1}, 3)
        self.assertEqual(rejected["frame_index"], 2)
        self.assertIsNone(rejected["analysis_frame_index"])
        self.assertEqual(analyzed["frame_index"], 3)
        self.assertEqual(analyzed["analysis_frame_index"], 1)

    def test_continuous_calibration_reaches_ready(self):
        analyzer = make_analyzer()
        states = [analyzer.process_frame(frame(), timestamp=i / 30)["state"] for i in range(3)]
        self.assertEqual(states, ["CALIBRATING", "CALIBRATING", "COURT_READY"])

    def test_live_frame_reports_intervals_inference_and_detections(self):
        analyzer = make_analyzer(candidate_provider=lambda _: [moving_candidate(1)])
        for t in (1.0, 1.1, 1.2):
            analyzer.process_frame(frame(), timestamp=t)
        update = analyzer.process_frame(frame(), timestamp=1.3)
        self.assertEqual(update["frame_interval_ms"], 100.0)
        self.assertEqual(update["detections_per_frame"], 1)
        self.assertEqual(update["detection_confidence"], 0.95)
        self.assertIsInstance(update["processing_ms"], float)
        self.assertIsInstance(update["court_inference_ms"], float)

    def test_material_camera_movement_replaces_calibration_after_consensus(self):
        def court(call):
            shift = 35 if call >= 4 else 0
            polygon = [(x + shift, y) for x, y in BASE_POLYGON]
            return reliable_court(polygon)
        analyzer = make_analyzer(court_provider=court)
        for i in range(3):
            analyzer.process_frame(frame(), timestamp=i / 30)
        candidate = analyzer.process_frame(frame(), timestamp=3 / 30)
        self.assertEqual(candidate["state"], "COURT_READY")
        self.assertTrue(candidate["calibration_valid"])
        self.assertFalse(candidate["geometry_valid"])
        self.assertEqual(candidate["decision"]["result"], "UNKNOWN")
        self.assertEqual(analyzer.process_frame(frame(), timestamp=4 / 30)["state"], "COURT_READY")
        replaced = analyzer.process_frame(frame(), timestamp=5 / 30)
        self.assertEqual(replaced["state"], "COURT_READY")
        self.assertTrue(replaced["geometry_valid"])
        self.assertEqual(replaced["calibration_status"], "valid")

    def test_temporary_geometry_loss_preserves_calibration_independent_of_elapsed_time(self):
        def court(call):
            if call in (4, 5):
                return {"reliable": False, "region": None, "mode": "low_confidence_polygon", "confidence": 0.2}
            return reliable_court()
        analyzer = make_analyzer(court_provider=court)
        for i in range(3):
            analyzer.process_frame(frame(), timestamp=i / 30)
        missed = analyzer.process_frame(frame(), timestamp=10.0)
        self.assertEqual(missed["state"], "COURT_READY")
        self.assertTrue(missed["calibration_valid"])
        self.assertFalse(missed["geometry_valid"])
        self.assertFalse(missed["geometry"]["ready"])
        self.assertEqual(missed["decision"]["result"], "UNKNOWN")
        analyzer.process_frame(frame(), timestamp=25.0)
        recovered = analyzer.process_frame(frame(), timestamp=40.0)
        self.assertEqual(recovered["state"], "COURT_READY")
        self.assertTrue(recovered["calibration_valid"])
        self.assertTrue(recovered["geometry_valid"])
        self.assertEqual(recovered["calibration_status"], "valid")

    def test_unobserved_geometry_does_not_destroy_last_calibration_or_issue_call(self):
        def court(call):
            if call >= 4:
                return {"reliable": False, "region": None, "mode": "low_confidence_polygon", "confidence": 0.2}
            return reliable_court()
        analyzer = make_analyzer(court_provider=court)
        for i in range(3):
            analyzer.process_frame(frame(), timestamp=i / 30)
        analyzer.process_frame(frame(), timestamp=0.1)
        update = analyzer.process_frame(frame(), timestamp=100.0)
        self.assertEqual(update["state"], "COURT_READY")
        self.assertIsNotNone(update["geometry"])
        self.assertTrue(update["calibration_valid"])
        self.assertFalse(update["geometry_valid"])
        self.assertEqual(update["decision"]["result"], "UNKNOWN")

    def test_same_geometry_reacquisition_does_not_restart_calibration(self):
        def court(call):
            if call in (4, 5):
                return {"reliable": False, "region": None, "mode": "insufficient_geometry", "confidence": 0}
            return reliable_court()
        analyzer = make_analyzer(court_provider=court)
        for i in range(3):
            analyzer.process_frame(frame(), timestamp=i / 30)
        analyzer.process_frame(frame(), timestamp=0.1)
        first = analyzer.process_frame(frame(), timestamp=1.2)
        self.assertEqual(first["state"], "COURT_READY")
        second = analyzer.process_frame(frame(), timestamp=1.23)
        self.assertEqual(second["state"], "COURT_READY")
        third = analyzer.process_frame(frame(), timestamp=1.26)
        self.assertEqual(third["state"], "COURT_READY")
        self.assertEqual(third["calibration_status"], "valid")

    def test_single_material_polygon_outlier_does_not_replace_calibration(self):
        def court(call):
            shift = 18 if call == 4 else 0
            polygon = [(x + shift, y) for x, y in BASE_POLYGON]
            return reliable_court(polygon)
        analyzer = make_analyzer(court_provider=court)
        for i in range(3):
            analyzer.process_frame(frame(), timestamp=i / 30)
        outlier = analyzer.process_frame(frame(), timestamp=3 / 30)
        self.assertEqual(outlier["state"], "COURT_READY")
        self.assertTrue(outlier["calibration_valid"])
        self.assertFalse(outlier["geometry_valid"])
        recovered = analyzer.process_frame(frame(), timestamp=4 / 30)
        self.assertEqual(recovered["state"], "COURT_READY")
        self.assertTrue(recovered["geometry_valid"])
        self.assertEqual(recovered["calibration_status"], "valid")
        self.assertEqual(recovered["geometry"]["region"], [list(point) for point in BASE_POLYGON])

    def test_temporary_court_miss_ends_track_safely_without_recalibrating(self):
        def court(call):
            if call == 8:
                return {"reliable": False, "region": None, "mode": "low_confidence_polygon", "confidence": 0.2}
            return reliable_court()
        analyzer = make_analyzer(
            court_provider=court,
            candidate_provider=lambda n: [] if n == 4 else [moving_candidate(n)],
        )
        updates = [analyzer.process_frame(frame(), timestamp=i / 30) for i in range(7)]
        self.assertTrue(any(update["state"] == "TRACKING_SHUTTLE" for update in updates))
        miss = analyzer.process_frame(frame(), timestamp=10.0)
        self.assertEqual(miss["state"], "COURT_READY")
        self.assertTrue(miss["calibration_valid"])
        self.assertFalse(miss["geometry_valid"])
        self.assertEqual(miss["decision"]["result"], "UNKNOWN")
        reacquired = analyzer.process_frame(frame(), timestamp=20.0)
        self.assertEqual(reacquired["state"], "COURT_READY")
        self.assertTrue(reacquired["calibration_valid"])
        self.assertTrue(reacquired["geometry_valid"])

    def test_material_geometry_change_replaces_calibration_after_consensus(self):
        def court(call):
            shift = 18 if call >= 4 else 0
            polygon = [(x + shift, y) for x, y in BASE_POLYGON]
            return reliable_court(polygon)
        analyzer = make_analyzer(court_provider=court)
        for i in range(3):
            analyzer.process_frame(frame(), timestamp=i / 30)
        first = analyzer.process_frame(frame(), timestamp=3 / 30)
        second = analyzer.process_frame(frame(), timestamp=4 / 30)
        self.assertEqual(first["state"], "COURT_READY")
        self.assertEqual(second["state"], "COURT_READY")
        self.assertEqual(second["calibration_status"], "change_candidate")
        self.assertFalse(second["geometry_valid"])
        replaced = analyzer.process_frame(frame(), timestamp=5 / 30)
        self.assertEqual(replaced["state"], "COURT_READY")
        self.assertTrue(replaced["geometry_valid"])
        self.assertTrue(replaced["calibration_valid"])
        self.assertEqual(replaced["calibration_status"], "valid")
        self.assertEqual(replaced["geometry"]["region"][0][0], BASE_POLYGON[0][0] + 18)

    def test_rotation_scale_and_perspective_changes_invalidate_calibration(self):
        reference = np.array([[80, 60], [560, 65], [550, 360], [90, 350]], dtype=np.float32)
        center = reference.mean(axis=0)
        angle = math.radians(4)
        rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        changed_polygons = [
            (reference - center) @ rotation.T + center,
            center + (reference - center) * 1.08,
            reference + np.array([[0, 25], [0, 0], [0, 0], [0, 0]], dtype=np.float32),
        ]

        for changed in changed_polygons:
            with self.subTest(changed=changed.tolist()):
                def court(call):
                    polygon = reference if call <= 3 else changed
                    return reliable_court(polygon.tolist())

                analyzer = make_analyzer(court_provider=court)
                test_frame = np.zeros((400, 640, 3), dtype=np.uint8)
                for index in range(3):
                    analyzer.process_frame(test_frame, timestamp=index / 30)
                candidate = analyzer.process_frame(test_frame, timestamp=3 / 30)
                self.assertEqual(candidate["calibration_status"], "change_candidate")
                self.assertFalse(candidate["geometry_valid"])
                analyzer.process_frame(test_frame, timestamp=4 / 30)
                replaced = analyzer.process_frame(test_frame, timestamp=5 / 30)
                self.assertTrue(replaced["geometry_valid"])
                self.assertTrue(replaced["calibration_valid"])

    def test_state_transition_log_has_geometry_calibration_and_frame_context(self):
        analyzer = make_analyzer()
        with self.assertLogs("runtime_analyzer", level="INFO") as captured:
            for i in range(3):
                analyzer.process_frame(frame(), timestamp=i / 30)
        self.assertTrue(any('"previous_state":"CALIBRATING"' in line for line in captured.output))
        self.assertTrue(any('"next_state":"COURT_READY"' in line for line in captured.output))
        self.assertTrue(any('"court_confidence":0.9' in line for line in captured.output))
        self.assertTrue(any('"geometry_valid":true' in line for line in captured.output))
        self.assertTrue(any('"calibration_valid":true' in line for line in captured.output))
        self.assertTrue(any('"frame_index":3' in line for line in captured.output))

    def test_user_marked_baseline_locks_without_calling_full_court_detector(self):
        def no_full_court(_):
            raise AssertionError("baseline mode must not run full-court calibration")
        analyzer = make_analyzer(court_provider=no_full_court,
                                 baseline_validator=lambda frame, config: {"valid": True, "confidence": 0.9, "lines": []})
        self.assertTrue(analyzer.configure_baseline({"type": "baseline_calibration", "court_type": "singles",
            "points": [(0.1, 0.5), (0.9, 0.5), (0.5, 0.75)]}))
        updates = [analyzer.process_frame(frame(), timestamp=i / 30) for i in range(3)]
        self.assertEqual(updates[-1]["state"], "LIVE_LINE_JUDGE")
        self.assertTrue(updates[-1]["baseline_locked"])
        self.assertEqual(updates[-1]["calibration_valid"], True)

    def test_baseline_detector_blip_and_temporary_shuttle_loss_do_not_recalibrate(self):
        detector_calls = {"count": 0}
        def baseline(_frame, _config):
            detector_calls["count"] += 1
            valid = detector_calls["count"] != 5
            return {"valid": valid, "confidence": 0.9 if valid else 0.0,
                    "reason": "marked_baseline_supported" if valid else "marked_baseline_not_visible", "lines": []}
        def candidate(number):
            return [moving_candidate(number)] if number <= 5 else []
        def deterministic_tracker(frames):
            path = [{"frame": index, "x": items[0]["x"], "y": items[0]["y"]}
                    for index, items in enumerate(frames) if items]
            return path, 0.9 if len(path) >= 3 else 0.0
        analyzer = make_analyzer(court_provider=lambda _: (_ for _ in ()).throw(AssertionError("full court called")),
            candidate_provider=candidate, landing_provider=lambda *_: None, baseline_validator=baseline,
            tracker_provider=deterministic_tracker)
        analyzer.configure_baseline({"type": "baseline_calibration", "court_type": "singles",
            "points": [(0.1, 0.5), (0.9, 0.5), (0.5, 0.75)]})
        updates = [analyzer.process_frame(frame(), timestamp=i / 30) for i in range(14)]
        self.assertTrue(any(update["state"] == "SHUTTLE_TRACKING" for update in updates))
        self.assertTrue(any(update["state"] == "TEMPORARY_SHUTTLE_LOSS" for update in updates))
        self.assertTrue(all(update["baseline_locked"] for update in updates[2:]))
        self.assertEqual(updates[-1]["state"], "LIVE_LINE_JUDGE")

    def test_baseline_landing_result_returns_to_live_judging_without_recalibration(self):
        def landing(trajectory, _fps, _region):
            return {"frame": 2, "x": 50, "y": 50, "method": "downward_phase_bounce"} if len(trajectory) >= 3 else None
        def deterministic_tracker(frames):
            path = [{"frame": index, "x": items[0]["x"], "y": items[0]["y"]}
                    for index, items in enumerate(frames) if items]
            return path, 0.9 if len(path) >= 3 else 0.0
        analyzer = make_analyzer(court_provider=lambda _: (_ for _ in ()).throw(AssertionError("full court called")),
            candidate_provider=lambda number: [moving_candidate(number)] if number <= 3 else [],
            landing_provider=landing,
            baseline_validator=lambda _frame, _config: {"valid": True, "confidence": 0.9, "lines": []},
            tracker_provider=deterministic_tracker)
        analyzer.configure_baseline({"type": "baseline_calibration", "court_type": "singles",
            "points": [(0.1, 0.5), (0.9, 0.5), (0.5, 0.75)]})
        updates = [analyzer.process_frame(frame(), timestamp=i / 30) for i in range(6)]
        verdict = next(update for update in updates if update["decision"]["result"] == "IN")
        self.assertEqual(verdict["state"], "IN")
        self.assertTrue(verdict["baseline_locked"])
        after = updates[-1]
        self.assertEqual(after["state"], "LIVE_LINE_JUDGE")
        self.assertTrue(after["baseline_locked"])

    def test_transport_reconnect_retains_baseline_but_requires_new_frame_verification(self):
        calls = {"count": 0}
        def baseline(_frame, _config):
            calls["count"] += 1
            valid = calls["count"] > 1
            return {"valid": valid, "confidence": 0.9 if valid else 0.0,
                    "reason": "marked_baseline_supported" if valid else "one_frame_blur", "lines": []}
        analyzer = make_analyzer(baseline_validator=baseline)
        self.assertTrue(analyzer.configure_baseline({"type": "baseline_calibration", "court_type": "singles",
            "previously_locked": True, "points": [(0.1, 0.5), (0.9, 0.5), (0.5, 0.75)]}))
        missed = analyzer.process_frame(frame(), timestamp=1.0)
        self.assertTrue(missed["baseline_locked"])
        self.assertFalse(missed["baseline_valid"])
        restored = analyzer.process_frame(frame(), timestamp=1.1)
        self.assertTrue(restored["baseline_locked"])
        self.assertTrue(restored["baseline_valid"])

    def test_partial_geometry_can_calibrate(self):
        analyzer = make_analyzer(court_provider=lambda _: reliable_court(mode="partial_geometry_polygon", confidence=0.8))
        for i in range(3):
            update = analyzer.process_frame(frame(), timestamp=i / 30)
        self.assertEqual(update["state"], "COURT_READY")
        self.assertEqual(update["geometry"]["mode"], "partial_geometry_polygon")
        self.assertGreaterEqual(update["geometry_confidence"], 0.65)

    def test_stationary_false_track_never_calls(self):
        stationary = {"x": 50.0, "y": 50.0, "score": 0.95, "motion_ratio": 0.8,
                      "yellow_ratio": 0.0, "appearance_ratio": 0.8, "area": 16, "w": 4, "h": 4}
        analyzer = make_analyzer(candidate_provider=lambda _: [dict(stationary)])
        updates = [analyzer.process_frame(frame(), timestamp=i / 30) for i in range(20)]
        self.assertTrue(all(update["decision"]["result"] == "UNKNOWN" for update in updates))
        self.assertNotIn("IN", [u["state"] for u in updates])
        self.assertNotIn("OUT", [u["state"] for u in updates])

    def test_temporal_shuttle_track_is_reported(self):
        analyzer = make_analyzer(candidate_provider=lambda n: [moving_candidate(n)] if 1 <= n <= 7 else [])
        updates = [analyzer.process_frame(frame(), timestamp=i / 30) for i in range(14)]
        tracked = [update for update in updates if update["state"] == "TRACKING_SHUTTLE"]
        self.assertTrue(tracked)
        self.assertGreaterEqual(max(update["tracked_points"] for update in tracked), 3)

    def test_ambiguous_landing_times_out_to_unknown(self):
        analyzer = make_analyzer(
            candidate_provider=lambda n: [moving_candidate(n)] if 1 <= n <= 7 else [],
            landing_provider=lambda trajectory, fps, region: None,
        )
        updates = [analyzer.process_frame(frame(), timestamp=i / 30) for i in range(20)]
        self.assertTrue(any(update["state"] == "TRACKING_SHUTTLE" for update in updates))
        verdicts = [update for update in updates if update["state"] == "UNKNOWN"]
        self.assertTrue(verdicts)
        self.assertEqual(verdicts[-1]["decision"]["result"], "UNKNOWN")

    def test_confident_landing_transitions_through_analysis_to_decision(self):
        landing = {"frame": 5, "x": 50, "y": 50, "method": "downward_phase_bounce"}
        analyzer = make_analyzer(
            candidate_provider=lambda n: [moving_candidate(n)] if 1 <= n <= 7 else [],
            landing_provider=lambda trajectory, fps, region: landing if len(trajectory) >= 5 else None,
        )
        updates = [analyzer.process_frame(frame(), timestamp=i / 30) for i in range(12)]
        verdicts = [update for update in updates if update["state"] == "IN"]
        self.assertTrue(verdicts)
        self.assertEqual(verdicts[0]["decision"]["result"], "IN")
        self.assertEqual(verdicts[0]["decision"]["track_id"], verdicts[0]["track_id"])
        self.assertTrue(verdicts[0]["decision"]["event_id"].endswith(":landing"))
        self.assertGreaterEqual(verdicts[0]["decision_latency_ms"], 0)

    def test_new_trajectory_gets_a_distinct_landing_event_id(self):
        landing = {"frame": 4, "x": 50, "y": 50, "method": "downward_phase_bounce"}
        analyzer = make_analyzer(
            candidate_provider=lambda n: [moving_candidate(n)] if n % 9 < 7 else [],
            landing_provider=lambda trajectory, fps, region: landing if len(trajectory) >= 5 else None,
        )
        updates = [analyzer.process_frame(frame(), timestamp=i / 30) for i in range(40)]
        event_ids = [u["decision"]["event_id"] for u in updates if u["state"] in ("IN", "OUT")]
        self.assertGreaterEqual(len(event_ids), 2)
        self.assertEqual(len(event_ids), len(set(event_ids)))

    def test_tracker_bridges_short_missed_frame_gap(self):
        frames = [[moving_candidate(i)] for i in range(1, 5)] + [[], []] + [[moving_candidate(i)] for i in range(7, 13)]
        trajectory, confidence = track_shuttle(frames)
        self.assertGreaterEqual(confidence, 0.45)
        self.assertTrue(any(b["frame"] - a["frame"] > 1 for a, b in zip(trajectory, trajectory[1:])))

    def test_tracker_rejects_false_trajectory_teleport_after_stationary_object(self):
        stationary = {"x": 40.0, "y": 40.0, "score": 0.9, "motion_ratio": 0.7}
        unrelated = {"x": 500.0, "y": 500.0, "score": 0.99, "motion_ratio": 1.0}
        self.assertLess(_transition_score(stationary, unrelated, stationary, frame_gap=1), -1e8)

    def test_large_gap_cannot_supply_false_landing_reversal(self):
        trajectory = [
            {"frame": 0, "x": 20, "y": 10},
            {"frame": 1, "x": 22, "y": 30},
            {"frame": 2, "x": 24, "y": 60},
            {"frame": 8, "x": 26, "y": 130},
            {"frame": 9, "x": 28, "y": 20},
        ]
        self.assertIsNone(choose_landing_point(trajectory, 30, max_frame_gap=2))


if __name__ == "__main__":
    unittest.main()
