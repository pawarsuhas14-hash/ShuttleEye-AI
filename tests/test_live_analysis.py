"""Regression coverage ported from the former LiveAnalysisSession API."""
import json
import unittest

import cv2
import numpy as np
from fastapi.testclient import TestClient

import main
from runtime_analyzer import LiveFrameAnalyzer


BASE_POLYGON = [(10, 10), (90, 10), (90, 90), (10, 90)]


def make_analyzer(court_provider=None, candidate_provider=None, tracker=None,
                  landing_estimator=None, decision_maker=None):
    court_calls = {"count": 0}
    candidate_calls = {"count": 0}

    def court(frame):
        court_calls["count"] += 1
        if court_provider:
            return court_provider(court_calls["count"])
        return {"region": BASE_POLYGON, "mode": "geometry_polygon", "confidence": 0.9,
                "reliable": True, "diagnostics": {"reliable": True}, "lines": []}

    def extract(frame, previous, previous_previous, static_line_mask=None):
        candidate_calls["count"] += 1
        candidates = candidate_provider(candidate_calls["count"]) if candidate_provider else []
        return frame[:, :, 0].copy(), candidates

    analyzer = LiveFrameAnalyzer(
        court_analyzer=court,
        candidate_extractor=extract,
        tracker=tracker or main.track_shuttle,
        landing_estimator=landing_estimator or main.choose_landing_point,
        decision_maker=decision_maker or (lambda point, region: {"result": "UNKNOWN", "inside": None}),
        static_mask_builder=lambda frame, lines: None,
    )
    return analyzer, court_calls


def frame():
    return np.zeros((100, 100, 3), dtype=np.uint8)


class LegacyLiveAnalysisCompatibilityTests(unittest.TestCase):
    def test_geometry_is_checked_each_frame_while_ready_state_stays_stable(self):
        analyzer, court_calls = make_analyzer()
        updates = [analyzer.process_frame(frame(), timestamp=index / 30) for index in range(6)]
        self.assertEqual(court_calls["count"], 6)
        self.assertTrue(all(update["state"] == "COURT_READY" for update in updates[2:]))
        self.assertTrue(all(update["calibration_valid"] for update in updates[2:]))

    def test_camera_geometry_change_replaces_calibration_only_after_consensus(self):
        def court(call):
            shift = 30 if call >= 4 else 0
            polygon = [(x + shift, y) for x, y in BASE_POLYGON]
            return {"region": polygon, "mode": "geometry_polygon", "confidence": 0.9,
                    "reliable": True, "diagnostics": {"reliable": True}, "lines": []}

        analyzer, _ = make_analyzer(court_provider=court)
        updates = [analyzer.process_frame(frame(), timestamp=index / 30) for index in range(6)]
        self.assertEqual(updates[3]["calibration_status"], "change_candidate")
        self.assertFalse(updates[3]["geometry_valid"])
        self.assertFalse(updates[4]["geometry_valid"])
        self.assertTrue(updates[5]["geometry_valid"])
        self.assertTrue(updates[5]["calibration_valid"])
        self.assertEqual(updates[5]["geometry"]["region"][0][0], BASE_POLYGON[0][0] + 30)
        self.assertTrue(all(update["decision"]["result"] == "UNKNOWN" for update in updates[3:5]))

    def test_low_confidence_track_never_emits_in_or_out(self):
        path = [{"frame": index, "x": 20 + index * 20, "y": 20 + index * 15} for index in range(5)]
        calls = {"count": 0}

        def decide(_point, _region):
            calls["count"] += 1
            return {"result": "IN", "inside": True}

        analyzer, _ = make_analyzer(
            candidate_provider=lambda n: [{"x": 20 + n * 8, "y": 20 + n * 10, "score": 0.95}],
            tracker=lambda _history: (path, 0.2),
            landing_estimator=lambda *_args: {"frame": 2, "x": 50, "y": 60, "method": "downward_phase_bounce"},
            decision_maker=decide,
        )
        updates = [analyzer.process_frame(frame(), timestamp=index / 30) for index in range(10)]
        self.assertEqual(calls["count"], 0)
        self.assertTrue(all(update["decision"]["result"] not in ("IN", "OUT") for update in updates))

    def test_landing_events_belong_to_one_track_and_new_trajectories_get_new_ids(self):
        path = [{"frame": index, "x": 20 + index * 12, "y": 20 + index * 15} for index in range(5)]
        analyzer, _ = make_analyzer(
            candidate_provider=lambda n: [{"x": 20 + n * 8, "y": 20 + n * 10, "score": 0.95}],
            tracker=lambda _history: (path, 0.9),
            landing_estimator=lambda *_args: {"frame": 2, "x": 50, "y": 60, "method": "downward_phase_bounce"},
            decision_maker=lambda _point, _region: {"result": "IN", "inside": True},
        )
        updates = [analyzer.process_frame(frame(), timestamp=index / 30) for index in range(5)]
        verdicts = [update for update in updates if update["decision"]["result"] in ("IN", "OUT")]
        self.assertGreaterEqual(len(verdicts), 2)
        event_ids = [update["decision"]["event_id"] for update in verdicts]
        self.assertEqual(len(event_ids), len(set(event_ids)))
        self.assertTrue(all(update["decision"]["track_id"] == update["track_id"] for update in verdicts))

    def test_legacy_session_frame_id_protocol_survives_bad_frames_and_reconnect(self):
        client = TestClient(main.app)
        for expected_frame_id in (40, 41):
            with client.websocket_connect("/ws/analyze-live") as websocket:
                greeting = websocket.receive_json()
                self.assertEqual(greeting["state"], "SEARCHING_FOR_COURT")
                self.assertEqual(greeting["message"], "Send binary JPEG frames to begin.")
                websocket.send_text(json.dumps({
                    "type": "session", "session_id": "legacy-client", "next_frame_id": expected_frame_id,
                }))
                websocket.send_bytes(b"not a JPEG")
                update = websocket.receive_json()
                self.assertEqual(update["decision"]["result"], "UNKNOWN")
                self.assertEqual(update["frame_id"], expected_frame_id)
                self.assertEqual(update["frame_index"], 1)
                self.assertIsNone(update["analysis_frame_index"])

    def test_portrait_and_landscape_landing_bounds_remain_supported(self):
        landscape = [
            {"frame": index, "x": 1150 + index * 8, "y": y}
            for index, y in enumerate((300, 400, 500, 580, 650, 580))
        ]
        portrait = [
            {"frame": index, "x": 500 + index * 4, "y": y}
            for index, y in enumerate((300, 550, 800, 1100, 1450, 1350))
        ]
        self.assertIsNotNone(main.choose_landing_point(landscape, 8, frame_size=(720, 1280)))
        self.assertIsNotNone(main.choose_landing_point(portrait, 8, frame_size=(1700, 800)))
        self.assertIsNone(main.choose_landing_point(landscape, 8, frame_size=(480, 640)))


if __name__ == "__main__":
    unittest.main()
