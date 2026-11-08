import unittest

from baseline_geometry import classify_baseline_landing, match_baseline_observation, normalize_baseline_config
from baseline_state import BaselineLock


def config():
    return normalize_baseline_config({
        "type": "baseline_calibration",
        "court_type": "singles",
        "points": [{"x": 0.2, "y": 0.5}, {"x": 0.8, "y": 0.5}, {"x": 0.5, "y": 0.75}],
    })


class BaselineGeometryTests(unittest.TestCase):
    def test_user_marked_baseline_can_lock_without_full_court_polygon(self):
        marked = config()
        support = match_baseline_observation([{"x1": 20, "y1": 50, "x2": 80, "y2": 50}], marked, 100, 100)
        self.assertTrue(support["valid"])
        self.assertGreaterEqual(support["confidence"], 0.35)

    def test_unmarked_middle_or_side_line_cannot_silently_replace_selected_baseline(self):
        marked = config()
        other_line = match_baseline_observation([{"x1": 20, "y1": 25, "x2": 80, "y2": 25}], marked, 100, 100)
        self.assertFalse(other_line["valid"])
        self.assertEqual(other_line["reason"], "marked_baseline_not_visible")

    def test_missing_user_confirmation_does_not_select_any_detected_line(self):
        self.assertIsNone(normalize_baseline_config({"type": "baseline_calibration", "points": [], "court_type": "singles"}))

    def test_baseline_lock_survives_tracking_state_changes_and_brief_detector_misses(self):
        lock = BaselineLock(stable_frames=3, loss_limit=4)
        lock.configure()
        self.assertFalse(lock.observe(True)["locked"])
        self.assertFalse(lock.observe(True)["locked"])
        self.assertTrue(lock.observe(True)["locked"])
        for _ in range(3):
            snapshot = lock.observe(False, "temporary_line_miss")
            self.assertTrue(snapshot["locked"])
            self.assertEqual(snapshot["state"], "BASELINE_LOCKED")
        # Shuttle updates/landing transitions do not touch the independent lock.
        self.assertTrue(lock.snapshot()["locked"])
        self.assertTrue(lock.observe(True)["locked"])

    def test_persistent_boundary_loss_enters_controlled_reacquisition(self):
        lock = BaselineLock(stable_frames=3, loss_limit=4)
        lock.configure()
        for _ in range(3):
            lock.observe(True)
        for _ in range(4):
            snapshot = lock.observe(False, "marked_baseline_not_visible")
        self.assertFalse(snapshot["locked"])
        self.assertEqual(snapshot["state"], "BASELINE_REACQUISITION")
        self.assertEqual(lock.observe(True)["state"], "BASELINE_REACQUISITION")
        self.assertEqual(lock.observe(True)["state"], "BASELINE_REACQUISITION")
        self.assertTrue(lock.observe(True)["locked"])

    def test_camera_shifted_marking_causes_reacquisition_not_silent_line_substitution(self):
        marked = config()
        shifted = [{"x1": 20, "y1": 68, "x2": 80, "y2": 68}]
        self.assertFalse(match_baseline_observation(shifted, marked, 100, 100)["valid"])
        lock = BaselineLock(stable_frames=3, loss_limit=4)
        lock.configure()
        for _ in range(3):
            lock.observe(True)
        for _ in range(4):
            status = lock.observe(False, "marked_baseline_not_visible")
        self.assertEqual(status["state"], "BASELINE_REACQUISITION")

    def test_confirmed_lock_can_be_retained_for_frame_revalidation_after_reconnect(self):
        lock = BaselineLock(stable_frames=3, loss_limit=4)
        lock.configure()
        for _ in range(3):
            lock.observe(True)
        restored = BaselineLock(stable_frames=3, loss_limit=4)
        restored.restore_locked()
        self.assertTrue(restored.snapshot()["locked"])
        self.assertIn("revalidation", restored.snapshot()["status"])
        self.assertTrue(restored.observe(False, "one_frame_blur")["locked"])
        self.assertTrue(restored.observe(True)["locked"])

    def test_baseline_landing_classifies_in_out_and_ambiguous_evidence_safely(self):
        marked = config()
        on_line = classify_baseline_landing({"x": 50, "y": 50, "method": "downward_phase_bounce"}, marked, 100, 100, 0.9)
        inside = classify_baseline_landing({"x": 50, "y": 65, "method": "downward_phase_bounce"}, marked, 100, 100, 0.9)
        outside = classify_baseline_landing({"x": 50, "y": 40, "method": "downward_phase_bounce"}, marked, 100, 100, 0.9)
        ambiguous = classify_baseline_landing(None, marked, 100, 100, 0.9)
        beyond_segment = classify_baseline_landing({"x": 2, "y": 40, "method": "downward_phase_bounce"}, marked, 100, 100, 0.9)
        low_confidence = classify_baseline_landing({"x": 50, "y": 40, "method": "downward_phase_bounce"}, marked, 100, 100, 0.2)
        malformed = classify_baseline_landing({"x": float("nan"), "y": 40, "method": "downward_phase_bounce"}, marked, 100, 100, 0.9)
        self.assertEqual(on_line["result"], "IN")
        self.assertEqual(inside["result"], "IN")
        self.assertEqual(outside["result"], "OUT")
        self.assertEqual(ambiguous["result"], "UNKNOWN")
        self.assertEqual(beyond_segment["result"], "UNKNOWN")
        self.assertEqual(low_confidence["result"], "UNKNOWN")
        self.assertEqual(malformed["result"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
