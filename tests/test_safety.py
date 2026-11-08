import unittest

import numpy as np

from main import (
    build_court_region,
    check_landing_inside_court,
    choose_landing_point,
    track_shuttle,
)


class SafetyRegressionTests(unittest.TestCase):
    def test_partial_or_missing_geometry_fails_closed(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        region, mode = build_court_region(frame, None)
        self.assertIsNone(region)
        self.assertNotEqual(mode, "geometry_polygon")

    def test_near_boundary_without_exact_contact_is_unknown(self):
        polygon = [(10, 10), (210, 10), (210, 110), (10, 110)]
        result = check_landing_inside_court({"x": 12, "y": 50}, polygon, margin=8)
        self.assertEqual(result["result"], "UNKNOWN")

    def test_exact_boundary_contact_counts_as_in(self):
        polygon = [(10, 10), (210, 10), (210, 110), (10, 110)]
        result = check_landing_inside_court({"x": 10, "y": 50}, polygon, margin=14)
        self.assertEqual(result["result"], "IN")
        self.assertEqual(result["boundary_distance_pixels"], 0.0)

    def test_clear_inside_and_outside_are_distinguished(self):
        polygon = [(10, 10), (210, 10), (210, 110), (10, 110)]
        self.assertEqual(check_landing_inside_court({"x": 100, "y": 50}, polygon)["result"], "IN")
        self.assertEqual(check_landing_inside_court({"x": 240, "y": 50}, polygon)["result"], "OUT")

    def test_landing_requires_observed_descent_and_reversal(self):
        descending_only = [
            {"frame": 0, "x": 20, "y": 20},
            {"frame": 1, "x": 35, "y": 70},
            {"frame": 2, "x": 55, "y": 120},
            {"frame": 3, "x": 75, "y": 180},
        ]
        self.assertIsNone(choose_landing_point(descending_only, 30))
        bounce = descending_only + [{"frame": 4, "x": 85, "y": 90}]
        self.assertIsNotNone(choose_landing_point(bounce, 30))

    def test_landing_can_use_observed_settle_but_not_disappearance(self):
        settling = [
            {"frame": 0, "x": 20, "y": 20},
            {"frame": 1, "x": 35, "y": 70},
            {"frame": 2, "x": 55, "y": 120},
            {"frame": 3, "x": 56, "y": 122},
            {"frame": 4, "x": 57, "y": 123},
        ]
        landing = choose_landing_point(settling, 30, max_frame_gap=2)
        self.assertIsNotNone(landing)
        self.assertEqual(landing["method"], "downward_phase_settle")
        self.assertIsNone(choose_landing_point(settling[:3], 30, max_frame_gap=2))

    def test_stationary_candidates_do_not_become_a_shuttle(self):
        candidate = {
            "x": 100.0, "y": 100.0, "score": 0.9,
            "motion_ratio": 0.8, "yellow_ratio": 0.0,
        }
        path, confidence = track_shuttle([[dict(candidate)] for _ in range(24)])
        self.assertLess(confidence, 0.45)


if __name__ == "__main__":
    unittest.main()
