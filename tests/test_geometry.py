import unittest

import cv2
import numpy as np

from court_geometry import infer_court_region_from_lines
from main import detect_court, build_court_region, check_landing_inside_court, choose_landing_point


def render_edges(shape, corners, color=(240, 240, 240), background=(25, 90, 35)):
    height, width = shape
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = background
    lines = []
    for a, b in zip(corners, corners[1:] + corners[:1]):
        ok, p1, p2 = cv2.clipLine((0, 0, width, height), tuple(map(int, a)), tuple(map(int, b)))
        if ok:
            cv2.line(frame, p1, p2, color, 3, cv2.LINE_AA)
            lines.append({"x1": p1[0], "y1": p1[1], "x2": p2[0], "y2": p2[1]})
    # A badminton service/net marking inside the outer boundary helps
    # distinguish court line networks from arbitrary rectangular structures.
    a = ((corners[0][0] + corners[3][0]) / 2, (corners[0][1] + corners[3][1]) / 2)
    b = ((corners[1][0] + corners[2][0]) / 2, (corners[1][1] + corners[2][1]) / 2)
    ok, p1, p2 = cv2.clipLine((0, 0, width, height), tuple(map(int, a)), tuple(map(int, b)))
    if ok:
        cv2.line(frame, p1, p2, color, 3, cv2.LINE_AA)
        lines.append({"x1": p1[0], "y1": p1[1], "x2": p2[0], "y2": p2[1]})
    return frame, lines


class PartialCourtGeometryTests(unittest.TestCase):
    def test_full_court_geometry(self):
        corners = [(90, 60), (550, 70), (535, 420), (105, 410)]
        frame, _ = render_edges((480, 640), corners)
        detected = detect_court(frame)
        region, mode = build_court_region(frame, corners, detected["lines"])
        self.assertEqual(mode, "geometry_polygon")
        self.assertEqual(len(region), 4)

    def test_partial_court_without_visible_corners_has_supported_region(self):
        # All four boundary lines cross the image, while every corner lies
        # outside it: geometry must come from line families, not corner pixels.
        corners = [(-100, 200), (400, -100), (740, 280), (240, 580)]
        frame, _ = render_edges((480, 640), corners)
        detected = detect_court(frame)
        self.assertGreaterEqual(detected["court_lines_detected"], 5)
        region, mode = build_court_region(frame, None, detected["lines"])
        self.assertEqual(mode, "partial_geometry_polygon")
        region, diagnostics = infer_court_region_from_lines(frame, detected["lines"])
        self.assertTrue(diagnostics["reliable"], diagnostics)
        self.assertFalse(diagnostics["corners_visible"])
        self.assertEqual(len(region), 4)

    def test_insufficient_partial_lines_returns_unknown_geometry(self):
        corners = [(-100, 200), (400, -100), (740, 280), (240, 580)]
        frame, lines = render_edges((480, 640), corners)
        region, diagnostics = infer_court_region_from_lines(frame, lines[:3])
        self.assertIsNone(region)
        self.assertFalse(diagnostics["reliable"])
        region, mode = build_court_region(frame, None, lines[:3])
        self.assertIsNone(region)
        self.assertEqual(check_landing_inside_court({"x": 320, "y": 240}, region)["result"], "UNKNOWN")

    def test_rotated_camera_geometry(self):
        # A rotated, perspective-skewed court; no edge is assumed horizontal.
        corners = [(-40, 20), (530, -130), (700, 550), (50, 350)]
        frame, lines = render_edges((480, 640), corners)
        region, diagnostics = infer_court_region_from_lines(frame, lines)
        self.assertTrue(diagnostics["reliable"], diagnostics)

    def test_geometry_is_independent_of_floor_color(self):
        corners = [(100, 80), (530, 70), (550, 390), (80, 410)]
        for floor in ((30, 120, 30), (150, 50, 80), (35, 35, 150)):
            frame, _ = render_edges((480, 640), corners, background=floor)
            detected = detect_court(frame)
            region, mode = build_court_region(frame, corners, detected["lines"])
            self.assertEqual(mode, "geometry_polygon", floor)
            self.assertEqual(len(region), 4)

    def test_valid_landing_trajectory_has_observed_reversal(self):
        trajectory = [
            {"frame": 0, "x": 40, "y": 35},
            {"frame": 1, "x": 58, "y": 82},
            {"frame": 2, "x": 77, "y": 145},
            {"frame": 3, "x": 92, "y": 220},
            {"frame": 4, "x": 98, "y": 125},
        ]
        point = choose_landing_point(trajectory, 60)
        self.assertIsNotNone(point)
        self.assertEqual(point["method"], "downward_phase_bounce")

    def test_ambiguous_landing_stays_unknown(self):
        polygon = [(50, 50), (350, 50), (350, 250), (50, 250)]
        result = check_landing_inside_court({"x": 54, "y": 150}, polygon, margin=10)
        self.assertEqual(result["result"], "UNKNOWN")
        self.assertEqual(check_landing_inside_court(None, None)["result"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
