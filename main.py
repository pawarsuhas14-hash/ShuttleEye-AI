from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, Response
from court_corners import get_court_corners
import cv2
import numpy as np
import os
import uuid
import shutil
import math
import time

app = FastAPI(
    title="ShuttleEye AI",
    description="AI-powered badminton shuttle detection and line-call analysis",
    version="2.1.3",
)


@app.get("/")
def home():
    return {
        "message": "Welcome to ShuttleEye AI",
        "status": "running",
        "version": "2.1.3",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


def detect_court(frame):
    """Detect visible court lines for diagnostics."""
    if frame is None:
        return {"court_detected": False, "court_lines_detected": 0, "lines": []}

    height, width = frame.shape[:2]
    scale = min(1.0, 900.0 / max(width, height))
    small = cv2.resize(
        frame,
        (max(1, int(width * scale)), max(1, int(height * scale))),
    )

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.bitwise_or(
        cv2.Canny(gray, 40, 120),
        cv2.Canny(gray, 70, 180),
    )

    min_len = max(35, int(min(small.shape[:2]) * 0.10))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=45,
        minLineLength=min_len,
        maxLineGap=25,
    )

    detected_lines = []
    if lines is not None:
        for raw in lines:
            vals = np.asarray(raw).reshape(-1)
            if len(vals) < 4:
                continue
            x1, y1, x2, y2 = vals[:4]
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length < min_len:
                continue
            detected_lines.append({
                "x1": int(round(x1 / scale)),
                "y1": int(round(y1 / scale)),
                "x2": int(round(x2 / scale)),
                "y2": int(round(y2 / scale)),
            })

    detected_lines.sort(
        key=lambda item: np.hypot(
            item["x2"] - item["x1"],
            item["y2"] - item["y1"],
        ),
        reverse=True,
    )

    return {
        "court_detected": len(detected_lines) >= 4,
        "court_lines_detected": len(detected_lines),
        "lines": detected_lines[:40],
    }


def build_court_region(frame, court_corners):
    """Build a playable court region without assuming camera orientation or court colour.

    Prefer a validated quadrilateral. If the corner detector produces a
    frame-spanning polygon, use the visible court-line network to find a
    substantial interior boundary and infer the playable side from the amount
    of perpendicular court-line support on each side.
    """
    if frame is None or not court_corners:
        return court_corners, "court_not_detected"

    try:
        if isinstance(court_corners, dict):
            keys = ("top_left", "top_right", "bottom_right", "bottom_left")
            if not all(k in court_corners for k in keys):
                return court_corners, "invalid_corners"
            pts = np.array([court_corners[k] for k in keys], dtype=np.float32)
        else:
            pts = np.array(list(court_corners), dtype=np.float32)

        if pts.shape != (4, 2) or not np.isfinite(pts).all():
            return court_corners, "invalid_corners"

        h, w = frame.shape[:2]
        hull = cv2.convexHull(pts).reshape(-1, 2)
        if len(hull) != 4 or not cv2.isContourConvex(hull.reshape(-1, 1, 2)):
            return pts.astype(np.int32).tolist(), "low_confidence_polygon"

        polygon = hull.astype(np.float32)
        area_ratio = abs(float(cv2.contourArea(polygon))) / float(max(1, w * h))
        side_lengths = [
            float(np.linalg.norm(polygon[(i + 1) % 4] - polygon[i]))
            for i in range(4)
        ]

        if min(side_lengths) < max(35.0, min(w, h) * 0.025):
            return polygon.astype(np.int32).tolist(), "low_confidence_polygon"
        if max(side_lengths) / max(min(side_lengths), 1.0) > 18.0:
            return polygon.astype(np.int32).tolist(), "low_confidence_polygon"

        def direction_similarity(a, b):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-6 or nb < 1e-6:
                return 0.0
            return abs(float(np.dot(a, b) / (na * nb)))

        ev = [polygon[(i + 1) % 4] - polygon[i] for i in range(4)]
        parallel_a = direction_similarity(ev[0], ev[2])
        parallel_b = direction_similarity(ev[1], ev[3])
        geometry_score = (
            0.35 * min(1.0, area_ratio / 0.25)
            + 0.325 * parallel_a
            + 0.325 * parallel_b
        )

        # A geometrically neat quad is not enough: Hough can lock onto walls,
        # phone framing, or other long structures.  Require visible court-line
        # evidence before trusting the four-corner result.
        hsv_geom = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        value_geom = hsv_geom[:, :, 2]
        sat_geom = hsv_geom[:, :, 1]
        geom_mask = ((value_geom >= 150) & (sat_geom <= 130)).astype(np.uint8)
        geom_mask = cv2.morphologyEx(
            geom_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
        )

        edge_support = []
        for i in range(4):
            a = polygon[i]
            b = polygon[(i + 1) % 4]
            edge_len = float(np.linalg.norm(b - a))
            if edge_len < 1.0:
                edge_support.append(0.0)
                continue
            samples = max(24, int(edge_len / 4.0))
            ts = np.linspace(0.0, 1.0, samples)
            hits = 0
            for t in ts:
                px, py = np.round(a + (b - a) * t).astype(int)
                if 0 <= px < w and 0 <= py < h:
                    r = 3
                    patch = geom_mask[max(0, py-r):min(h, py+r+1),
                                       max(0, px-r):min(w, px+r+1)]
                    if patch.size and float(np.mean(patch)) >= 0.25:
                        hits += 1
            edge_support.append(hits / float(samples))

        near_frame = 0
        frame_tol_x = max(8.0, 0.025 * w)
        frame_tol_y = max(8.0, 0.025 * h)
        for x, y in polygon:
            if x <= frame_tol_x or x >= (w - 1 - frame_tol_x) or y <= frame_tol_y or y >= (h - 1 - frame_tol_y):
                near_frame += 1

        support_ok = (
            max(edge_support) >= 0.42
            and sorted(edge_support, reverse=True)[1] >= 0.08
            and near_frame <= 2
        )

        if area_ratio <= 0.50 and geometry_score >= 0.58 and support_ok:
            return polygon.astype(np.int32).tolist(), "geometry_polygon"

        # ------------------------------------------------------------------
        # Fallback: identify an interior court boundary from bright court
        # lines. Badminton boundary lines are normally bright/low-saturation;
        # this does NOT assume a green/blue/red court surface and does not
        # assume the boundary is vertical/horizontal.
        # ------------------------------------------------------------------
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2]
        saturation = hsv[:, :, 1]
        line_mask = ((value >= 150) & (saturation <= 130)).astype(np.uint8)
        line_mask = cv2.morphologyEx(
            line_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
        )

        raw = cv2.HoughLinesP(
            line_mask * 255,
            1,
            np.pi / 180.0,
            25,
            minLineLength=max(55, int(min(w, h) * 0.15)),
            maxLineGap=28,
        )
        if raw is None:
            return polygon.astype(np.int32).tolist(), "low_confidence_polygon"

        lines = []
        for r in raw:
            vals = np.asarray(r).reshape(-1)
            if len(vals) < 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in vals[:4]]
            length = math.hypot(x2 - x1, y2 - y1)
            if length < max(55.0, min(w, h) * 0.15):
                continue
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
            lines.append((x1, y1, x2, y2, length, angle))

        if not lines:
            return polygon.astype(np.int32).tolist(), "low_confidence_polygon"

        best = None
        min_dim = float(min(w, h))
        frame_diag = math.hypot(w, h)

        for idx, line in enumerate(lines):
            x1, y1, x2, y2, length, angle = line
            # The boundary itself should be a substantial image structure;
            # shorter court lines are used only as supporting evidence.
            if length < 0.40 * max(w, h):
                continue

            p1 = np.array([x1, y1], dtype=np.float32)
            p2 = np.array([x2, y2], dtype=np.float32)
            tangent = (p2 - p1) / max(length, 1e-6)
            normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
            midpoint = (p1 + p2) * 0.5

            # Ignore structures whose midpoint is very close to a frame edge.
            edge_clearance = min(
                float(midpoint[0]), float(midpoint[1]),
                float(w - midpoint[0]), float(h - midpoint[1])
            )
            if edge_clearance < 0.035 * min_dim:
                continue

            side_length = [0.0, 0.0]
            side_count = [0, 0]
            near_crossings = [0, 0]

            for j, other in enumerate(lines):
                if j == idx:
                    continue
                ox1, oy1, ox2, oy2, olen, oangle = other
                ov = np.array([ox2 - ox1, oy2 - oy1], dtype=np.float32)
                on = np.linalg.norm(ov)
                if on < 1e-6:
                    continue

                # Perpendicular (within perspective tolerance) lines are the
                # strongest evidence of a court-line network.
                if abs(float(np.dot(ov / on, tangent))) > 0.42:
                    continue

                omp = np.array([(ox1 + ox2) * 0.5, (oy1 + oy2) * 0.5], dtype=np.float32)
                signed = float(np.dot(omp - midpoint, normal))
                side = 0 if signed >= 0 else 1

                # Only count clearly separated line support. The boundary line
                # itself should not make both sides look equally rich.
                separation = abs(signed)
                if separation < max(8.0, 0.018 * min_dim):
                    continue

                weight = min(olen, 0.85 * min_dim)
                side_length[side] += weight
                side_count[side] += 1

                # A line whose segment comes close to the candidate boundary
                # is especially useful evidence of a real court intersection.
                rel1 = np.array([ox1, oy1], dtype=np.float32) - p1
                rel2 = np.array([ox2, oy2], dtype=np.float32) - p1
                along1 = float(np.dot(rel1, tangent))
                along2 = float(np.dot(rel2, tangent))
                if max(along1, along2) >= -35.0 and min(along1, along2) <= length + 35.0:
                    near_crossings[side] += 1

            total_support = side_length[0] + side_length[1]
            if total_support < 0.55 * min_dim:
                continue

            richer = 0 if side_length[0] >= side_length[1] else 1
            poorer = 1 - richer
            imbalance = (
                side_length[richer] - side_length[poorer]
            ) / max(total_support, 1.0)
            count_support = min(1.0, side_count[richer] / 4.0)
            crossing_support = min(1.0, near_crossings[richer] / 3.0)
            length_support = min(1.0, length / max(0.60 * min_dim, 1.0))
            centrality = min(1.0, edge_clearance / max(0.25 * min_dim, 1.0))

            score = (
                0.40 * imbalance
                + 0.25 * count_support
                + 0.20 * crossing_support
                + 0.10 * length_support
                + 0.05 * centrality
            )

            if best is None or score > best[0]:
                best = (score, line, normal, richer, side_length, side_count)

        if best is None or best[0] < 0.38:
            return polygon.astype(np.int32).tolist(), "low_confidence_polygon"

        score, line, normal, richer, side_length, side_count = best
        x1, y1, x2, y2, length, angle = line
        p1 = np.array([x1, y1], dtype=np.float32)
        p2 = np.array([x2, y2], dtype=np.float32)
        tangent = (p2 - p1) / max(length, 1e-6)
        chosen_normal = normal if richer == 0 else -normal

        # Extend the boundary line to the image edges, then take the inferred
        # playable half-plane. This handles arbitrary camera rotation.
        intersections = []
        for x_edge in (0.0, float(w - 1)):
            if abs(tangent[0]) > 1e-6:
                t = (x_edge - p1[0]) / tangent[0]
                y = p1[1] + t * tangent[1]
                if -frame_diag <= y <= h - 1 + frame_diag:
                    intersections.append(np.array([x_edge, y], dtype=np.float32))
        for y_edge in (0.0, float(h - 1)):
            if abs(tangent[1]) > 1e-6:
                t = (y_edge - p1[1]) / tangent[1]
                x = p1[0] + t * tangent[0]
                if -frame_diag <= x <= w - 1 + frame_diag:
                    intersections.append(np.array([x, y_edge], dtype=np.float32))

        if len(intersections) < 2:
            return polygon.astype(np.int32).tolist(), "low_confidence_polygon"

        # Keep the two intersections that are farthest apart.
        pair = max(
            ((a, b) for i, a in enumerate(intersections) for b in intersections[i + 1:]),
            key=lambda ab: float(np.linalg.norm(ab[1] - ab[0]))
        )
        a, b = pair
        extension = frame_diag * 1.5
        a = a - tangent * extension
        b = b + tangent * extension
        c = b + chosen_normal * extension
        d = a + chosen_normal * extension
        region = np.array([a, b, c, d], dtype=np.float32)

        # Clip to frame with a convex hull; this creates a stable half-plane
        # polygon for point-in-polygon testing.
        region[:, 0] = np.clip(region[:, 0], 0, w - 1)
        region[:, 1] = np.clip(region[:, 1], 0, h - 1)
        region = cv2.convexHull(region.astype(np.float32)).reshape(-1, 2)
        if len(region) < 4:
            return polygon.astype(np.int32).tolist(), "low_confidence_polygon"

        return region.astype(np.int32).tolist(), "geometry_polygon"

    except Exception:
        return court_corners, "low_confidence_polygon"

def check_landing_inside_court(landing_point, court_corners, margin=0):
    if landing_point is None:
        return {
            "result": "UNKNOWN",
            "inside": None,
            "reason": "Landing point not detected",
        }

    if court_corners is None:
        return {
            "result": "UNKNOWN",
            "inside": None,
            "reason": "Court corners not detected",
        }

    try:
        if isinstance(court_corners, dict):
            polygon_points = [
                court_corners["top_left"],
                court_corners["top_right"],
                court_corners["bottom_right"],
                court_corners["bottom_left"],
            ]
        elif isinstance(court_corners, (list, tuple)) and len(court_corners) == 4:
            polygon_points = list(court_corners)
        else:
            raise ValueError("Unsupported court corner format")

        polygon = np.array(polygon_points, dtype=np.float32)
        point = (float(landing_point["x"]), float(landing_point["y"]))

        signed_distance = cv2.pointPolygonTest(polygon, point, True)

        if signed_distance >= -float(margin):
            return {
                "result": "IN",
                "inside": True,
                "boundary_distance_pixels": round(float(signed_distance), 2),
                "message": "Shuttle landed inside court",
            }

        return {
            "result": "OUT",
            "inside": False,
            "boundary_distance_pixels": round(float(signed_distance), 2),
            "message": "Shuttle landed outside court",
        }
    except Exception as e:
        return {
            "result": "UNKNOWN",
            "inside": None,
            "reason": f"Landing decision error: {str(e)}",
        }


def _candidate_score(gray, hsv, contour, x, y, w, h, area):
    roi_gray = gray[y:y + h, x:x + w]
    roi_hsv = hsv[y:y + h, x:x + w]
    if roi_gray.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    sat = roi_hsv[:, :, 1]
    val = roi_hsv[:, :, 2]
    hue = roi_hsv[:, :, 0]

    white = (val > 135) & (sat < 155)
    yellow = (hue >= 8) & (hue <= 45) & (sat > 45) & (val > 80)
    appearance_ratio = float(np.mean(white | yellow))
    yellow_ratio = float(np.mean(yellow))
    mean_value = float(np.mean(val)) / 255.0

    perimeter = cv2.arcLength(contour, True)
    circularity = 0.0
    if perimeter > 1.0:
        circularity = min(
            1.0,
            float(4.0 * math.pi * area / (perimeter * perimeter)),
        )

    if 2 <= area <= 120:
        area_score = 1.0
    elif area <= 350:
        area_score = max(0.0, 1.0 - (area - 120.0) / 230.0)
    else:
        area_score = 0.0

    aspect = max(w, h) / max(1, min(w, h))
    shape_score = 1.0 if aspect <= 5.0 else max(0.0, 1.0 - (aspect - 5.0) / 4.0)

    score = (
        0.36 * appearance_ratio
        + 0.18 * yellow_ratio
        + 0.16 * mean_value
        + 0.14 * circularity
        + 0.10 * area_score
        + 0.06 * shape_score
    )
    return float(score), appearance_ratio, yellow_ratio, circularity



def build_static_line_mask(frame, court_analysis):
    """Build a mask of strong static court lines from the first frame."""
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if not court_analysis or not court_analysis.get("lines"):
        return mask

    for line in court_analysis["lines"]:
        try:
            x1 = int(line["x1"])
            y1 = int(line["y1"])
            x2 = int(line["x2"])
            y2 = int(line["y2"])
            # Thick enough to suppress small contour fragments generated by
            # camera/compression noise around a court line.
            cv2.line(mask, (x1, y1), (x2, y2), 255, 22)
        except Exception:
            continue

    return mask


def extract_frame_candidates(frame, previous_gray, previous_previous_gray=None, static_line_mask=None):
    """
    Generate shuttle candidates from motion + white/yellow appearance.

    The analysis is performed at reduced resolution for Render performance,
    then candidate coordinates are scaled back to the original frame.
    """
    original_h, original_w = frame.shape[:2]
    scale = min(1.0, 960.0 / max(original_w, original_h))

    if scale < 0.999:
        work = cv2.resize(
            frame,
            (max(1, int(original_w * scale)), max(1, int(original_h * scale))),
        )
    else:
        work = frame

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)

    if previous_gray is None:
        return gray, []

    diff1 = cv2.absdiff(previous_gray, gray)
    if previous_previous_gray is not None:
        diff2 = cv2.absdiff(previous_previous_gray, gray)
        difference = cv2.max(diff1, diff2)
    else:
        difference = diff1

    _, motion = cv2.threshold(difference, 18, 255, cv2.THRESH_BINARY)

    kernel3 = np.ones((3, 3), np.uint8)
    motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, kernel3, iterations=1)
    motion = cv2.dilate(motion, kernel3, iterations=1)

    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 120], dtype=np.uint8),
        np.array([180, 175, 255], dtype=np.uint8),
    )
    yellow_mask = cv2.inRange(
        hsv,
        np.array([8, 35, 70], dtype=np.uint8),
        np.array([45, 255, 255], dtype=np.uint8),
    )
    appearance = cv2.bitwise_or(white_mask, yellow_mask)

    # Primary detector: motion + shuttle appearance.
    combined = cv2.bitwise_and(motion, appearance)

    contours, _ = cv2.findContours(
        combined,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    def collect(contour_list, motion_only=False, appearance_only=False):
        for contour in contour_list:
            area = float(cv2.contourArea(contour))
            if not (1.5 <= area <= 450.0):
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w < 2 or h < 2 or w > 70 or h > 70:
                continue

            roi_motion = motion[y:y + h, x:x + w]
            motion_ratio = float(np.mean(roi_motion > 0)) if roi_motion.size else 0.0

            line_ratio = 0.0
            if static_line_mask is not None:
                mask_h, mask_w = static_line_mask.shape[:2]
                sx = int(round(x / scale))
                sy = int(round(y / scale))
                sw = max(1, int(round(w / scale)))
                sh = max(1, int(round(h / scale)))
                sx2 = min(mask_w, sx + sw)
                sy2 = min(mask_h, sy + sh)
                if 0 <= sx < sx2 and 0 <= sy < sy2:
                    roi_line = static_line_mask[sy:sy2, sx:sx2]
                    line_ratio = float(np.mean(roi_line > 0)) if roi_line.size else 0.0

            score, appearance_ratio, yellow_ratio, circularity = _candidate_score(
                gray, hsv, contour, x, y, w, h, area
            )

            if motion_only and score < 0.24:
                continue
            if appearance_only:
                # Do not accept a stationary yellow object as the shuttle.
                # The yellow fallback is allowed only when there is measurable
                # temporal motion inside the blob.
                if yellow_ratio < 0.10:
                    continue
                if motion_ratio < 0.015:
                    continue
                if area < 2.0 or area > 700.0:
                    continue
            elif not motion_only and appearance_ratio < 0.05:
                continue

            # Static court lines are a major false-positive source. Suppress
            # candidates sitting on a detected court line unless they have
            # strong yellow/compact-shuttle evidence.
            if line_ratio > 0.35:
                if not (yellow_ratio > 0.25 and circularity > 0.08 and area < 180.0):
                    continue

            candidates.append({
                "x": (x + w / 2.0) / scale,
                "y": (y + h / 2.0) / scale,
                "area": round(area / max(scale * scale, 1e-6), 2),
                "w": int(round(w / scale)),
                "h": int(round(h / scale)),
                "score": round(score, 4),
                "motion_ratio": round(motion_ratio, 4),
                "appearance_ratio": round(appearance_ratio, 4),
                "yellow_ratio": round(yellow_ratio, 4),
                "circularity": round(circularity, 4),
                "line_ratio": round(line_ratio, 4),
            })

    collect(contours, motion_only=False)

    # Fallback 1: motion-only blobs are useful when the shuttle is blurred.
    if not candidates:
        motion_contours, _ = cv2.findContours(
            motion,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        collect(motion_contours, motion_only=True)

    # Fallback 2: detect compact yellow shuttle blobs even when the frame
    # difference is too weak. This is important for high-speed shuttle motion.
    if not candidates:
        yellow_contours, _ = cv2.findContours(
            yellow_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        collect(yellow_contours, appearance_only=True)

    candidates.sort(
        key=lambda c: (
            c["score"],
            c["yellow_ratio"],
            c["appearance_ratio"],
        ),
        reverse=True,
    )

    return gray, candidates[:16]


def _transition_score(previous, current, previous_previous=None, frame_gap=1):
    dx = current["x"] - previous["x"]
    dy = current["y"] - previous["y"]
    distance = math.hypot(dx, dy)

    # Perspective and fast shuttle motion can create large frame-to-frame jumps.
    max_jump = 700.0 * max(1, frame_gap)
    if distance > max_jump:
        return -1e9

    score = -1.05 * (distance / max_jump)
    score += 1.8 * current["score"]

    # A line/fixture or other yellow object can remain in almost the same
    # place for hundreds of frames. Penalize zero-motion transitions so the
    # tracker cannot turn a static object into a fake shuttle trajectory.
    if distance < 3.0:
        score -= 0.55
    elif distance < 8.0:
        score -= 0.10

    # Prefer candidates that actually overlap temporal motion.
    score += 0.55 * current.get("motion_ratio", 0.0)

    if previous_previous is not None:
        pdx = previous["x"] - previous_previous["x"]
        pdy = previous["y"] - previous_previous["y"]
        previous_distance = math.hypot(pdx, pdy)

        # Do not bridge a missing frame from a stationary false positive to
        # a distant court line/object. A real shuttle should continue moving
        # or remain very close when a single frame is missed.
        if frame_gap > 1 and previous_distance < 15.0 and distance > 80.0:
            return -1e9

        # Reject implausible teleportation from a nearly stationary object to
        # a distant court line. This is a common failure mode in camera videos.
        if previous_distance < 50.0 and distance > 350.0:
            return -1e9

        if previous_distance > 2.0 and distance > 1.0:
            dot = (pdx * dx + pdy * dy) / (
                previous_distance * distance
            )
            dot = max(-1.0, min(1.0, dot))
            direction_similarity = (dot + 1.0) / 2.0

            speed_ratio = distance / previous_distance
            speed_consistency = math.exp(
                -abs(math.log(max(speed_ratio, 0.05)))
            )

            score += 0.70 * direction_similarity
            score += 0.35 * speed_consistency

    return float(score)


def _track_one_direction(frame_candidates, start_frame, start_candidate, step):
    total_frames = len(frame_candidates)
    path = [{
        "frame": int(start_frame),
        "x": int(round(start_candidate["x"])),
        "y": int(round(start_candidate["y"])),
    }]

    previous = start_candidate
    previous_previous = None
    previous_frame = start_frame
    frame_no = start_frame + step

    misses = 0
    stalled = 0
    while 0 <= frame_no < total_frames:
        candidates = frame_candidates[frame_no]

        if not candidates:
            misses += 1
            if misses > 8:
                break
            frame_no += step
            continue

        gap = abs(frame_no - previous_frame)
        best = None
        best_score = -1e9

        for candidate in candidates:
            score = _transition_score(
                previous,
                candidate,
                previous_previous,
                frame_gap=gap,
            )
            if score > best_score:
                best_score = score
                best = candidate

        if best is None or best_score < -0.70:
            misses += 1
            if misses > 5:
                break
            frame_no += step
            continue

        misses = 0

        step_distance = math.hypot(
            best["x"] - previous["x"],
            best["y"] - previous["y"],
        )
        if step_distance < 3.0:
            stalled += 1
        else:
            stalled = 0

        # A genuine shuttle trajectory cannot remain pixel-stationary for
        # dozens of frames. Stop this path before it contaminates confidence.
        if stalled >= 8:
            break

        previous_previous = previous
        previous = best
        previous_frame = frame_no

        path.append({
            "frame": int(frame_no),
            "x": int(round(best["x"])),
            "y": int(round(best["y"])),
        })
        frame_no += step

    return path


def track_shuttle(frame_candidates_by_frame):
    """
    Bidirectional trajectory tracking with multi-seed temporal validation.

    Instead of choosing only the strongest candidate frame, test strong
    candidates from across the clip. This prevents a late false detection
    from becoming the entire trajectory when an earlier, longer trajectory
    is available.
    """
    usable = [
        (i, candidates)
        for i, candidates in enumerate(frame_candidates_by_frame)
        if candidates
    ]
    if not usable:
        return [], 0.0

    def candidate_strength(candidate):
        return float(
            candidate.get("score", 0.0)
            * (0.55 + 1.45 * candidate.get("motion_ratio", 0.0))
        )

    # Build seed hypotheses across the whole clip, not just one frame.
    # A real shuttle should be supported by a continuous sequence of
    # detections, while an accidental bright object is often isolated.
    seed_pool = []
    for frame_no, candidates in usable:
        ranked = sorted(candidates, key=candidate_strength, reverse=True)[:3]
        for candidate in ranked:
            nearby = 0
            for j in range(max(0, frame_no - 2), min(len(frame_candidates_by_frame), frame_no + 3)):
                if j != frame_no and frame_candidates_by_frame[j]:
                    nearby += 1
            support = 1.0 + 0.20 * (nearby / 4.0)
            seed_pool.append((candidate_strength(candidate) * support, frame_no, candidate))

    seed_pool.sort(key=lambda item: item[0], reverse=True)

    # Keep seeds distributed through time so one late false detection cannot
    # dominate all hypotheses. At most 24 seeds are evaluated.
    selected_seeds = []
    used_frames = []
    for strength, frame_no, candidate in seed_pool:
        if any(abs(frame_no - f) < 4 for f in used_frames):
            continue
        selected_seeds.append((frame_no, candidate))
        used_frames.append(frame_no)
        if len(selected_seeds) >= 24:
            break

    # If temporal spacing filtered too aggressively, fill from remaining
    # strongest candidates.
    if len(selected_seeds) < 8:
        for _, frame_no, candidate in seed_pool:
            if (frame_no, candidate) in selected_seeds:
                continue
            selected_seeds.append((frame_no, candidate))
            if len(selected_seeds) >= 8:
                break

    best_path = []
    best_quality = -1e9

    for seed_frame, seed in selected_seeds:
        backward = _track_one_direction(
            frame_candidates_by_frame,
            seed_frame,
            seed,
            -1,
        )
        forward = _track_one_direction(
            frame_candidates_by_frame,
            seed_frame,
            seed,
            +1,
        )

        path = list(reversed(backward[1:])) + forward
        path.sort(key=lambda p: p["frame"])

        if len(path) < 2:
            continue

        jumps = []
        for a, b in zip(path[:-1], path[1:]):
            gap = max(1, b["frame"] - a["frame"])
            jumps.append(
                math.hypot(b["x"] - a["x"], b["y"] - a["y"])
                / (700.0 * gap)
            )

        smoothness = max(
            0.0,
            1.0 - float(np.mean(np.clip(jumps, 0.0, 1.0))),
        )
        length_score = min(1.0, len(path) / 10.0)
        x_span = max(p["x"] for p in path) - min(p["x"] for p in path)
        y_span = max(p["y"] for p in path) - min(p["y"] for p in path)
        movement_score = min(1.0, math.hypot(x_span, y_span) / 180.0)

        # Strongly favor trajectories supported by multiple frames. A short
        # isolated 3-4 point false track should lose to a longer continuous
        # shuttle track even if its seed candidate is visually strong.
        quality = (
            0.50 * length_score
            + 0.25 * smoothness
            + 0.20 * movement_score
            + 0.05 * seed["score"]
        )
        if len(path) < 5:
            quality -= 0.20
        if len(path) < 4:
            quality -= 0.20

        if quality > best_quality:
            best_quality = quality
            best_path = path

    if len(best_path) < 3:
        return best_path, 0.0

    jumps = []
    for a, b in zip(best_path[:-1], best_path[1:]):
        gap = max(1, b["frame"] - a["frame"])
        jumps.append(
            math.hypot(b["x"] - a["x"], b["y"] - a["y"])
            / (700.0 * gap)
        )

    smoothness = max(
        0.0,
        1.0 - float(np.mean(np.clip(jumps, 0.0, 1.0))),
    )
    length_score = min(1.0, len(best_path) / 10.0)
    x_span = max(p["x"] for p in best_path) - min(p["x"] for p in best_path)
    y_span = max(p["y"] for p in best_path) - min(p["y"] for p in best_path)
    movement_score = min(1.0, math.hypot(x_span, y_span) / 180.0)

    if movement_score < 0.20:
        return best_path, 0.0

    confidence = (
        0.40 * smoothness
        + 0.35 * length_score
        + 0.25 * movement_score
    )

    return best_path, float(confidence)


def create_debug_frame(frame, trajectory, court_corners=None, landing_point=None,
                       decision=None):
    """
    Create a compact annotated frame for visual debugging.
    This is intentionally separate from the API response so we can later
    expose an annotated JPEG/video without changing the detection pipeline.
    """
    debug = frame.copy()

    # Court polygon.
    if court_corners:
        try:
            if isinstance(court_corners, dict):
                pts = [
                    court_corners["top_left"],
                    court_corners["top_right"],
                    court_corners["bottom_right"],
                    court_corners["bottom_left"],
                ]
            else:
                pts = list(court_corners)

            poly = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(debug, [poly], True, (255, 180, 0), 4)
        except Exception:
            pass

    # Trajectory.
    if trajectory:
        pts = [(int(p["x"]), int(p["y"])) for p in trajectory]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(debug, a, b, (255, 0, 255), 4)

        for p in pts:
            cv2.circle(debug, p, 8, (0, 255, 255), -1)

        # Latest tracked position.
        cv2.circle(debug, pts[-1], 16, (0, 255, 0), 3)

    # Estimated landing.
    if landing_point:
        lp = (int(landing_point["x"]), int(landing_point["y"]))
        cv2.circle(debug, lp, 25, (0, 0, 255), 5)
        cv2.drawMarker(
            debug, lp, (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=50,
            thickness=5,
        )

    # Decision banner.
    result = None
    if isinstance(decision, dict):
        result = decision.get("result")

    label = f"ShuttleEye 2.1.2 | {result or 'TRACKING'}"
    cv2.rectangle(debug, (20, 20), (650, 90), (20, 20, 20), -1)
    cv2.putText(
        debug, label, (40, 68),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3,
        cv2.LINE_AA,
    )

    return debug


def choose_landing_point(trajectory, fps, court_corners=None):
    """
    Conservative landing estimator with bounce/reversal detection.

    A badminton shuttle landing can look like:
        descending -> lowest image position -> rapid upward movement

    The previous version rejected that upward reversal. That was too strict:
    a strong reversal immediately after a deep downward trajectory is useful
    evidence of a floor/court contact. We now accept it when the reversal is
    temporally close and the low point is inside/near the detected court.

    If there is no credible downward phase, return UNKNOWN rather than guess.
    """
    if len(trajectory) < 4:
        return None

    x_span = max(p["x"] for p in trajectory) - min(p["x"] for p in trajectory)
    y_span = max(p["y"] for p in trajectory) - min(p["y"] for p in trajectory)
    if math.hypot(x_span, y_span) < 50.0:
        return None

    h_limit = 1920
    w_limit = 1080
    last = trajectory[-1]

    # Leaving the frame is not a landing.
    if (
        last["y"] < 40 or last["y"] > h_limit - 20
        or last["x"] < 5 or last["x"] > w_limit - 5
    ):
        return None

    best_idx = None
    best_score = -1e9

    for i in range(2, len(trajectory) - 1):
        p0 = trajectory[i - 2]
        p1 = trajectory[i - 1]
        p2 = trajectory[i]
        p3 = trajectory[i + 1]

        d1 = p1["y"] - p0["y"]
        d2 = p2["y"] - p1["y"]
        reversal = p2["y"] - p3["y"]

        # Require a sustained downward phase before the low point.
        if d1 < 2 or d2 < 2:
            continue

        # A landing/bounce candidate is a local maximum in image-Y followed
        # by a meaningful upward reversal. A small non-reversal is also
        # accepted for clips where the shuttle simply settles near the floor.
        bounce_evidence = reversal >= 60
        settle_evidence = abs(reversal) <= 15

        if not (bounce_evidence or settle_evidence):
            continue

        # Prefer deeper, more clearly descending points.
        score = float(p2["y"]) + min(float(reversal), 300.0) * 0.35
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx is None:
        return None

    p = trajectory[best_idx]

    # Average only a tiny neighborhood around the contact point. This keeps
    # the estimated point close to the actual low point while reducing noise.
    local = trajectory[max(0, best_idx - 1):min(len(trajectory), best_idx + 2)]
    if len(local) < 2:
        return None

    weights = np.arange(1, len(local) + 1, dtype=np.float32)
    xs = np.array([q["x"] for q in local], dtype=np.float32)
    ys = np.array([q["y"] for q in local], dtype=np.float32)

    method = (
        "downward_phase_bounce"
        if best_idx + 1 < len(trajectory)
        and p["y"] - trajectory[best_idx + 1]["y"] >= 60
        else "downward_phase_settle"
    )

    landing = {
        "frame": int(p["frame"]),
        "x": int(round(float(np.average(xs, weights=weights)))),
        "y": int(round(float(np.average(ys, weights=weights)))),
        "method": method,
    }

    # Require the contact point to be on or reasonably close to the detected
    # court. This rejects unrelated moving objects elsewhere in the frame.
    if court_corners:
        try:
            if isinstance(court_corners, dict):
                polygon_points = [
                    court_corners["top_left"],
                    court_corners["top_right"],
                    court_corners["bottom_right"],
                    court_corners["bottom_left"],
                ]
            else:
                polygon_points = list(court_corners)

            polygon = np.array(polygon_points, dtype=np.float32)
            distance = cv2.pointPolygonTest(
                polygon,
                (float(landing["x"]), float(landing["y"])),
                True,
            )

            # Keep a modest tolerance for detector/camera error, but don't
            # accept points far outside the court.
            if distance < -250.0:
                return None
        except Exception:
            pass

    return landing


@app.post("/debug-video")
async def debug_video(video: UploadFile = File(...)):
    """
    Analyze the uploaded video using the same detector/tracker and return
    one annotated JPEG frame for visual verification.

    Overlay:
      - court polygon
      - tracked trajectory
      - latest tracked position
      - estimated landing point (when available)
      - IN/OUT/UNKNOWN decision
    """
    input_path = None
    try:
        file_id = str(uuid.uuid4())
        safe_filename = os.path.basename(video.filename or "video.mp4")
        input_path = f"/tmp/{file_id}_{safe_filename}"

        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Could not open video"},
            )

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        success, first_frame = cap.read()
        if not success:
            cap.release()
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Could not read video"},
            )

        court_analysis = detect_court(first_frame)
        court_corners = get_court_corners(first_frame)
        court_region, court_region_mode = build_court_region(first_frame, court_corners)
        static_line_mask = build_static_line_mask(first_frame, court_analysis)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        previous_gray = None
        previous_previous_gray = None
        frame_candidates_by_frame = []

        while True:
            success, frame = cap.read()
            if not success:
                break

            current_gray, candidates = extract_frame_candidates(
                frame,
                previous_gray,
                previous_previous_gray,
                static_line_mask=static_line_mask,
            )
            frame_candidates_by_frame.append(candidates)
            previous_previous_gray = previous_gray
            previous_gray = current_gray

        cap.release()

        trajectory, tracking_confidence = track_shuttle(
            frame_candidates_by_frame
        )
        landing_point = (
            choose_landing_point(trajectory, fps, court_region)
            if len(trajectory) >= 3
            else None
        )
        decision = check_landing_inside_court(
            landing_point,
            court_region,
        )

        # Use the landing frame when available; otherwise show the final
        # tracked frame so we can inspect exactly what the tracker followed.
        if landing_point:
            debug_frame_index = int(landing_point["frame"])
        elif trajectory:
            debug_frame_index = int(trajectory[-1]["frame"])
        else:
            debug_frame_index = max(0, total_frames - 1)

        cap = cv2.VideoCapture(input_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, debug_frame_index)
        success, debug_frame = cap.read()
        cap.release()

        if not success:
            debug_frame = first_frame

        debug_frame = create_debug_frame(
            debug_frame,
            trajectory,
            court_region,
            landing_point,
            decision,
        )

        result = decision.get("result", "UNKNOWN") if isinstance(decision, dict) else "UNKNOWN"
        cv2.putText(
            debug_frame,
            f"Confidence: {tracking_confidence:.2f}",
            (40, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        ok, encoded = cv2.imencode(".jpg", debug_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "Could not encode debug image"},
            )

        return Response(
            content=encoded.tobytes(),
            media_type="image/jpeg",
            headers={
                "Content-Disposition": 'inline; filename="shuttleeye_debug.jpg"',
                "X-ShuttleEye-Result": str(result),
                "X-ShuttleEye-Tracked-Points": str(len(trajectory)),
                "X-ShuttleEye-Confidence": f"{tracking_confidence:.3f}",
                "X-ShuttleEye-Debug-Frame": str(debug_frame_index),
            },
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Debug analysis failed: {str(e)}",
            },
        )
    finally:
        if input_path and os.path.exists(input_path):
            try:
                os.remove(input_path)
            except Exception:
                pass


@app.post("/analyze-video")
async def analyze_video(video: UploadFile = File(...)):
    input_path = None
    started = time.time()

    try:
        file_id = str(uuid.uuid4())
        safe_filename = os.path.basename(video.filename or "video.mp4")
        input_path = f"/tmp/{file_id}_{safe_filename}"

        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Could not open video",
                },
            )

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        success, first_frame = cap.read()
        if not success:
            cap.release()
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Could not read video frames",
                },
            )

        court_analysis = detect_court(first_frame)
        court_corners = get_court_corners(first_frame)
        court_analysis["corners"] = court_corners

        # Court geometry is orientation-independent. The phone can be placed
        # anywhere around/inside the court; no left/right or colour assumption.
        court_region, court_region_mode = build_court_region(
            first_frame,
            court_corners,
        )

        static_line_mask = build_static_line_mask(first_frame, court_analysis)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        motion_frames = 0
        raw_candidate_count = 0
        frame_candidates_by_frame = []
        previous_gray = None
        previous_previous_gray = None
        processed_frames = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            current_gray, candidates = extract_frame_candidates(
                frame,
                previous_gray,
                previous_previous_gray,
                static_line_mask=static_line_mask,
            )

            if candidates:
                motion_frames += 1
                raw_candidate_count += len(candidates)

            frame_candidates_by_frame.append(candidates)

            previous_previous_gray = previous_gray
            previous_gray = current_gray
            processed_frames += 1

        cap.release()

        trajectory, tracking_confidence = track_shuttle(
            frame_candidates_by_frame
        )

        # Three points are the minimum for a meaningful trajectory estimate.
        shuttle_detected = len(trajectory) >= 3

        estimated_landing_point = (
            choose_landing_point(trajectory, fps, court_region)
            if shuttle_detected
            else None
        )

        if court_region_mode == "geometry_polygon":
            landing_decision = check_landing_inside_court(
                estimated_landing_point,
                court_region,
            )
        else:
            landing_decision = {
                "result": "UNKNOWN",
                "inside": None,
                "boundary_distance_pixels": None,
                "message": (
                    "Court boundary is not reliable enough for an IN/OUT call. "
                    "Reposition the phone so multiple court boundary lines and "
                    "court corners are clearly visible."
                ),
            }

        motion_percentage = (
            motion_frames / processed_frames * 100.0
            if processed_frames > 0
            else 0.0
        )

        overall_confidence = tracking_confidence

        elapsed = round(time.time() - started, 2)

        return {
            "status": "success",
            "message": "Video analyzed successfully",
            "processing_seconds": elapsed,
            "video_info": {
                "filename": safe_filename,
                "fps": round(fps, 3),
                "total_frames": total_frames,
                "duration_seconds": round(duration, 2),
                "resolution": {
                    "width": width,
                    "height": height,
                },
            },
            "motion_analysis": {
                "processed_frames": processed_frames,
                "frames_with_motion": motion_frames,
                "motion_percentage": round(motion_percentage, 2),
            },
            "court_analysis": {
                "court_detected": bool(
                    court_analysis.get("court_detected")
                ),
                "court_lines_detected": int(
                    court_analysis.get("court_lines_detected", 0)
                ),
                "corners": court_corners,
                "playable_region": court_region,
                "region_mode": court_region_mode,
                "setup": {
                    "status": (
                        "READY"
                        if court_region_mode == "geometry_polygon"
                        else "REPOSITION_PHONE"
                    ),
                    "instruction": (
                        "Court geometry detected. Keep the phone steady."
                        if court_region_mode == "geometry_polygon"
                        else
                        "Move the phone until multiple court boundary lines "
                        "and the court corners are clearly visible."
                    ),
                    "colour_independent": True,
                    "orientation_independent": True,
                },
            },
            "shuttle_detection": {
                "shuttle_detected": bool(shuttle_detected),
                "candidate_points": int(raw_candidate_count),
                "tracked_points": int(len(trajectory)),
                "tracking_confidence": round(
                    float(tracking_confidence), 3
                ),
            },
            "trajectory": trajectory[-30:],
            "estimated_landing_point": estimated_landing_point,
            "decision": {
                **landing_decision,
                "confidence": round(float(overall_confidence), 3),
            },
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Video analysis failed: {str(e)}",
            },
        )

    finally:
        if input_path and os.path.exists(input_path):
            try:
                os.remove(input_path)
            except OSError:
                pass
