from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from court_corners import get_court_corners
from court_geometry import infer_court_region_from_lines
from runtime_analyzer import LiveFrameAnalyzer
from baseline_geometry import classify_baseline_landing, match_baseline_observation, normalize_baseline_config
import asyncio
import cv2
import numpy as np
import os
import tempfile
import uuid
import math
import time
import json
import logging

app = FastAPI(
    title="ShuttleEye AI",
    description="AI-powered badminton shuttle detection and line-call analysis",
    version="2.1.5",
)

MAX_UPLOAD_BYTES = 250 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
logger = logging.getLogger(__name__)


async def save_video_upload(upload, destination):
    """Stream uploads to disk with a bounded size to protect worker memory."""
    size = 0
    with open(destination, "wb") as output:
        while True:
            chunk = await upload.read(UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise ValueError("Video exceeds the 250 MB upload limit")
            output.write(chunk)
    if size == 0:
        raise ValueError("Uploaded video is empty")
    return size


@app.get("/")
def home():
    return {
        "message": "Welcome to ShuttleEye AI",
        "status": "running",
        "version": "2.1.5",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


def _index_live_frame_update(update, received_frame_index, legacy_frame_id=None):
    """Correlate each binary WebSocket message, even when JPEG decoding fails."""
    indexed = dict(update)
    indexed.setdefault("analysis_frame_index", indexed.get("frame_index"))
    indexed["frame_index"] = received_frame_index
    if legacy_frame_id is not None:
        indexed["frame_id"] = legacy_frame_id
    return indexed


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

    # Report line intersections as geometry evidence; the detector does not
    # promote them to court corners unless a complete supported quadrilateral
    # is independently validated below.
    intersections = []
    for i, first in enumerate(detected_lines[:24]):
        for second in detected_lines[i + 1:24]:
            point = _line_intersection(first, second)
            if point is not None and -0.02 * width <= point[0] <= 1.02 * width and -0.02 * height <= point[1] <= 1.02 * height:
                if all(math.hypot(point[0] - q["x"], point[1] - q["y"]) > 12 for q in intersections):
                    intersections.append({"x": round(point[0], 1), "y": round(point[1], 1)})
    return {
        "court_detected": False,  # geometry is established by build_court_region
        "court_lines_detected": len(detected_lines),
        "lines": detected_lines[:40],
        "intersections": intersections[:40],
    }


def _line_intersection(a, b):
    x1, y1, x2, y2 = (a["x1"], a["y1"], a["x2"], a["y2"])
    x3, y3, x4, y4 = (b["x1"], b["y1"], b["x2"], b["y2"])
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    det1, det2 = x1 * y2 - y1 * x2, x3 * y4 - y3 * x4
    return ((det1 * (x3 - x4) - (x1 - x2) * det2) / denom,
            (det1 * (y3 - y4) - (y1 - y2) * det2) / denom)


def build_court_region(frame, court_corners, court_lines=None):
    """Return a playable polygon only when a four-sided region is supported.

    A partial view may contain useful line structure, but without enough
    evidence to locate all court boundaries, the playable side/extent is
    ambiguous. That case intentionally returns no region for safe UNKNOWN.
    """
    if frame is None:
        return None, "court_not_detected"

    def partial_region():
        region, diagnostics = infer_court_region_from_lines(frame, court_lines or [])
        return (region, "partial_geometry_polygon") if diagnostics["reliable"] else (None, "insufficient_geometry")

    if not court_corners:
        return partial_region()

    try:
        if isinstance(court_corners, dict):
            keys = ("top_left", "top_right", "bottom_right", "bottom_left")
            if not all(k in court_corners for k in keys):
                return partial_region()
            pts = np.array([court_corners[k] for k in keys], dtype=np.float32)
        else:
            pts = np.array(list(court_corners), dtype=np.float32)

        if pts.shape != (4, 2) or not np.isfinite(pts).all():
            return partial_region()

        h, w = frame.shape[:2]
        hull = cv2.convexHull(pts).reshape(-1, 2)
        if len(hull) != 4 or not cv2.isContourConvex(hull.reshape(-1, 1, 2)):
            return partial_region()

        polygon = hull.astype(np.float32)
        area_ratio = abs(float(cv2.contourArea(polygon))) / float(max(1, w * h))
        side_lengths = [
            float(np.linalg.norm(polygon[(i + 1) % 4] - polygon[i]))
            for i in range(4)
        ]

        if min(side_lengths) < max(35.0, min(w, h) * 0.025):
            return partial_region()
        if max(side_lengths) / max(min(side_lengths), 1.0) > 18.0:
            return partial_region()

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
        # Verify boundary support with image gradients rather than brightness:
        # a light-coloured floor is not itself evidence of a court marking.
        gray_geom = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        geom_mask = cv2.Canny(cv2.GaussianBlur(gray_geom, (5, 5), 0), 35, 130)

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
                    if patch.size and float(np.mean(patch > 0)) >= 0.12:
                        hits += 1
            edge_support.append(hits / float(samples))

        near_frame = 0
        frame_tol_x = max(8.0, 0.025 * w)
        frame_tol_y = max(8.0, 0.025 * h)
        for x, y in polygon:
            if x <= frame_tol_x or x >= (w - 1 - frame_tol_x) or y <= frame_tol_y or y >= (h - 1 - frame_tol_y):
                near_frame += 1

        # Every side must have image-gradient support. Vertices may touch the
        # frame, but a clipped polygon cannot establish an unseen boundary.
        support_ok = min(edge_support) >= 0.14 and near_frame <= 4

        if area_ratio <= 0.65 and geometry_score >= 0.55 and support_ok:
            if court_lines:
                line_region, line_diagnostics = infer_court_region_from_lines(frame, court_lines)
                if not line_diagnostics.get("reliable"):
                    return partial_region()
                line_polygon = np.asarray(line_region, dtype=np.float32)
                variants = [np.roll(line_polygon, shift, axis=0) for shift in range(4)]
                variants += [np.roll(line_polygon[::-1], shift, axis=0) for shift in range(4)]
                agreement = min(float(np.mean(np.linalg.norm(polygon - variant, axis=1))) for variant in variants)
                if agreement > math.hypot(w, h) * 0.08:
                    # The corner detector and court-line network disagree;
                    # use only the uniquely supported line solution.
                    return line_region, "partial_geometry_polygon"
            return polygon.astype(np.int32).tolist(), "geometry_polygon"

        # A single line and some perpendicular segments do not establish
        # which side is the playable court or where its far boundary lies.
        # Returning a frame-spanning half-plane here caused confident, but
        # unsupported, calls on partial views. Until a complete court polygon
        # is supported by four visible boundaries, fail closed.
        return partial_region()

    except Exception:
        return partial_region()

def check_landing_inside_court(landing_point, court_corners, margin=14):
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

        distance = abs(float(signed_distance))
        if distance <= 1e-6:
            return {
                "result": "IN",
                "inside": True,
                "boundary_distance_pixels": 0.0,
                "message": "Shuttle touched the boundary line; boundary contact is in",
            }
        if distance <= float(margin):
            return {
                "result": "UNKNOWN",
                "inside": None,
                "boundary_distance_pixels": round(distance, 2),
                "reason": "Landing is too close to a boundary for a safe call",
            }

        if signed_distance > 0:
            return {
                "result": "IN",
                "inside": True,
                "boundary_distance_pixels": round(float(signed_distance), 2),
                "message": "Shuttle landed inside court",
            }

        return {
            "result": "OUT",
            "inside": False,
            "boundary_distance_pixels": round(distance, 2),
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

            # Court lines and other long bright fixtures are a major
            # false-positive source. A shuttle can be slightly elongated,
            # but a long white contour is much more likely to be a line,
            # screen edge, text stroke, or fixture than the shuttle.
            aspect = max(w, h) / max(1, min(w, h))
            if aspect > 4.5 and yellow_ratio < 0.20:
                continue

            # Static court lines are a major false-positive source. Suppress
            # candidates sitting on a detected court line unless they have
            # strong yellow/compact-shuttle evidence.
            if line_ratio > 0.22:
                if not (yellow_ratio > 0.25 and circularity > 0.10 and area < 180.0):
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
    total_frames = len(frame_candidates_by_frame)
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

    # A path that exists on almost every frame is suspicious for a fixed
    # court line, screen edge, camera artifact, or another persistent object.
    # A real shuttle should normally occupy only part of the clip. Be
    # conservative here so a persistent false track cannot produce IN/OUT.
    coverage = len(best_path) / max(1, total_frames)
    if coverage > 0.90:
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

    label = f"ShuttleEye 2.1.5 | {result or 'TRACKING'}"
    cv2.rectangle(debug, (20, 20), (650, 90), (20, 20, 20), -1)
    cv2.putText(
        debug, label, (40, 68),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3,
        cv2.LINE_AA,
    )

    return debug


def choose_landing_point(trajectory, fps, court_corners=None, max_frame_gap=None, frame_size=None):
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

    if frame_size is not None:
        if len(frame_size) < 2:
            return None
        h_limit, w_limit = int(frame_size[0]), int(frame_size[1])
        if h_limit <= 0 or w_limit <= 0:
            return None
        last = trajectory[-1]
        if (last["y"] < 40 or last["y"] > h_limit - 20
                or last["x"] < 5 or last["x"] > w_limit - 5):
            return None

    x_span = max(p["x"] for p in trajectory) - min(p["x"] for p in trajectory)
    y_span = max(p["y"] for p in trajectory) - min(p["y"] for p in trajectory)
    if math.hypot(x_span, y_span) < 50.0:
        return None

    last = trajectory[-1]

    best_idx = None
    best_score = -1e9

    for i in range(2, len(trajectory) - 1):
        p0 = trajectory[i - 2]
        p1 = trajectory[i - 1]
        p2 = trajectory[i]
        p3 = trajectory[i + 1]

        frame_gaps = (int(p1["frame"]) - int(p0["frame"]),
                      int(p2["frame"]) - int(p1["frame"]),
                      int(p3["frame"]) - int(p2["frame"]))
        # A large tracker gap around contact cannot establish a genuine
        # descent/reversal; the apparent bounce may be an unrelated object.
        if any(gap < 1 or (max_frame_gap is not None and gap > max_frame_gap) for gap in frame_gaps):
            continue

        d1 = p1["y"] - p0["y"]
        d2 = p2["y"] - p1["y"]
        reversal = p2["y"] - p3["y"]

        # Require a sustained downward phase before the low point.
        if d1 < 2 or d2 < 2:
            continue

        # A landing/bounce candidate is a local maximum in image-Y followed
        # by a meaningful reversal or an observed near-stationary settle.
        bounce_evidence = reversal >= 60
        settle_evidence = (
            abs(reversal) <= 15
            and i + 2 < len(trajectory)
            and int(trajectory[i + 2]["frame"]) - int(p3["frame"]) <= 2
            and abs(trajectory[i + 2]["y"] - p2["y"]) <= 15
            and math.hypot(trajectory[i + 2]["x"] - p2["x"], trajectory[i + 2]["y"] - p2["y"]) <= 20
        )

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

    method = ("downward_phase_bounce"
              if p["y"] - trajectory[best_idx + 1]["y"] >= 60
              else "downward_phase_settle")

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


def analyze_live_court_frame(frame):
    """Redetect court geometry on every live frame; never reuse stale pose."""
    analysis = detect_court(frame)
    corners = get_court_corners(frame)
    region, mode = build_court_region(frame, corners, analysis.get("lines", []))
    reliable = mode in ("geometry_polygon", "partial_geometry_polygon") and region is not None
    confidence = 0.0
    diagnostics = {"reliable": False, "reason": mode}
    if reliable:
        confidence = 0.78 if mode == "geometry_polygon" else 0.72
        diagnostics = {"reliable": True, "reason": mode}
        if mode == "partial_geometry_polygon":
            _, diagnostics = infer_court_region_from_lines(frame, analysis.get("lines", []))
    return {
        "region": region,
        "mode": mode,
        "confidence": confidence,
        "reliable": reliable,
        "diagnostics": diagnostics,
        "lines": analysis.get("lines", []),
    }


def analyze_live_baseline_frame(frame, config):
    """Validate only the explicitly user-marked baseline; do not infer a court polygon."""
    started = time.perf_counter()
    analysis = detect_court(frame)
    height, width = frame.shape[:2]
    match = match_baseline_observation(analysis.get("lines", []), config, width, height)
    return {**match, "lines": analysis.get("lines", []),
            "processing_ms": (time.perf_counter() - started) * 1000.0}


def create_live_analyzer():
    """Build a per-connection state machine with the shared vision primitives."""
    return LiveFrameAnalyzer(
        court_analyzer=analyze_live_court_frame,
        candidate_extractor=extract_frame_candidates,
        tracker=track_shuttle,
        landing_estimator=lambda trajectory, fps, region: choose_landing_point(
            trajectory, fps, region, max_frame_gap=2
        ),
        decision_maker=lambda landing, region: check_landing_inside_court(landing, region),
        static_mask_builder=lambda frame, lines: build_static_line_mask(frame, {"lines": lines}),
        baseline_validator=analyze_live_baseline_frame,
        baseline_decision_maker=classify_baseline_landing,
    )


@app.websocket("/ws/analyze-live")
async def analyze_live(websocket: WebSocket):
    """Live endpoint with explicit baseline setup and legacy binary-only support.

    New clients send one JSON baseline_calibration message followed by binary
    JPEG frames. Legacy clients that begin with binary frames keep the existing
    full-court analysis path. Each processed frame receives a JSON update.
    """
    await websocket.accept()
    analyzer = create_live_analyzer()
    baseline_mode_requested = False
    received_frame_index = 0
    legacy_next_frame_id = None
    await websocket.send_json({
        "state": "SEARCHING_FOR_COURT",
        "message": "Send binary JPEG frames to begin.",
        "decision": {"result": "UNKNOWN", "reason": "waiting_for_camera_frames"},
    })
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            encoded = message.get("bytes")
            if encoded is None and message.get("text") is not None:
                try:
                    control = json.loads(message["text"])
                except (TypeError, ValueError, json.JSONDecodeError):
                    control = None
                if isinstance(control, dict) and control.get("type") == "session":
                    session_id = control.get("session_id")
                    first_frame_id = control.get("next_frame_id", 0)
                    valid_session = (
                        isinstance(session_id, str) and 1 <= len(session_id) <= 80
                        and isinstance(first_frame_id, int) and not isinstance(first_frame_id, bool)
                        and 0 <= first_frame_id <= 2**53 - 1
                    )
                    if valid_session:
                        analyzer._session_id = session_id
                        legacy_next_frame_id = first_frame_id
                    else:
                        await websocket.send_json({"state": "UNKNOWN", "message": "Invalid live-session handshake."})
                    continue

                baseline_mode_requested = True
                config = normalize_baseline_config(control)
                if config is None or not analyzer.configure_baseline(config):
                    await websocket.send_json({"state": "BASELINE_SEARCH", "baseline_valid": False,
                                               "decision": {"result": "UNKNOWN", "reason": "baseline_confirmation_invalid"},
                                               "message": "Mark the intended baseline endpoints and the court side again."})
                    continue
                await websocket.send_json(analyzer._baseline_snapshot(reason="baseline_confirmation_started"))
                continue
            if encoded is not None:
                received_frame_index += 1
            legacy_frame_id = legacy_next_frame_id
            if legacy_next_frame_id is not None:
                legacy_next_frame_id += 1
            if encoded is not None and baseline_mode_requested and analyzer._baseline_config is None:
                await websocket.send_json(_index_live_frame_update({
                    "mode": "baseline_live", "state": "BASELINE_SEARCH", "baseline_valid": False,
                    "baseline_locked": False,
                    "decision": {"result": "UNKNOWN", "reason": "baseline_confirmation_required"},
                    "message": "Confirm the intended baseline before live line judging.",
                }, received_frame_index, legacy_frame_id))
                continue
            if encoded is None:
                await websocket.close(code=1003, reason="Expected a binary JPEG frame")
                return
            if not encoded or len(encoded) > 8 * 1024 * 1024:
                await websocket.send_json(_index_live_frame_update({
                    "state": analyzer.state.value,
                    "decision": {"result": "UNKNOWN", "reason": "invalid_or_oversized_frame"},
                }, received_frame_index, legacy_frame_id))
                continue
            frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                await websocket.send_json(_index_live_frame_update({
                    "state": analyzer.state.value,
                    "decision": {"result": "UNKNOWN", "reason": "frame_decode_failed"},
                }, received_frame_index, legacy_frame_id))
                continue
            update = await asyncio.to_thread(analyzer.process_frame, frame)
            await websocket.send_json(_index_live_frame_update(update, received_frame_index, legacy_frame_id))
    except WebSocketDisconnect:
        return
    except Exception:
        # Keep protocol and server failures diagnosable by clients while avoiding
        # exposing internal exception details over the public WebSocket.
        logger.exception("Live WebSocket frame processing failed")
        try:
            await websocket.close(code=1011, reason="Live frame processing failed")
        except RuntimeError:
            pass


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

        await save_video_upload(video, input_path)

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
        court_region, court_region_mode = build_court_region(first_frame, court_corners, court_analysis.get("lines"))
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
        shuttle_detected = (
            len(trajectory) >= 3
            and float(tracking_confidence) >= 0.45
        )
        landing_point = (
            choose_landing_point(trajectory, fps, court_region)
            if shuttle_detected
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

    except ValueError as e:
        return JSONResponse(status_code=413 if "250 MB" in str(e) else 400,
                            content={"status": "error", "message": str(e)})
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
        input_path = os.path.join(tempfile.gettempdir(), f"{file_id}_{safe_filename}")

        await save_video_upload(video, input_path)

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
            court_analysis.get("lines"),
        )
        reliable_geometry = court_region_mode in ("geometry_polygon", "partial_geometry_polygon")
        geometry_confidence = 0.72 if court_region_mode == "geometry_polygon" else (0.68 if reliable_geometry else 0.0)
        geometry_diagnostics = {
            "reliable": reliable_geometry,
            "reason": ("partial_view_supported_by_two_line_families" if court_region_mode == "partial_geometry_polygon"
                       else "validated_quadrilateral" if court_region_mode == "geometry_polygon"
                       else "four supported boundary edges and an internal court marking were not established"),
        }
        if court_region_mode == "partial_geometry_polygon":
            _, geometry_diagnostics = infer_court_region_from_lines(first_frame, court_analysis.get("lines", []))
        court_analysis["court_detected"] = geometry_confidence > 0

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

        # Three points are the minimum for a trajectory, but a trajectory
        # alone is not enough for a public IN/OUT call. Require meaningful
        # tracker confidence so persistent bright objects cannot become a
        # false shuttle.
        shuttle_detected = (
            len(trajectory) >= 3
            and float(tracking_confidence) >= 0.45
        )

        estimated_landing_point = (
            choose_landing_point(trajectory, fps, court_region)
            if shuttle_detected
            else None
        )

        if reliable_geometry:
            landing_decision = check_landing_inside_court(
                estimated_landing_point,
                court_region,
                margin=max(10.0, math.hypot(width, height) * 0.008),
            )
            # Only a clearly observed contact reversal plus a strong track can
            # support a line call. A terminal image point or weak track stays
            # UNKNOWN even when its coordinate falls inside the polygon.
            if (landing_decision.get("result") != "UNKNOWN"
                    and (tracking_confidence < 0.65
                         or not estimated_landing_point
                         or estimated_landing_point.get("method") != "downward_phase_bounce")):
                landing_decision = {
                    "result": "UNKNOWN",
                    "inside": None,
                    "boundary_distance_pixels": landing_decision.get("boundary_distance_pixels"),
                    "reason": "Strong track and observed contact reversal are required",
                }
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

        overall_confidence = min(float(tracking_confidence), float(geometry_confidence))

        elapsed = round(time.time() - started, 2)

        payload = {
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
                "lines": court_analysis.get("lines", []),
                "intersections": court_analysis.get("intersections", []),
                "corners": court_corners,
                "playable_region": court_region,
                "region_mode": court_region_mode,
                "geometry_confidence": geometry_confidence,
                "diagnostics": geometry_diagnostics,
                "setup": {
                    "status": (
                        "READY"
                        if reliable_geometry
                        else "REPOSITION_PHONE"
                    ),
                    "instruction": (
                        "Court geometry detected. Keep the phone steady."
                        if reliable_geometry
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
                "selected_track_score": round(float(tracking_confidence), 3),
                "rejection_reasons": ([] if shuttle_detected else [
                    "no sufficiently long and moving candidate track"
                ]),
            },
            "setup": {
                "status": "READY" if reliable_geometry else "REPOSITION_PHONE",
                "instruction": "Court geometry detected. Keep the phone steady." if reliable_geometry else "Move the phone until multiple court boundary lines and corners are clearly visible.",
            },
            "trajectory": trajectory[-30:],
            "estimated_landing_point": estimated_landing_point,
            "decision": {
                **landing_decision,
                "confidence": round(float(overall_confidence), 3),
            },
        }
        # Keep legacy top-level fields while also providing the mobile app's
        # expected data.decision.result envelope.
        payload["court_detection"] = payload["court_analysis"]
        payload["data"] = {
            "status": payload["status"],
            "message": payload["message"],
            "processing_seconds": payload["processing_seconds"],
            "video_info": payload["video_info"],
            "motion_analysis": payload["motion_analysis"],
            "court_detection": payload["court_detection"],
            "setup": payload["court_detection"]["setup"],
            "shuttle_detection": payload["shuttle_detection"],
            "trajectory": payload["trajectory"],
            "estimated_landing_point": payload["estimated_landing_point"],
            "decision": payload["decision"],
        }
        return payload

    except ValueError as e:
        return JSONResponse(status_code=413 if "250 MB" in str(e) else 400,
                            content={"status": "error", "message": str(e)})
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
