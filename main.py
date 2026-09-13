from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from court_corners import get_court_corners
import cv2
import numpy as np
import os
import uuid
import shutil
import math


app = FastAPI(
    title="ShuttleEye AI",
    description="AI-powered badminton shuttle detection and video analysis",
    version="0.9.0"
)


@app.get("/")
def home():
    return {
        "message": "Welcome to ShuttleEye AI",
        "status": "running",
        "version": "0.9.0"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


def detect_court(frame):
    """Detect visible court lines for diagnostics."""
    if frame is None:
        return {
            "court_detected": False,
            "court_lines_detected": 0,
            "lines": []
        }

    height, width = frame.shape[:2]
    scale = min(1.0, 900.0 / max(width, height))

    small = cv2.resize(
        frame,
        (max(1, int(width * scale)), max(1, int(height * scale)))
    )

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges1 = cv2.Canny(gray, 40, 120)
    edges2 = cv2.Canny(gray, 70, 180)
    edges = cv2.bitwise_or(edges1, edges2)

    min_len = max(35, int(min(small.shape[:2]) * 0.10))

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=45,
        minLineLength=min_len,
        maxLineGap=25
    )

    detected_lines = []

    if lines is not None:
        for raw in lines:
            x1, y1, x2, y2 = np.asarray(raw).reshape(-1)[:4]
            length = float(np.hypot(x2 - x1, y2 - y1))

            if length < min_len:
                continue

            detected_lines.append({
                "x1": int(round(x1 / scale)),
                "y1": int(round(y1 / scale)),
                "x2": int(round(x2 / scale)),
                "y2": int(round(y2 / scale))
            })

    detected_lines.sort(
        key=lambda item: np.hypot(
            item["x2"] - item["x1"],
            item["y2"] - item["y1"]
        ),
        reverse=True
    )

    return {
        "court_detected": len(detected_lines) >= 4,
        "court_lines_detected": len(detected_lines),
        "lines": detected_lines[:40]
    }


def check_landing_inside_court(landing_point, court_corners, margin=0):
    """
    Classify landing point against the detected court polygon.

    margin=0 means the exact detected boundary is used.
    """
    if landing_point is None:
        return {
            "result": "UNKNOWN",
            "inside": None,
            "reason": "Landing point not detected"
        }

    if court_corners is None:
        return {
            "result": "UNKNOWN",
            "inside": None,
            "reason": "Court corners not detected"
        }

    try:
        polygon = np.array([
            court_corners["top_left"],
            court_corners["top_right"],
            court_corners["bottom_right"],
            court_corners["bottom_left"]
        ], dtype=np.float32)

        point = (
            float(landing_point["x"]),
            float(landing_point["y"])
        )

        # Positive = inside, zero = on line, negative = outside.
        signed_distance = cv2.pointPolygonTest(
            polygon,
            point,
            True
        )

        if signed_distance >= -float(margin):
            return {
                "result": "IN",
                "inside": True,
                "boundary_distance_pixels": round(float(signed_distance), 2),
                "message": "Shuttle landed inside court"
            }

        return {
            "result": "OUT",
            "inside": False,
            "boundary_distance_pixels": round(float(signed_distance), 2),
            "message": "Shuttle landed outside court"
        }

    except Exception as e:
        return {
            "result": "UNKNOWN",
            "inside": None,
            "reason": f"Landing decision error: {str(e)}"
        }


def _candidate_visual_score(gray, hsv, contour, x, y, w, h, area):
    """
    Score how shuttle-like a moving blob looks.

    This is intentionally a classical-CV stage. It is not yet a trained
    neural shuttle detector.
    """
    if w <= 0 or h <= 0:
        return 0.0, 0.0, 0.0

    roi_gray = gray[y:y + h, x:x + w]
    roi_hsv = hsv[y:y + h, x:x + w]

    if roi_gray.size == 0:
        return 0.0, 0.0, 0.0

    # Shuttle feathers/cork are usually bright and relatively low saturation.
    sat = roi_hsv[:, :, 1]
    val = roi_hsv[:, :, 2]

    bright = val > 145
    low_sat = sat < 125
    white_ratio = float(np.mean(bright & low_sat))

    mean_value = float(np.mean(val)) / 255.0

    perimeter = cv2.arcLength(contour, True)
    compactness = 0.0
    if perimeter > 1.0:
        compactness = min(
            1.0,
            float(4.0 * math.pi * area / (perimeter * perimeter))
        )

    # Very small blobs are common noise; extremely large blobs are usually
    # players/rackets rather than the shuttle.
    if 3 <= area <= 100:
        area_score = 1.0
    elif area < 3:
        area_score = 0.0
    else:
        area_score = max(0.0, 1.0 - (area - 100.0) / 400.0)

    # Keep a little shape information without requiring a perfect circle.
    aspect = max(w, h) / max(1, min(w, h))
    shape_score = 1.0 if aspect <= 4.5 else max(
        0.0, 1.0 - (aspect - 4.5) / 4.0
    )

    visual = (
        0.45 * white_ratio +
        0.20 * mean_value +
        0.15 * compactness +
        0.12 * area_score +
        0.08 * shape_score
    )

    return float(visual), float(white_ratio), float(compactness)


def extract_frame_candidates(frame, previous_gray):
    """
    Produce a small set of plausible moving shuttle candidates for one frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if previous_gray is None:
        return gray, []

    difference = cv2.absdiff(previous_gray, gray)

    # Motion mask.
    _, motion = cv2.threshold(
        difference,
        25,
        255,
        cv2.THRESH_BINARY
    )

    kernel3 = np.ones((3, 3), np.uint8)
    motion = cv2.morphologyEx(
        motion,
        cv2.MORPH_OPEN,
        kernel3,
        iterations=1
    )
    motion = cv2.dilate(motion, kernel3, iterations=1)

    # Bright/low-saturation mask. This helps suppress many dark moving objects.
    bright_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 135], dtype=np.uint8),
        np.array([180, 145, 255], dtype=np.uint8)
    )

    # Use motion ∩ bright first. If that is too sparse, also inspect motion-only.
    combined = cv2.bitwise_and(motion, bright_mask)

    contours, _ = cv2.findContours(
        combined,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    def collect(contour_list, allow_motion_only=False):
        for contour in contour_list:
            area = float(cv2.contourArea(contour))
            if not (2.0 <= area <= 350.0):
                continue

            x, y, w, h = cv2.boundingRect(contour)

            if w > 90 or h > 90:
                continue

            if w < 2 or h < 2:
                continue

            visual, white_ratio, compactness = _candidate_visual_score(
                gray, hsv, contour, x, y, w, h, area
            )

            if not allow_motion_only and white_ratio < 0.12:
                continue

            center_x = x + w / 2.0
            center_y = y + h / 2.0

            candidates.append({
                "x": float(center_x),
                "y": float(center_y),
                "area": round(area, 2),
                "w": int(w),
                "h": int(h),
                "visual_score": round(visual, 4),
                "white_ratio": round(white_ratio, 4),
                "compactness": round(compactness, 4)
            })

    collect(contours, allow_motion_only=False)

    # If bright-motion intersection produced nothing, use motion blobs with
    # a stricter visual score. This preserves fast/blurred shuttle detections.
    if not candidates:
        motion_contours, _ = cv2.findContours(
            motion,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        collect(motion_contours, allow_motion_only=True)
        candidates = [
            c for c in candidates
            if c["visual_score"] >= 0.28
        ]

    # Keep only the strongest few candidates per frame. This prevents the
    # trajectory optimizer from being overwhelmed by Hough-like noise.
    candidates.sort(
        key=lambda c: (
            c["visual_score"],
            c["white_ratio"],
            -c["area"]
        ),
        reverse=True
    )

    return gray, candidates[:12]


def _transition_score(previous, current, previous_previous=None, frame_gap=1):
    """
    Score a candidate transition using position continuity and, when
    available, velocity continuity.
    """
    dx = current["x"] - previous["x"]
    dy = current["y"] - previous["y"]
    distance = math.hypot(dx, dy)

    # Allow larger motion for larger frame gaps.
    max_jump = 520.0 * max(1, frame_gap)
    if distance > max_jump:
        return -1e9

    # Badminton shuttle can move very quickly, but random contour jumps are
    # usually much larger or directionally inconsistent.
    distance_penalty = distance / max_jump

    score = -1.25 * distance_penalty

    if previous_previous is not None:
        pdx = previous["x"] - previous_previous["x"]
        pdy = previous["y"] - previous_previous["y"]
        previous_distance = math.hypot(pdx, pdy)

        if previous_distance > 1.0:
            # Compare direction of current motion with prior motion.
            dot = (pdx * dx + pdy * dy) / (
                previous_distance * max(distance, 1.0)
            )
            dot = max(-1.0, min(1.0, dot))
            direction_similarity = (dot + 1.0) / 2.0

            # Allow speed changes, but prefer smooth movement.
            speed_ratio = distance / previous_distance
            speed_consistency = math.exp(
                -abs(math.log(max(speed_ratio, 0.05)))
            )

            score += 0.75 * direction_similarity
            score += 0.45 * speed_consistency

    score += 1.6 * current["visual_score"]

    return float(score)


def track_shuttle(frame_candidates_by_frame, total_frames):
    """
    Greedy tracking with short-gap prediction.

    We keep one candidate per useful frame. The important difference from the
    old implementation is that ALL motion contours are no longer appended to
    the trajectory.
    """
    usable = [
        (frame_no, candidates)
        for frame_no, candidates in enumerate(frame_candidates_by_frame)
        if candidates
    ]

    if not usable:
        return [], 0.0

    # Start at the strongest visually supported candidate.
    start_frame, start_candidates = max(
        usable,
        key=lambda item: max(c["visual_score"] for c in item[1])
    )
    current = max(
        start_candidates,
        key=lambda c: c["visual_score"]
    )

    path = [{
        "frame": start_frame,
        "x": int(round(current["x"])),
        "y": int(round(current["y"]))
    }]

    previous = current
    previous_previous = None
    previous_frame = start_frame

    for frame_no in range(start_frame + 1, total_frames):
        candidates = frame_candidates_by_frame[frame_no]

        if not candidates:
            if frame_no - previous_frame > 3:
                break
            continue

        gap = frame_no - previous_frame

        best = None
        best_score = -1e9

        for candidate in candidates:
            score = _transition_score(
                previous,
                candidate,
                previous_previous,
                frame_gap=gap
            )

            if score > best_score:
                best_score = score
                best = candidate

        if best is None or best_score < -0.9:
            # A bad candidate should not teleport the trajectory.
            if gap <= 2:
                continue
            break

        previous_previous = previous
        previous = best
        previous_frame = frame_no

        path.append({
            "frame": frame_no,
            "x": int(round(best["x"])),
            "y": int(round(best["y"]))
        })

    # Normalize confidence from path continuity and length.
    if len(path) < 3:
        confidence = 0.0
    else:
        jumps = []
        for a, b in zip(path[:-1], path[1:]):
            gap = max(1, b["frame"] - a["frame"])
            jumps.append(
                math.hypot(
                    b["x"] - a["x"],
                    b["y"] - a["y"]
                ) / (520.0 * gap)
            )

        smoothness = max(
            0.0,
            1.0 - float(np.mean(np.clip(jumps, 0.0, 1.0)))
        )
        length_score = min(1.0, len(path) / 12.0)
        confidence = 0.55 * smoothness + 0.45 * length_score

    return path, float(confidence)


def choose_landing_point(trajectory, fps):
    """
    Estimate the landing location from the stable end of the tracked path.

    We use a robust weighted average of the final points instead of blindly
    selecting the last raw motion contour.
    """
    if len(trajectory) < 3:
        return None

    recent = trajectory[-5:]

    weights = np.arange(1, len(recent) + 1, dtype=np.float32)

    xs = np.array([p["x"] for p in recent], dtype=np.float32)
    ys = np.array([p["y"] for p in recent], dtype=np.float32)

    x = int(round(float(np.average(xs, weights=weights))))
    y = int(round(float(np.average(ys, weights=weights))))
    frame = int(recent[-1]["frame"])

    return {
        "frame": frame,
        "x": x,
        "y": y,
        "method": "weighted_recent_trajectory"
    }


@app.post("/analyze-video")
async def analyze_video(video: UploadFile = File(...)):
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
                content={
                    "status": "error",
                    "message": "Could not open video"
                }
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
                    "message": "Could not read video frames"
                }
            )

        court_analysis = detect_court(first_frame)
        court_corners = get_court_corners(first_frame)
        court_analysis["corners"] = court_corners

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        motion_frames = 0
        raw_candidate_count = 0
        frame_candidates_by_frame = []

        previous_gray = None
        frame_number = 0

        while True:
            success, frame = cap.read()

            if not success:
                break

            current_gray, candidates = extract_frame_candidates(
                frame,
                previous_gray
            )

            if candidates:
                motion_frames += 1
                raw_candidate_count += len(candidates)

            frame_candidates_by_frame.append(candidates)
            previous_gray = current_gray
            frame_number += 1

        cap.release()

        trajectory, tracking_confidence = track_shuttle(
            frame_candidates_by_frame,
            frame_number
        )

        shuttle_detected = len(trajectory) >= 3

        estimated_landing_point = (
            choose_landing_point(trajectory, fps)
            if shuttle_detected else None
        )

        landing_decision = check_landing_inside_court(
            estimated_landing_point,
            court_corners
        )

        motion_percentage = (
            motion_frames / total_frames * 100.0
            if total_frames > 0 else 0.0
        )

        # Overall confidence is deliberately conservative.
        overall_confidence = (
            tracking_confidence
            if court_corners is not None
            else tracking_confidence * 0.65
        )

        return {
            "status": "success",
            "message": "Video analyzed successfully",
            "video_info": {
                "fps": round(fps, 3),
                "total_frames": total_frames,
                "duration_seconds": round(duration, 2),
                "resolution": {
                    "width": width,
                    "height": height
                }
            },
            "motion_analysis": {
                "frames_with_motion": motion_frames,
                "motion_percentage": round(motion_percentage, 2)
            },
            "court_analysis": court_analysis,
            "shuttle_detection": {
                "shuttle_detected": shuttle_detected,
                "candidate_points": raw_candidate_count,
                "tracked_points": len(trajectory),
                "tracking_confidence": round(tracking_confidence, 3)
            },
            "trajectory": trajectory[-100:],
            "estimated_landing_point": estimated_landing_point,
            "decision": {
                **landing_decision,
                "confidence": round(overall_confidence, 3)
            }
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Video analysis failed: {str(e)}"
            }
        )

    finally:
        if input_path and os.path.exists(input_path):
            try:
                os.remove(input_path)
            except OSError:
                pass
