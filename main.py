from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
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
    version="1.0.0",
)


@app.get("/")
def home():
    return {
        "message": "Welcome to ShuttleEye AI",
        "status": "running",
        "version": "1.0.0",
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


def extract_frame_candidates(frame, previous_gray, previous_previous_gray=None):
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

    combined = cv2.bitwise_and(motion, appearance)

    contours, _ = cv2.findContours(
        combined,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    def collect(contour_list, motion_only=False):
        for contour in contour_list:
            area = float(cv2.contourArea(contour))
            if not (1.5 <= area <= 450.0):
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w < 2 or h < 2 or w > 70 or h > 70:
                continue

            score, appearance_ratio, yellow_ratio, circularity = _candidate_score(
                gray, hsv, contour, x, y, w, h, area
            )

            if motion_only and score < 0.30:
                continue
            if not motion_only and appearance_ratio < 0.08:
                continue

            candidates.append({
                "x": (x + w / 2.0) / scale,
                "y": (y + h / 2.0) / scale,
                "area": round(area / max(scale * scale, 1e-6), 2),
                "w": int(round(w / scale)),
                "h": int(round(h / scale)),
                "score": round(score, 4),
                "appearance_ratio": round(appearance_ratio, 4),
                "yellow_ratio": round(yellow_ratio, 4),
                "circularity": round(circularity, 4),
            })

    collect(contours, motion_only=False)

    # Fallback: motion-only blobs are useful when the shuttle is blurred.
    if not candidates:
        motion_contours, _ = cv2.findContours(
            motion,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        collect(motion_contours, motion_only=True)

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
    max_jump = 430.0 * max(1, frame_gap)
    if distance > max_jump:
        return -1e9

    score = -1.05 * (distance / max_jump)
    score += 1.8 * current["score"]

    if previous_previous is not None:
        pdx = previous["x"] - previous_previous["x"]
        pdy = previous["y"] - previous_previous["y"]
        previous_distance = math.hypot(pdx, pdy)

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
    while 0 <= frame_no < total_frames:
        candidates = frame_candidates[frame_no]

        if not candidates:
            misses += 1
            if misses > 5:
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
            if misses > 3:
                break
            frame_no += step
            continue

        misses = 0
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
    Bidirectional trajectory tracking.

    The old tracker only searched forward from one candidate. That could
    produce one-point tracks when the strongest candidate appeared late in a
    clip. This tracker searches both directions and selects the strongest
    continuous path.
    """
    usable = [
        (i, candidates)
        for i, candidates in enumerate(frame_candidates_by_frame)
        if candidates
    ]
    if not usable:
        return [], 0.0

    seed_frame, seed_candidates = max(
        usable,
        key=lambda item: max(c["score"] for c in item[1])
    )
    seed_candidates = sorted(
        seed_candidates,
        key=lambda c: c["score"],
        reverse=True,
    )[:4]

    best_path = []
    best_quality = -1e9

    for seed in seed_candidates:
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
            quality = seed["score"]
        else:
            jumps = []
            for a, b in zip(path[:-1], path[1:]):
                gap = max(1, b["frame"] - a["frame"])
                jumps.append(
                    math.hypot(b["x"] - a["x"], b["y"] - a["y"])
                    / (430.0 * gap)
                )
            smoothness = 1.0 - float(np.mean(np.clip(jumps, 0.0, 1.0)))
            length_score = min(1.0, len(path) / 14.0)
            quality = (
                0.45 * smoothness
                + 0.40 * length_score
                + 0.15 * seed["score"]
            )

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
            / (430.0 * gap)
        )

    smoothness = max(
        0.0,
        1.0 - float(np.mean(np.clip(jumps, 0.0, 1.0))),
    )
    length_score = min(1.0, len(best_path) / 14.0)
    confidence = 0.50 * smoothness + 0.50 * length_score

    return best_path, float(confidence)


def choose_landing_point(trajectory, fps):
    """
    Estimate the landing point from the end/downward phase of the trajectory.

    We prefer the lowest reliable point in the final part of the track rather
    than blindly averaging all final contours.
    """
    if len(trajectory) < 3:
        return None

    recent = trajectory[-12:]

    # Look for the point with the greatest y after the trajectory begins
    # moving downward. In a normal court-facing phone view, larger y is closer
    # to the floor.
    best = recent[-1]
    if len(recent) >= 4:
        for i in range(1, len(recent)):
            dy = recent[i]["y"] - recent[i - 1]["y"]
            if dy >= 0:
                if recent[i]["y"] >= best["y"]:
                    best = recent[i]

    # Robust weighted average of the final 3-5 points around the landing area.
    anchor_index = recent.index(best)
    start = max(0, anchor_index - 2)
    end = min(len(recent), anchor_index + 2)
    local = recent[start:end]

    if len(local) < 2:
        local = recent[-3:]

    weights = np.arange(1, len(local) + 1, dtype=np.float32)
    xs = np.array([p["x"] for p in local], dtype=np.float32)
    ys = np.array([p["y"] for p in local], dtype=np.float32)

    return {
        "frame": int(local[-1]["frame"]),
        "x": int(round(float(np.average(xs, weights=weights)))),
        "y": int(round(float(np.average(ys, weights=weights)))),
        "method": "downward_phase_weighted_trajectory",
    }


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
            choose_landing_point(trajectory, fps)
            if shuttle_detected
            else None
        )

        landing_decision = check_landing_inside_court(
            estimated_landing_point,
            court_corners,
        )

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
