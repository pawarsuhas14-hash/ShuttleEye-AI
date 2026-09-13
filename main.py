from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from court_corners import get_court_corners
import cv2
import numpy as np
import os
import uuid
import shutil


app = FastAPI(
    title="ShuttleEye AI",
    description="AI-powered badminton shuttle detection and video analysis",
    version="0.8.0"
)


@app.get("/")
def home():
    return {"message": "Welcome to ShuttleEye AI", "status": "running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


def detect_court(frame):
    """Detect court lines. Corner geometry is calculated separately."""
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

    # Two edge settings improve robustness to different lighting.
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
            x1, y1, x2, y2 = raw[0]
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length < min_len:
                continue

            detected_lines.append({
                "x1": int(round(x1 / scale)),
                "y1": int(round(y1 / scale)),
                "x2": int(round(x2 / scale)),
                "y2": int(round(y2 / scale))
            })

    # Longest lines first; this keeps the response useful without
    # overwhelming Swagger with hundreds of short Hough segments.
    def line_length(item):
        return np.hypot(
            item["x2"] - item["x1"],
            item["y2"] - item["y1"]
        )

    detected_lines.sort(key=line_length, reverse=True)

    return {
        "court_detected": len(detected_lines) >= 4,
        "court_lines_detected": len(detected_lines),
        "lines": detected_lines[:40]
    }


def check_landing_inside_court(landing_point, court_corners):
    """Classify the landing point against the detected court polygon."""
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
        court_polygon = np.array([
            court_corners["top_left"],
            court_corners["top_right"],
            court_corners["bottom_right"],
            court_corners["bottom_left"]
        ], dtype=np.float32)

        point = (
            float(landing_point["x"]),
            float(landing_point["y"])
        )

        result = cv2.pointPolygonTest(court_polygon, point, False)

        if result >= 0:
            return {
                "result": "IN",
                "inside": True,
                "message": "Shuttle landed inside court"
            }

        return {
            "result": "OUT",
            "inside": False,
            "message": "Shuttle landed outside court"
        }

    except Exception as e:
        return {
            "result": "UNKNOWN",
            "inside": None,
            "reason": f"Landing decision error: {str(e)}"
        }


def choose_landing_point(trajectory):
    """
    Choose a stable final motion point.

    This is still a motion-based prototype, not a trained shuttle model.
    We deliberately avoid making an IN/OUT claim when the trajectory is too
    short or clearly unreliable.
    """
    if not trajectory:
        return None

    # Use the last few points and prefer the latest point. The later
    # shuttle model will replace this function.
    recent = trajectory[-8:]
    if len(recent) < 2:
        return None

    # Remove wild one-frame jumps from the end.
    selected = recent[0]
    for point in recent[1:]:
        jump = np.hypot(
            point["x"] - selected["x"],
            point["y"] - selected["y"]
        )
        if jump <= 450:
            selected = point

    return {
        "frame": int(selected["frame"]),
        "x": int(selected["x"]),
        "y": int(selected["y"])
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
                content={"status": "error", "message": "Could not open video"}
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
        shuttle_candidates = []
        trajectory = []
        previous_gray = None
        frame_number = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            if previous_gray is not None:
                difference = cv2.absdiff(previous_gray, gray)

                _, threshold = cv2.threshold(
                    difference, 30, 255, cv2.THRESH_BINARY
                )

                # Remove tiny noise and connect nearby motion.
                kernel = np.ones((3, 3), np.uint8)
                threshold = cv2.morphologyEx(
                    threshold, cv2.MORPH_OPEN, kernel
                )

                contours, _ = cv2.findContours(
                    threshold,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                frame_has_motion = False

                for contour in contours:
                    area = cv2.contourArea(contour)

                    if 5 < area < 500:
                        x, y, w, h = cv2.boundingRect(contour)

                        # Extremely thin/huge blobs are usually not a shuttle.
                        if w > 120 or h > 120:
                            continue

                        center_x = x + w // 2
                        center_y = y + h // 2

                        candidate = {
                            "frame": frame_number,
                            "x": center_x,
                            "y": center_y,
                            "area": round(float(area), 2)
                        }

                        shuttle_candidates.append(candidate)
                        trajectory.append({
                            "frame": frame_number,
                            "x": center_x,
                            "y": center_y
                        })
                        frame_has_motion = True

                if frame_has_motion:
                    motion_frames += 1

            previous_gray = gray
            frame_number += 1

        cap.release()

        motion_percentage = (
            motion_frames / total_frames * 100
            if total_frames > 0 else 0
        )

        # Require more evidence than the old >5 rule.
        shuttle_detected = len(shuttle_candidates) >= 8

        estimated_landing_point = (
            choose_landing_point(trajectory)
            if shuttle_detected else None
        )

        landing_decision = check_landing_inside_court(
            estimated_landing_point,
            court_corners
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
                "candidate_points": len(shuttle_candidates)
            },
            "trajectory": trajectory[-100:],
            "estimated_landing_point": estimated_landing_point,
            "decision": landing_decision
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
