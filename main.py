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
    version="0.7.0"
)


@app.get("/")
def home():
    return {
        "message": "Welcome to ShuttleEye AI",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


def detect_court(frame):
    """Detect badminton court lines in a video frame."""

    height, width = frame.shape[:2]

    # Resize for faster processing
    scale = 0.5
    small = cv2.resize(
        frame,
        (int(width * scale), int(height * scale))
    )

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=80,
        maxLineGap=20
    )

    detected_lines = []

    if lines is not None:
        for line in lines:
            values = np.array(line).flatten()

            if len(values) < 4:
                continue

            x1, y1, x2, y2 = values[:4]

            detected_lines.append({
                "x1": int(x1 / scale),
                "y1": int(y1 / scale),
                "x2": int(x2 / scale),
                "y2": int(y2 / scale)
            })

    court_detected = len(detected_lines) >= 4

    return {
        "court_detected": court_detected,
        "court_lines_detected": len(detected_lines),
        "lines": detected_lines[:30]
    }


def check_landing_inside_court(landing_point, court_corners):
    """Check whether the estimated landing point is inside the court."""

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

        # >= 0 means inside or exactly on the boundary.
        polygon_result = cv2.pointPolygonTest(
            court_polygon,
            point,
            False
        )

        if polygon_result >= 0:
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


@app.post("/analyze-video")
async def analyze_video(video: UploadFile = File(...)):
    input_path = None

    try:
        # Create unique temporary filename.
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

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration = 0
        if fps > 0:
            duration = total_frames / fps

        # Read first valid frame for court detection.
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

        # Detect court lines.
        court_analysis = detect_court(first_frame)

        # Detect court corners.
        court_corners = get_court_corners(first_frame)
        court_analysis["corners"] = court_corners

        # Reset video to first frame.
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
                    difference,
                    30,
                    255,
                    cv2.THRESH_BINARY
                )

                contours, _ = cv2.findContours(
                    threshold,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                frame_has_motion = False

                for contour in contours:
                    area = cv2.contourArea(contour)

                    # Initial motion candidate filter.
                    if 5 < area < 500:
                        x, y, w, h = cv2.boundingRect(contour)

                        center_x = x + w // 2
                        center_y = y + h // 2

                        candidate = {
                            "frame": frame_number,
                            "x": center_x,
                            "y": center_y,
                            "area": float(area)
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

        motion_percentage = 0

        if total_frames > 0:
            motion_percentage = (
                motion_frames / total_frames
            ) * 100

        shuttle_detected = len(shuttle_candidates) > 5

        # Initial landing estimate.
        # NOTE: This is currently the last detected motion point.
        # Later we will replace this with a true shuttle-flight/impact model.
        estimated_landing_point = None

        if len(trajectory) > 0:
            last_point = trajectory[-1]

            estimated_landing_point = {
                "frame": last_point["frame"],
                "x": last_point["x"],
                "y": last_point["y"]
            }

        # Final IN / OUT decision.
        landing_decision = check_landing_inside_court(
            estimated_landing_point,
            court_corners
        )

        return {
            "status": "success",
            "message": "Video analyzed successfully",
            "video_info": {
                "fps": fps,
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
