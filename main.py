from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import os
import uuid
import shutil

app = FastAPI(
    title="ShuttleEye AI",
    description="AI-powered badminton shuttle detection and video analysis",
    version="0.6.0"
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
    """
    Detect badminton court lines and estimate outer court corners.
    """

    height, width = frame.shape[:2]

    # Resize for faster processing
    scale = 0.5
    small = cv2.resize(frame, (int(width * scale), int(height * scale)))

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edges = cv2.Canny(blurred, 50, 150)

    # Detect lines
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

    # Temporary court detection logic
    court_detected = len(detected_lines) >= 4

    return {
        "court_detected": court_detected,
        "court_lines_detected": len(detected_lines),
        "lines": detected_lines[:30]
    }


@app.post("/analyze-video")
async def analyze_video(video: UploadFile = File(...)):

    # Create unique filename
    file_id = str(uuid.uuid4())

    input_path = f"/tmp/{file_id}_{video.filename}"

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

    # Read first valid frame for court detection
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

    # Detect court
    court_analysis = detect_court(first_frame)

    # Reset video
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

                # Filter very small and very large movement
                if 5 < area < 500:

                    x, y, w, h = cv2.boundingRect(contour)

                    center_x = x + w // 2
                    center_y = y + h // 2

                    shuttle_candidates.append({
                        "frame": frame_number,
                        "x": center_x,
                        "y": center_y,
                        "area": float(area)
                    })

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

    estimated_landing_point = None

    if len(trajectory) > 0:

        last_point = trajectory[-1]

        estimated_landing_point = {
            "frame": last_point["frame"],
            "x": last_point["x"],
            "y": last_point["y"]
        }

    # Clean temporary file
    if os.path.exists(input_path):
        os.remove(input_path)

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

        "estimated_landing_point": estimated_landing_point
    }
