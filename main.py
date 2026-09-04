import os
import tempfile
import uuid
import cv2
import numpy as np

from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="ShuttleEye AI",
    version="0.4.0",
    description="AI-powered badminton shuttle analysis system"
)


# Allow frontend/mobile applications to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "name": "ShuttleEye AI",
        "status": "online",
        "version": "0.4.0"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "ShuttleEye AI"
    }


def analyze_shuttle(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception("Unable to open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = 0

    if fps > 0:
        duration = total_frames / fps

    # Background subtractor
    background_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=100,
        varThreshold=25,
        detectShadows=False
    )

    shuttle_candidates = []

    frame_number = 0

    previous_centers = []

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # Resize for faster processing
        processing_frame = cv2.resize(
            frame,
            (640, 360)
        )

        gray = cv2.cvtColor(
            processing_frame,
            cv2.COLOR_BGR2GRAY
        )

        # Reduce noise
        blurred = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        # Detect moving objects
        foreground_mask = background_subtractor.apply(
            blurred
        )

        # Threshold mask
        _, threshold = cv2.threshold(
            foreground_mask,
            200,
            255,
            cv2.THRESH_BINARY
        )

        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)

        threshold = cv2.morphologyEx(
            threshold,
            cv2.MORPH_OPEN,
            kernel
        )

        contours, _ = cv2.findContours(
            threshold,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        frame_candidates = []

        for contour in contours:

            area = cv2.contourArea(contour)

            # Shuttlecock usually appears as a small moving object
            if area < 2 or area > 500:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            aspect_ratio = w / float(h) if h != 0 else 0

            # Ignore very large or unusual objects
            if w > 80 or h > 80:
                continue

            center_x = x + w // 2
            center_y = y + h // 2

            candidate = {
                "frame": frame_number,
                "x": int(center_x),
                "y": int(center_y),
                "area": float(area),
                "width": int(w),
                "height": int(h)
            }

            frame_candidates.append(candidate)

        # Store possible shuttle candidates
        if len(frame_candidates) > 0:

            # Select smallest moving object
            best_candidate = min(
                frame_candidates,
                key=lambda c: c["area"]
            )

            shuttle_candidates.append(
                best_candidate
            )

    cap.release()

    # Analyze trajectory
    shuttle_detected = len(shuttle_candidates) > 3

    landing_point = None

    if shuttle_detected:

        # Current prototype:
        # Assume the last detected point is closest to landing
        last_point = shuttle_candidates[-1]

        landing_point = {
            "x": last_point["x"],
            "y": last_point["y"],
            "frame": last_point["frame"]
        }

    return {
        "video_info": {
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "duration_seconds": round(duration, 2),
            "resolution": {
                "width": width,
                "height": height
            }
        },

        "shuttle_detection": {
            "shuttle_detected": shuttle_detected,
            "candidate_points": len(shuttle_candidates),
            "trajectory_points": shuttle_candidates[:100],
            "estimated_landing_point": landing_point
        }
    }


@app.post("/analyze-video")
async def analyze_video(video: UploadFile = File(...)):

    allowed_extensions = [
        ".mp4",
        ".mov",
        ".avi",
        ".mkv"
    ]

    file_extension = Path(video.filename).suffix.lower()

    if file_extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail="Unsupported video format"
        )

    # Create temporary filename
    temp_filename = (
        f"{uuid.uuid4()}{file_extension}"
    )

    temp_path = os.path.join(
        tempfile.gettempdir(),
        temp_filename
    )

    try:

        # Save uploaded video
        with open(temp_path, "wb") as buffer:

            content = await video.read()

            buffer.write(content)

        # Analyze video
        result = analyze_shuttle(temp_path)

        file_size_mb = (
            os.path.getsize(temp_path)
            / (1024 * 1024)
        )

        return {
            "status": "success",

            "message": "Video analyzed successfully",

            "filename": video.filename,

            "file_size_mb": round(
                file_size_mb,
                2
            ),

            "analysis": result
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        # Delete temporary file
        if os.path.exists(temp_path):

            os.remove(temp_path)
