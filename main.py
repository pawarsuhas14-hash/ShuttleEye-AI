import os
import tempfile
import shutil

import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="ShuttleEye AI",
    version="0.5.0"
)

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
        "version": "0.5.0"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/analyze-video")
async def analyze_video(video: UploadFile = File(...)):

    # Create temporary file
    suffix = os.path.splitext(video.filename)[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        shutil.copyfileobj(video.file, temp_file)
        temp_path = temp_file.name

    try:

        # Open video
        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            return {
                "status": "error",
                "message": "Could not open video"
            }

        # Extract video properties
        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_count = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        duration = frame_count / fps if fps > 0 else 0

        cap.release()

        file_size = os.path.getsize(temp_path)

        return {
            "status": "success",
            "message": "Video analyzed successfully",
            "filename": video.filename,
            "video_info": {
                "fps": round(fps, 2),
                "total_frames": frame_count,
                "duration_seconds": round(duration, 2),
                "resolution": {
                    "width": width,
                    "height": height
                },
                "file_size_mb": round(
                    file_size / (1024 * 1024),
                    2
                )
            },
            "analysis": {
                "shuttle_detection": "coming soon",
                "court_detection": "coming soon",
                "landing_detection": "coming soon",
                "line_decision": "coming soon"
            }
        }

    finally:

        # Delete temporary video
        if os.path.exists(temp_path):
            os.remove(temp_path)
