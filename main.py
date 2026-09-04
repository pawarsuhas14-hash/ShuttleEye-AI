import os
import tempfile
import shutil

import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="ShuttleEye AI",
    version="0.6.0"
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
        "version": "0.6.0"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/analyze-video")
async def analyze_video(video: UploadFile = File(...)):

    suffix = os.path.splitext(video.filename)[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        shutil.copyfileobj(video.file, temp_file)
        temp_path = temp_file.name

    try:

        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            return {
                "status": "error",
                "message": "Could not open video"
            }

        # Video information
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


        # -----------------------------------
        # MOTION DETECTION
        # -----------------------------------

        ret, previous_frame = cap.read()

        if not ret:
            cap.release()

            return {
                "status": "error",
                "message": "Could not read video frames"
            }


        previous_gray = cv2.cvtColor(
            previous_frame,
            cv2.COLOR_BGR2GRAY
        )


        motion_frames = 0

        max_motion_pixels = 0

        motion_samples = []


        frame_number = 1


        while True:

            ret, current_frame = cap.read()

            if not ret:
                break


            current_gray = cv2.cvtColor(
                current_frame,
                cv2.COLOR_BGR2GRAY
            )


            # Calculate difference between frames
            frame_difference = cv2.absdiff(
                previous_gray,
                current_gray
            )


            # Remove small noise
            blurred = cv2.GaussianBlur(
                frame_difference,
                (5, 5),
                0
            )


            # Convert difference to black/white image
            _, threshold = cv2.threshold(
                blurred,
                25,
                255,
                cv2.THRESH_BINARY
            )


            # Count changed pixels
            motion_pixels = cv2.countNonZero(
                threshold
            )


            # Store motion information
            if motion_pixels > 500:

                motion_frames += 1


                if motion_pixels > max_motion_pixels:

                    max_motion_pixels = motion_pixels


                # Store limited samples
                if len(motion_samples) < 10:

                    motion_samples.append({
                        "frame": frame_number,
                        "motion_pixels": motion_pixels
                    })


            previous_gray = current_gray

            frame_number += 1


        cap.release()


        file_size = os.path.getsize(temp_path)


        return {

            "status": "success",

            "message": "Video analyzed successfully",

            "filename": video.filename,


            "video_info": {

                "fps": round(fps, 2),

                "total_frames": frame_count,

                "duration_seconds": round(
                    duration,
                    2
                ),

                "resolution": {
                    "width": width,
                    "height": height
                },

                "file_size_mb": round(
                    file_size / (1024 * 1024),
                    2
                )

            },


            "motion_analysis": {

                "frames_with_motion": motion_frames,

                "motion_percentage": round(
                    (motion_frames / frame_count) * 100,
                    2
                ) if frame_count > 0 else 0,

                "max_motion_pixels": max_motion_pixels,

                "motion_samples": motion_samples

            },


            "shuttleeye_status": {

                "motion_detection": "active",

                "shuttle_detection": "next phase",

                "shuttle_tracking": "planned",

                "landing_detection": "planned",

                "line_decision": "planned"

            }

        }


    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)
