import os, tempfile, shutil, uuid
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ShuttleEye AI", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # prototype only; restrict this in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_BYTES = 150 * 1024 * 1024

@app.get("/")
def root():
    return {"name": "ShuttleEye AI", "status": "online", "version": "0.3.0"}

@app.get("/health")
def health():
    return {"status": "ok"}

def analyze_video(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Unable to read video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=120, varThreshold=28, detectShadows=False
    )

    points = []
    prev = None
    frame_index = 0
    max_frames = min(frame_count, 450) if frame_count else 450

    while frame_index < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1

        # Downscale for speed while preserving relative coordinates.
        scale = 1.0
        if width > 720:
            scale = 720.0 / width
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

        fg = subtractor.apply(frame)
        fg = cv2.medianBlur(fg, 5)
        _, mask = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if 2 <= area <= 220:
                x, y, w, h = cv2.boundingRect(c)
                if max(w, h) <= 35:
                    cx, cy = x + w/2, y + h/2
                    candidates.append((cx, cy, area))

        if candidates:
            if prev is None:
                # Prefer small, compact moving candidate.
                cand = min(candidates, key=lambda p: p[2])
            else:
                # Choose candidate closest to previous trajectory point.
                cand = min(
                    candidates,
                    key=lambda p: (p[0]-prev[0])**2 + (p[1]-prev[1])**2
                )
                dist = ((cand[0]-prev[0])**2 + (cand[1]-prev[1])**2) ** 0.5
                if dist > 110 * scale:
                    cand = None

            if cand is not None:
                cx, cy, _ = cand
                points.append((frame_index, float(cx/scale), float(cy/scale)))
                prev = (cx, cy)

    cap.release()

    if len(points) < 4:
        return {
            "decision": "UNKNOWN",
            "confidence": 0,
            "message": "Shuttle could not be tracked reliably in this video yet.",
            "trajectory_points": 0,
            "video": {"fps": fps, "frames": frame_count, "width": width, "height": height},
            "landing_point": None
        }

    # Use last tracked point as prototype landing estimate.
    last = points[-1]
    landing = {"x": round(last[1], 1), "y": round(last[2], 1)}

    # Prototype court boundary: central 80% of frame.
    margin_x = width * 0.10
    margin_y = height * 0.08
    inside = (
        margin_x <= last[1] <= width - margin_x and
        margin_y <= last[2] <= height - margin_y
    )

    # Confidence depends on trajectory length; capped because this is not trained AI yet.
    confidence = min(72, 35 + len(points) * 2)

    return {
        "decision": "IN" if inside else "OUT",
        "confidence": confidence,
        "message": "Prototype computer-vision result. Court calibration and custom shuttle model will improve accuracy.",
        "trajectory_points": len(points),
        "landing_point": landing,
        "video": {"fps": fps, "frames": frame_count, "width": width, "height": height}
    }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No video supplied")

    suffix = Path(file.filename).suffix or ".mp4"
    tmp_dir = Path(tempfile.mkdtemp(prefix="shuttleeye_"))
    tmp_file = tmp_dir / f"{uuid.uuid4()}{suffix}"

    try:
        size = 0
        with open(tmp_file, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, "Video is too large. Maximum is 150 MB.")
                out.write(chunk)

        return analyze_video(str(tmp_file))

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Analysis failed: {str(exc)}")
    finally:
        await file.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)
