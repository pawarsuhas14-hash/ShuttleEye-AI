from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uuid

app = FastAPI(
    title="ShuttleEye AI",
    version="0.4.0"
)

# Allow ShuttleEye PWA to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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
        "service": "ShuttleEye AI Backend"
    }


@app.post("/analyze-video")
async def analyze_video(video: UploadFile = File(...)):

    # Create temporary folder
    os.makedirs("uploads", exist_ok=True)

    # Generate unique filename
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(video.filename)[1]

    file_path = f"uploads/{file_id}{file_extension}"

    # Save uploaded video
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # Placeholder for AI processing
    result = {
        "status": "success",
        "message": "Video received successfully",
        "filename": video.filename,
        "shuttle_detected": False,
        "landing_detected": False,
        "decision": "PROCESSING"
    }

    # Delete temporary file
    if os.path.exists(file_path):
        os.remove(file_path)

    return result
