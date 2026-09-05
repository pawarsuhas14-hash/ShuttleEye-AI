import os
import tempfile
import uuid
from pathlib import Path

import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------

app = FastAPI(
    title="ShuttleEye AI",
    version="0.5.0",
    description="AI-powered badminton shuttle detection and video analysis"
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# OUTPUT DIRECTORY
# --------------------------------------------------

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

app.mount(
    "/outputs",
    StaticFiles(directory="outputs"),
    name="outputs"
)


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "name": "ShuttleEye AI",
        "version": "0.5.0",
        "status": "running",
        "message": "AI-powered badminton shuttle analysis API"
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# --------------------------------------------------
# FIND MOVING OBJECT CANDIDATES
# --------------------------------------------------

def detect_motion_candidates(frame1, frame2):

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    difference = cv2.absdiff(gray1, gray2)

    _, threshold = cv2.threshold(
        difference,
        25,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones((3, 3), np.uint8)

    threshold = cv2.dilate(
        threshold,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore very tiny noise and large moving objects
        if area < 2 or area > 500:

            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Shuttle candidates should usually be relatively compact
        if w > 80 or h > 80:

            continue

        center_x = x + (w // 2)
        center_y = y + (h // 2)

        candidates.append({
            "x": int(center_x),
            "y": int(center_y),
            "area": float(area),
            "width": int(w),
            "height": int(h)
        })

    return candidates


# --------------------------------------------------
# SELECT BEST SHUTTLE CANDIDATE
# --------------------------------------------------

def select_best_candidate(candidates, previous_point=None):

    if not candidates:

        return None

    # First detected point
    if previous_point is None:

        # Prefer small compact object
        candidates.sort(
            key=lambda p: abs(p["width"] - p["height"]) + p["area"]
        )

        return candidates[0]

    best_candidate = None
    best_distance = float("inf")

    for candidate in candidates:

        distance = np.sqrt(
            (candidate["x"] - previous_point["x"]) ** 2 +
            (candidate["y"] - previous_point["y"]) ** 2
        )

        if distance < best_distance:

            best_distance = distance
            best_candidate = candidate

    # Reject candidate if it suddenly jumps too far
    if best_distance > 250:

        return None

    return best_candidate


# --------------------------------------------------
# DRAW TRAJECTORY
# --------------------------------------------------

def draw_trajectory(frame, trajectory):

    if len(trajectory) < 2:

        return

    points = []

    for point in trajectory:

        points.append(
            (int(point["x"]), int(point["y"]))
        )

    for i in range(1, len(points)):

        cv2.line(
            frame,
            points[i - 1],
            points[i],
            (0, 255, 255),
            2
        )


# --------------------------------------------------
# ANALYZE VIDEO
# --------------------------------------------------

@app.post("/analyze-video")
async def analyze_video(video: UploadFile = File(...)):

    if not video.filename:

        raise HTTPException(
            status_code=400,
            detail="No video file provided"
        )

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

    temp_input_path = None

    try:

        # ------------------------------------------
        # SAVE UPLOADED VIDEO
        # ------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_extension
        ) as temp_file:

            temp_input_path = temp_file.name

            contents = await video.read()

            temp_file.write(contents)

        file_size_mb = round(
            len(contents) / (1024 * 1024),
            2
        )


        # ------------------------------------------
        # OPEN VIDEO
        # ------------------------------------------

        cap = cv2.VideoCapture(temp_input_path)

        if not cap.isOpened():

            raise HTTPException(
                status_code=400,
                detail="Unable to open video"
            )

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:

            fps = 30

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        duration_seconds = round(
            total_frames / fps,
            2
        )


        # ------------------------------------------
        # CREATE OUTPUT VIDEO
        # ------------------------------------------

        output_filename = (
            f"shuttleeye_{uuid.uuid4().hex}.mp4"
        )

        output_path = OUTPUT_DIR / output_filename

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )


        # ------------------------------------------
        # READ FIRST FRAME
        # ------------------------------------------

        success, previous_frame = cap.read()

        if not success:

            raise HTTPException(
                status_code=400,
                detail="Could not read video"
            )


        trajectory = []
        all_candidate_points = []

        previous_shuttle_point = None

        frame_number = 1
        frames_with_motion = 0


        # Write first frame

        cv2.putText(
            previous_frame,
            "ShuttleEye AI",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        writer.write(previous_frame)


        # ------------------------------------------
        # PROCESS VIDEO FRAMES
        # ------------------------------------------

        while True:

            success, current_frame = cap.read()

            if not success:

                break

            frame_number += 1

            candidates = detect_motion_candidates(
                previous_frame,
                current_frame
            )

            if candidates:

                frames_with_motion += 1


            # Draw all candidate points

            for candidate in candidates:

                x = candidate["x"]
                y = candidate["y"]

                cv2.circle(
                    current_frame,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )


            # Select probable shuttle

            best_candidate = select_best_candidate(
                candidates,
                previous_shuttle_point
            )


            if best_candidate:

                shuttle_point = {
                    "frame": frame_number,
                    "x": best_candidate["x"],
                    "y": best_candidate["y"]
                }

                trajectory.append(shuttle_point)

                all_candidate_points.append(
                    shuttle_point
                )

                previous_shuttle_point = best_candidate


                # Red circle around probable shuttle

                cv2.circle(
                    current_frame,
                    (
                        best_candidate["x"],
                        best_candidate["y"]
                    ),
                    10,
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    current_frame,
                    "SHUTTLE",
                    (
                        best_candidate["x"] + 10,
                        best_candidate["y"] - 10
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )


            # Draw trajectory

            draw_trajectory(
                current_frame,
                trajectory
            )


            # Header information

            cv2.putText(
                current_frame,
                "ShuttleEye AI",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                current_frame,
                f"Frame: {frame_number}",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                current_frame,
                f"Trajectory points: {len(trajectory)}",
                (30, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


            writer.write(current_frame)

            previous_frame = current_frame.copy()


        # ------------------------------------------
        # MARK ESTIMATED LANDING POINT
        # ------------------------------------------

        estimated_landing_point = None

        if trajectory:

            estimated_landing_point = trajectory[-1]


        # ------------------------------------------
        # CLEANUP
        # ------------------------------------------

        cap.release()
        writer.release()


        # ------------------------------------------
        # RESPONSE
        # ------------------------------------------

        return {
            "status": "success",
            "message": "Video analyzed successfully",

            "filename": video.filename,

            "video_info": {
                "fps": round(fps, 2),
                "total_frames": total_frames,
                "duration_seconds": duration_seconds,

                "resolution": {
                    "width": width,
                    "height": height
                },

                "file_size_mb": file_size_mb
            },

            "motion_analysis": {
                "frames_with_motion": frames_with_motion,

                "motion_percentage": round(
                    (frames_with_motion / total_frames) * 100,
                    2
                ) if total_frames > 0 else 0
            },

            "shuttle_detection": {

                "shuttle_detected": (
                    len(trajectory) > 0
                ),

                "candidate_points": len(
                    all_candidate_points
                ),

                "trajectory_points": trajectory,

                "estimated_landing_point":
                    estimated_landing_point
            },

            "annotated_video": {
                "filename": output_filename,
                "url": f"/outputs/{output_filename}"
            }
        }


    except HTTPException:

        raise


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


    finally:

        if temp_input_path and os.path.exists(
            temp_input_path
        ):

            os.remove(temp_input_path)
