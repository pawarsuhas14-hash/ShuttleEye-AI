import cv2
import numpy as np


def detect_court_lines(frame):
    """
    Detect possible badminton court lines in a video frame.

    Returns:
        court_detected (bool)
        lines (list)
    """

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    # Detect straight lines
    detected_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=80,
        maxLineGap=15
    )

    lines = []

    if detected_lines is not None:
        for line in detected_lines:
            x1, y1, x2, y2 = line[0]

            lines.append({
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2)
            })

    court_detected = len(lines) >= 4

    return court_detected, lines
