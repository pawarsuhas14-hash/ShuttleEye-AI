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
            detected_lines.append(
                {
                    "x1": int(x1 / scale),
                    "y1": int(y1 / scale),
                    "x2": int(x2 / scale),
                    "y2": int(y2 / scale),
                }
            )

    # Temporary court detection logic
    court_detected = len(detected_lines) >= 4

    return {
        "court_detected": court_detected,
        "court_lines_detected": len(detected_lines),
        "lines": detected_lines[:30]
    }
