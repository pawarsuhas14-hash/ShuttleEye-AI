import numpy as np

def line_intersection(line1, line2):
    """
    Calculate the intersection point between two detected lines.

    Each line is represented as a dictionary:
    {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2
    }

    Returns:
        (x, y) intersection point
        or None if lines are parallel.
    """

    x1 = line1["x1"]
    y1 = line1["y1"]
    x2 = line1["x2"]
    y2 = line1["y2"]

    x3 = line2["x1"]
    y3 = line2["y1"]
    x4 = line2["x2"]
    y4 = line2["y2"]

    denominator = (
        (x1 - x2) * (y3 - y4)
        - (y1 - y2) * (x3 - x4)
    )

    if denominator == 0:
        return None

    px = (
        ((x1 * y2 - y1 * x2) * (x3 - x4)
        - (x1 - x2) * (x3 * y4 - y3 * x4))
        / denominator
    )

    py = (
        ((x1 * y2 - y1 * x2) * (y3 - y4)
        - (y1 - y2) * (x3 * y4 - y3 * x4))
        / denominator
    )

    return int(px), int(py)


def find_line_intersections(lines):
    """
    Find all possible intersection points
    between detected court lines.

    Args:
lines: List of detected line dictionaries:
       {
           "x1": int,
           "y1": int,
           "x2": int,
           "y2": int
       }

    Returns:
        List of intersection points.
    """

    intersections = []

    if lines is None:
        return intersections

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):

            point = line_intersection(
                lines[i],
                lines[j]
            )

            if point is not None:
                intersections.append(point)

    return intersections


def filter_points_inside_frame(points, frame_shape):
    """
    Remove intersection points outside
    the video frame.

    Args:
        points: List of (x, y) points
        frame_shape: Video frame shape

    Returns:
        Valid points inside the frame.
    """

    height, width = frame_shape[:2]

    valid_points = []

    for x, y in points:

        if 0 <= x < width and 0 <= y < height:
            valid_points.append((x, y))

    return valid_points

def get_court_corners(frame):
    """
    Detect badminton court corners from a video frame.

    Returns:
        Dictionary containing detected court corner points
        or None if detection fails.
    """

    try:
        import cv2
        import numpy as np

        if frame is None:
            return None

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Slight blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(
            blurred,
            50,
            150,
            apertureSize=3
        )

        # Detect lines
        lines_p = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=80,
            maxLineGap=20
        )

        if lines_p is None:
            return None

        lines = []

        for line in lines_p:
            x1, y1, x2, y2 = line[0]

            lines.append({
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2)
            })

        # Find intersections
        intersections = find_line_intersections(lines)

        # Keep points inside frame
        valid_points = filter_points_inside_frame(
            intersections,
            frame.shape
        )

        if len(valid_points) < 4:
            return None

        # Convert to numpy array
        points = np.array(valid_points, dtype=np.float32)

        # Find bounding rectangle
        x, y, w, h = cv2.boundingRect(points)

        top_left = (int(x), int(y))
        top_right = (int(x + w), int(y))
        bottom_right = (int(x + w), int(y + h))
        bottom_left = (int(x), int(y + h))

        return {
            "top_left": top_left,
            "top_right": top_right,
            "bottom_right": bottom_right,
            "bottom_left": bottom_left,
            "all_points": valid_points
        }

    except Exception as e:
        print(f"Court corner detection error: {e}")
        return None
