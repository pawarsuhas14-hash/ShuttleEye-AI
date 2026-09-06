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
