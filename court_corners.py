import numpy as np
import cv2


def line_intersection(line1, line2):
    """Calculate the intersection point between two lines."""

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

    if abs(denominator) < 0.001:
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

    return int(round(px)), int(round(py))


def get_court_corners(frame):
    """Estimate four outer court corners from a video frame."""

    if frame is None:
        return None

    try:
        height, width = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(
            blurred,
            50,
            150
        )

        lines_p = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=30
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

        intersections = []

        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                point = line_intersection(
                    lines[i],
                    lines[j]
                )

                if point is None:
                    continue

                x, y = point

                if 0 <= x < width and 0 <= y < height:
                    intersections.append(point)

        if len(intersections) < 4:
            return None

        points = np.array(
            intersections,
            dtype=np.float32
        )

        # Remove duplicate/near-duplicate intersection points.
        unique_points = []

        for point in points:
            if not any(
                np.linalg.norm(point - existing) < 20
                for existing in unique_points
            ):
                unique_points.append(point)

        if len(unique_points) < 4:
            return None

        points = np.array(
            unique_points,
            dtype=np.float32
        )

        # Use the outer convex hull as an estimate of the court boundary.
        hull = cv2.convexHull(points)

        if len(hull) < 4:
            return None

        epsilon = 0.02 * cv2.arcLength(
            hull,
            True
        )

        approx = cv2.approxPolyDP(
            hull,
            epsilon,
            True
        )

        approx_points = approx.reshape(-1, 2)

        # If approximation does not give exactly four corners,
        # use the minimum-area rectangle around the hull.
        if len(approx_points) != 4:
            rect = cv2.minAreaRect(hull)
            approx_points = cv2.boxPoints(rect)

        approx_points = np.asarray(
            approx_points,
            dtype=np.float32
        )

        if len(approx_points) != 4:
            return None

        # Order corners: top-left, top-right, bottom-right, bottom-left.
        ordered = _order_corners(approx_points)

        return {
            "top_left": ordered[0],
            "top_right": ordered[1],
            "bottom_right": ordered[2],
            "bottom_left": ordered[3]
        }

    except Exception as e:
        print(f"Court corner detection error: {e}")
        return None


def _order_corners(points):
    """Return four points in clockwise court order."""

    points = np.asarray(points, dtype=np.float32)

    # Sum: smallest is top-left, largest is bottom-right.
    sums = points.sum(axis=1)

    # Difference y - x: smallest is top-right, largest is bottom-left.
    diffs = points[:, 1] - points[:, 0]

    top_left = points[np.argmin(sums)]
    bottom_right = points[np.argmax(sums)]
    top_right = points[np.argmin(diffs)]
    bottom_left = points[np.argmax(diffs)]

    return [
        _point_to_tuple(top_left),
        _point_to_tuple(top_right),
        _point_to_tuple(bottom_right),
        _point_to_tuple(bottom_left)
    ]


def _point_to_tuple(point):
    return (
        int(round(float(point[0]))),
        int(round(float(point[1])))
    )
