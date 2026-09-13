import numpy as np
import cv2


def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denominator = ((x1 - x2) * (y3 - y4) -
                   (y1 - y2) * (x3 - x4))
    if abs(float(denominator)) < 1e-6:
        return None

    px = (((x1 * y2 - y1 * x2) * (x3 - x4) -
           (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator)
    py = (((x1 * y2 - y1 * x2) * (y3 - y4) -
           (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator)
    return float(px), float(py)


def _normalise_hough_lines(lines_p):
    """Safely convert OpenCV HoughLinesP output to (x1,y1,x2,y2) tuples."""
    result = []
    if lines_p is None:
        return result

    for raw in lines_p:
        try:
            values = np.asarray(raw, dtype=np.float32).reshape(-1)
            if values.size < 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in values[:4]]
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length >= 50:
                result.append((x1, y1, x2, y2))
        except (TypeError, ValueError):
            continue
    return result


def _angle(line):
    x1, y1, x2, y2 = line
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))


def _normalised_angle(angle):
    a = angle % 180.0
    return a


def _point_to_tuple(p):
    return (int(round(float(p[0]))), int(round(float(p[1]))))


def _order_corners(points):
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    # Robust ordering by centroid angle, then rotate to top-left first.
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]

    # Force clockwise order in image coordinates and start at top-left.
    area = cv2.contourArea(ordered.reshape(-1, 1, 2))
    if area < 0:
        ordered = ordered[::-1]

    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -start, axis=0)

    return {
        "top_left": _point_to_tuple(ordered[0]),
        "top_right": _point_to_tuple(ordered[1]),
        "bottom_right": _point_to_tuple(ordered[2]),
        "bottom_left": _point_to_tuple(ordered[3]),
    }


def get_court_corners(frame):
    """Estimate a reliable four-corner court quadrilateral from a frame.

    Returns a dict with top_left/top_right/bottom_right/bottom_left or None.
    """
    if frame is None or not hasattr(frame, "shape"):
        return None

    try:
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)

        # Slightly permissive settings because phone videos can have faint lines.
        lines_p = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=70,
            minLineLength=max(60, int(min(width, height) * 0.07)),
            maxLineGap=35,
        )
        lines = _normalise_hough_lines(lines_p)
        if len(lines) < 4:
            return None

        # Keep the strongest/longest lines. This avoids thousands of noisy
        # intersections and makes the geometry stable.
        lines.sort(key=lambda ln: np.hypot(ln[2] - ln[0], ln[3] - ln[1]), reverse=True)
        lines = lines[:80]

        # Build candidate intersections, but only between clearly different
        # directions (court sides should not be parallel to each other).
        intersections = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                a1 = _normalised_angle(_angle(lines[i]))
                a2 = _normalised_angle(_angle(lines[j]))
                delta = abs(a1 - a2)
                delta = min(delta, 180.0 - delta)
                if delta < 15.0:
                    continue

                p = line_intersection(lines[i], lines[j])
                if p is None:
                    continue
                x, y = p
                if -0.10 * width <= x <= 1.10 * width and -0.10 * height <= y <= 1.10 * height:
                    intersections.append((x, y))

        if len(intersections) < 4:
            return None

        # Cluster nearby intersections.
        clusters = []
        for p in intersections:
            p = np.asarray(p, dtype=np.float32)
            placed = False
            for c in clusters:
                if np.linalg.norm(p - c[0]) < 35:
                    c[0] = (c[0] * c[1] + p) / (c[1] + 1)
                    c[1] += 1
                    placed = True
                    break
            if not placed:
                clusters.append([p, 1])

        points = np.asarray([c[0] for c in clusters if c[1] >= 1], dtype=np.float32)
        if len(points) < 4:
            return None

        # Use the outer hull, then simplify to four points.
        hull = cv2.convexHull(points.reshape(-1, 1, 2))
        perimeter = cv2.arcLength(hull, True)
        approx = None
        for eps_ratio in (0.01, 0.02, 0.03, 0.04, 0.06, 0.08):
            candidate = cv2.approxPolyDP(hull, eps_ratio * perimeter, True)
            if len(candidate) == 4:
                approx = candidate.reshape(4, 2)
                break

        if approx is None:
            # Fallback: minimum-area rectangle is only used when the hull is
            # reasonably spread out, avoiding tiny/noisy rectangles.
            rect = cv2.minAreaRect(hull)
            rw, rh = rect[1]
            if min(rw, rh) < 0.12 * min(width, height):
                return None
            approx = cv2.boxPoints(rect)

        # Validate that the quadrilateral is not degenerate.
        quad = np.asarray(approx, dtype=np.float32).reshape(4, 2)
        quad_area = abs(cv2.contourArea(quad.reshape(-1, 1, 2)))
        if quad_area < 0.08 * width * height:
            return None

        return _order_corners(quad)

    except Exception as e:
        print(f"Court corner detection error: {type(e).__name__}: {e}")
        return None
