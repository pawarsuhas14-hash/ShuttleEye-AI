import numpy as np
import cv2
from itertools import combinations


def _line_to_normal(line):
    x1, y1, x2, y2 = (
        float(line["x1"]),
        float(line["y1"]),
        float(line["x2"]),
        float(line["y2"]),
    )

    dx = x2 - x1
    dy = y2 - y1
    length = np.hypot(dx, dy)

    if length < 1.0:
        return None

    a = dy / length
    b = -dx / length
    c = -(a * x1 + b * y1)

    return a, b, c


def line_intersection(line1, line2):
    n1 = _line_to_normal(line1)
    n2 = _line_to_normal(line2)

    if n1 is None or n2 is None:
        return None

    a1, b1, c1 = n1
    a2, b2, c2 = n2

    denominator = a1 * b2 - a2 * b1

    if abs(denominator) < 0.035:
        return None

    x = (b1 * c2 - b2 * c1) / denominator
    y = (c1 * a2 - c2 * a1) / denominator

    return float(x), float(y)


def _segment_length(line):
    return float(np.hypot(
        line["x2"] - line["x1"],
        line["y2"] - line["y1"]
    ))


def _angle(line):
    return np.degrees(np.arctan2(
        line["y2"] - line["y1"],
        line["x2"] - line["x1"]
    )) % 180.0


def _angle_difference(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _deduplicate_lines(lines, width, height):
    if not lines:
        return []

    diag = np.hypot(width, height)
    angle_tol = 5.0
    distance_tol = max(18.0, diag * 0.012)

    prepared = []

    for line in lines:
        length = _segment_length(line)

        if length < max(50.0, min(width, height) * 0.06):
            continue

        normal = _line_to_normal(line)

        if normal is None:
            continue

        a, b, c = normal

        prepared.append({
            **line,
            "_length": length,
            "_angle": _angle(line),
            "_normal": (a, b, c)
        })

    prepared.sort(key=lambda x: x["_length"], reverse=True)

    kept = []

    for line in prepared:
        a, b, c = line["_normal"]
        duplicate = False

        for other in kept:
            a2, b2, c2 = other["_normal"]

            if _angle_difference(
                line["_angle"],
                other["_angle"]
            ) > angle_tol:
                continue

            d = abs(c - c2)
            d_opposite = abs(c + c2)

            if min(d, d_opposite) < distance_tol:
                duplicate = True
                break

        if not duplicate:
            kept.append(line)

    return kept[:40]


def _cluster_orientation(lines):
    if len(lines) < 4:
        return [], []

    best = None

    for seed in lines[:15]:
        seed_angle = seed["_angle"]

        group_a = [
            line for line in lines
            if _angle_difference(
                line["_angle"],
                seed_angle
            ) <= 12.0
        ]

        group_b = [
            line for line in lines
            if _angle_difference(
                line["_angle"],
                (seed_angle + 90.0) % 180.0
            ) <= 18.0
        ]

        if len(group_a) < 2 or len(group_b) < 2:
            continue

        support = (
            sum(line["_length"] for line in group_a)
            + sum(line["_length"] for line in group_b)
        )

        if best is None or support > best[0]:
            best = (support, group_a, group_b)

    if best is None:
        return [], []

    return best[1], best[2]


def _pair_distance(line1, line2):
    n1 = line1["_normal"]
    n2 = line2["_normal"]

    a1, b1, c1 = n1
    a2, b2, c2 = n2

    if a1 * a2 + b1 * b2 < 0:
        c2 = -c2

    return abs(c1 - c2)


def _intersection_inside_frame(point, width, height, margin=0.02):
    if point is None:
        return False

    x, y = point
    mx = width * margin
    my = height * margin

    return (
        -mx <= x <= width + mx
        and -my <= y <= height + my
    )


def _quad_area(points):
    pts = np.asarray(points, dtype=np.float32)
    return abs(float(
        cv2.contourArea(
            pts.reshape(-1, 1, 2)
        )
    ))


def _order_corners(points):
    pts = np.asarray(points, dtype=np.float32)
    center = pts.mean(axis=0)

    angles = np.arctan2(
        pts[:, 1] - center[1],
        pts[:, 0] - center[0]
    )

    ordered = pts[np.argsort(angles)]

    start = int(np.argmin(
        ordered[:, 0] + ordered[:, 1]
    ))

    ordered = np.roll(
        ordered,
        -start,
        axis=0
    )

    area = cv2.contourArea(
        ordered.reshape(-1, 1, 2)
    )

    if area < 0:
        ordered = ordered[[0, 3, 2, 1]]

    return [
        _point_to_tuple(ordered[0]),
        _point_to_tuple(ordered[1]),
        _point_to_tuple(ordered[2]),
        _point_to_tuple(ordered[3]),
    ]


def _score_quad(quad, group_a, group_b, width, height):
    pts = np.asarray(quad, dtype=np.float32)

    if len(pts) != 4:
        return -1e9

    area = _quad_area(pts)
    frame_area = float(width * height)

    if area < frame_area * 0.015:
        return -1e9

    if area > frame_area * 0.98:
        return -1e9

    if not cv2.isContourConvex(
        pts.reshape(-1, 1, 2)
    ):
        return -1e9

    side_lengths = [
        np.linalg.norm(
            pts[(i + 1) % 4] - pts[i]
        )
        for i in range(4)
    ]

    if min(side_lengths) < min(width, height) * 0.05:
        return -1e9

    ratio1 = side_lengths[0] / max(
        side_lengths[2], 1.0
    )
    ratio2 = side_lengths[1] / max(
        side_lengths[3], 1.0
    )

    if not (
        0.25 < ratio1 < 4.0
        and 0.25 < ratio2 < 4.0
    ):
        return -1e9

    score = min(
        area / frame_area,
        0.80
    ) * 100.0

    for point in pts:
        if not _intersection_inside_frame(
            point,
            width,
            height
        ):
            return -1e9

    side_angles = []

    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]

        side_angles.append(
            np.degrees(np.arctan2(
                p2[1] - p1[1],
                p2[0] - p1[0]
            )) % 180
        )

    parallel_error = (
        _angle_difference(
            side_angles[0],
            side_angles[2]
        )
        +
        _angle_difference(
            side_angles[1],
            side_angles[3]
        )
    )

    score -= parallel_error * 4.0

    center = pts.mean(axis=0)

    if (
        center[1] < height * 0.10
        or center[1] > height * 0.95
    ):
        score -= 10

    return score


def _fallback_contour_corners(frame):
    height, width = frame.shape[:2]

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    mask = cv2.inRange(
        hsv,
        np.array(
            [0, 0, 145],
            dtype=np.uint8
        ),
        np.array(
            [180, 115, 255],
            dtype=np.uint8
        )
    )

    edges = cv2.Canny(
        mask,
        50,
        150
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    edges = cv2.dilate(
        edges,
        kernel,
        iterations=1
    )

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best = None
    frame_area = width * height

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < frame_area * 0.03:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        approx = cv2.approxPolyDP(
            contour,
            0.03 * perimeter,
            True
        )

        if (
            len(approx) != 4
            or not cv2.isContourConvex(approx)
        ):
            continue

        pts = approx.reshape(
            4, 2
        ).astype(np.float32)

        score = area

        if area > frame_area * 0.90:
            score *= 0.1

        if best is None or score > best[0]:
            best = (
                score,
                pts
            )

    if best is None:
        return None

    return _order_corners(
        best[1]
    )


def get_court_corners(frame):
    """
    Multi-stage court geometry detector.

    Returns:
        {
            "top_left": (x, y),
            "top_right": (x, y),
            "bottom_right": (x, y),
            "bottom_left": (x, y)
        }
        or None
    """
    if frame is None:
        return None

    try:
        height, width = frame.shape[:2]

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        edge_sets = [
            cv2.Canny(
                gray,
                35,
                110
            ),
            cv2.Canny(
                gray,
                55,
                150
            ),
            cv2.Canny(
                gray,
                80,
                190
            )
        ]

        all_lines = []

        for edges in edge_sets:
            lines_p = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=45,
                minLineLength=max(
                    45,
                    int(min(width, height) * 0.07)
                ),
                maxLineGap=35
            )

            if lines_p is None:
                continue

            for raw in lines_p:
                x1, y1, x2, y2 = np.asarray(raw).reshape(-1)[:4]

                all_lines.append({
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                })

        lines = _deduplicate_lines(
            all_lines,
            width,
            height
        )

        if len(lines) >= 4:
            group_a, group_b = _cluster_orientation(
                lines
            )

            group_a = sorted(
                group_a,
                key=lambda x: x["_length"],
                reverse=True
            )[:14]

            group_b = sorted(
                group_b,
                key=lambda x: x["_length"],
                reverse=True
            )[:14]

            best_score = -1e9
            best_quad = None

            for a1, a2 in combinations(
                group_a,
                2
            ):
                sep_a = _pair_distance(
                    a1,
                    a2
                )

                if sep_a < min(
                    width,
                    height
                ) * 0.08:
                    continue

                for b1, b2 in combinations(
                    group_b,
                    2
                ):
                    sep_b = _pair_distance(
                        b1,
                        b2
                    )

                    if sep_b < min(
                        width,
                        height
                    ) * 0.08:
                        continue

                    p11 = line_intersection(
                        a1,
                        b1
                    )
                    p12 = line_intersection(
                        a1,
                        b2
                    )
                    p22 = line_intersection(
                        a2,
                        b2
                    )
                    p21 = line_intersection(
                        a2,
                        b1
                    )

                    quad = [
                        p11,
                        p12,
                        p22,
                        p21
                    ]

                    if any(
                        not _intersection_inside_frame(
                            p,
                            width,
                            height,
                            0.05
                        )
                        for p in quad
                    ):
                        continue

                    score = _score_quad(
                        quad,
                        group_a,
                        group_b,
                        width,
                        height
                    )

                    score += (
                        min(
                            a1["_length"],
                            a2["_length"]
                        )
                        +
                        min(
                            b1["_length"],
                            b2["_length"]
                        )
                    ) * 0.04

                    if score > best_score:
                        best_score = score
                        best_quad = quad

            if (
                best_quad is not None
                and best_score > 8
            ):
                return _order_corners(
                    best_quad
                )

        return _fallback_contour_corners(
            frame
        )

    except Exception as e:
        print(
            f"Court corner detection error: {e}"
        )
        return None


def _point_to_tuple(point):
    return (
        int(round(float(point[0]))),
        int(round(float(point[1])))
    )
