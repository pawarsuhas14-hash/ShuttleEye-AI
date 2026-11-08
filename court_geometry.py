"""Conservative image-space badminton court geometry inference.

The partial-view path fits two line families and intersects two supported
boundary lines from each family. Intersections may lie outside the frame, so
visible corner pixels are not required. This does not invent unseen boundary
lines: each of the four inferred sides must have measured in-frame line support.
"""
import itertools
import math

import cv2
import numpy as np


def _segment(line):
    return np.array([[line["x1"], line["y1"]], [line["x2"], line["y2"]]], dtype=np.float64)


def _unit_line(line):
    p = _segment(line)
    v = p[1] - p[0]
    length = float(np.linalg.norm(v))
    if length < 1.0:
        return None
    v /= length
    n = np.array([-v[1], v[0]])
    return n, float(np.dot(n, p[0])), v, length


def _angle_distance(a, b):
    d = abs(a - b) % math.pi
    return min(d, math.pi - d)


def _line_intersection(first, second):
    a, c, _, _ = first
    b, d, _, _ = second
    matrix = np.stack((a, b))
    if abs(float(np.linalg.det(matrix))) < 0.12:
        return None
    return np.linalg.solve(matrix, np.array([c, d], dtype=np.float64))


def _merge_collinear_segments(lines, width, height):
    """Deduplicate Hough fragments on the same physical marking."""
    diag = math.hypot(width, height)
    items = []
    for line in lines or []:
        try:
            g = _unit_line(line)
            if g is None:
                continue
            n, c, v, length = g
            angle = math.atan2(v[1], v[0]) % math.pi
            items.append((line, n, c, angle, length))
        except (KeyError, TypeError, ValueError):
            continue
    items.sort(key=lambda q: q[4], reverse=True)
    groups = []
    for item in items:
        _, n, c, angle, _ = item
        match = next((group for group in groups
                      if _angle_distance(angle, group[0][3]) < math.radians(3.0)
                      and min(abs(c - group[0][2]), abs(c + group[0][2])) < max(8.0, diag * 0.006)), None)
        if match is None:
            groups.append([item])
        else:
            match.append(item)
    merged = []
    for group in groups:
        representative = dict(group[0][0])
        representative["_segments"] = [q[0] for q in group]
        merged.append(representative)
    return merged[:28]


def _clip_line_to_frame(point, direction, width, height):
    hits = []
    for x in (0.0, float(width - 1)):
        if abs(direction[0]) > 1e-9:
            t = (x - point[0]) / direction[0]
            y = point[1] + t * direction[1]
            if 0 <= y <= height - 1:
                hits.append(np.array([x, y]))
    for y in (0.0, float(height - 1)):
        if abs(direction[1]) > 1e-9:
            t = (y - point[1]) / direction[1]
            x = point[0] + t * direction[0]
            if 0 <= x <= width - 1:
                hits.append(np.array([x, y]))
    unique = []
    for p in hits:
        if all(np.linalg.norm(p - q) > 2 for q in unique):
            unique.append(p)
    if len(unique) < 2:
        return None
    return max(itertools.combinations(unique, 2), key=lambda pair: np.linalg.norm(pair[1] - pair[0]))


def _visible_support(line, edge, width, height):
    """Fraction of the inferred edge inside the image covered by Hough segments."""
    a, b = edge
    direction = b - a
    full_length = float(np.linalg.norm(direction))
    if full_length < 1:
        return 0.0
    direction /= full_length
    lo, hi = 0.0, full_length
    # Clip parametric segment to image rectangle with Liang-Barsky.
    for p, q in ((-direction[0], a[0]), (direction[0], width - 1 - a[0]),
                 (-direction[1], a[1]), (direction[1], height - 1 - a[1])):
        if abs(p) < 1e-9:
            if q < 0:
                return 0.0
            continue
        t = q / p
        if p < 0:
            lo = max(lo, t)
        else:
            hi = min(hi, t)
    lo, hi = max(0.0, lo), min(full_length, hi)
    visible_length = hi - lo
    if visible_length < 12:
        return 0.0

    intervals = []
    edge_normal = np.array([-direction[1], direction[0]])
    segments = line.get("_segments", [line]) if isinstance(line, dict) else line
    for segment in segments:
        p0, p1 = _segment(segment)
        if max(abs(np.dot(edge_normal, p0 - a)), abs(np.dot(edge_normal, p1 - a))) > max(8.0, min(width, height) * 0.012):
            continue
        t0, t1 = float(np.dot(p0 - a, direction)), float(np.dot(p1 - a, direction))
        start, end = max(lo, min(t0, t1)), min(hi, max(t0, t1))
        if end > start:
            intervals.append((start, end))
    if not intervals:
        return 0.0
    intervals.sort()
    covered = 0.0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end + 4:
            end = max(end, next_end)
        else:
            covered += end - start
            start, end = next_start, next_end
    covered += end - start
    return min(1.0, covered / visible_length)


def infer_court_region_from_lines(frame, lines):
    """Infer a supported court quadrilateral, including off-frame corners.

    Returns (polygon, diagnostics). Returns no polygon if independent line
    families or in-frame support for any of its four boundaries is missing.
    """
    if frame is None:
        return None, {"reliable": False, "reason": "no_frame", "support": []}
    height, width = frame.shape[:2]
    lines = _merge_collinear_segments(lines, width, height)
    if len(lines) < 4:
        return None, {"reliable": False, "reason": "fewer_than_four_boundary_candidates", "support": []}

    geom = [_unit_line(line) for line in lines]
    best = None
    candidates = []
    # Seed each direction family by observed segment direction. Families are
    # unoriented, allowing arbitrary phone rotation and either court side.
    for seed_i, seed in enumerate(geom):
        if seed is None:
            continue
        seed_angle = math.atan2(seed[2][1], seed[2][0]) % math.pi
        family_a = [i for i, g in enumerate(geom) if g is not None and _angle_distance(seed_angle, math.atan2(g[2][1], g[2][0]) % math.pi) <= math.radians(35)]
        family_b = [i for i, g in enumerate(geom) if g is not None and abs(_angle_distance(seed_angle, math.atan2(g[2][1], g[2][0]) % math.pi) - math.pi / 2) <= math.radians(35)]
        if len(family_a) < 2 or len(family_b) < 2:
            continue
        family_a = family_a[:10]
        family_b = family_b[:10]
        for a1, a2 in itertools.combinations(family_a, 2):
            # Reject duplicate/nearly coincident markings.
            if _angle_distance(math.atan2(geom[a1][2][1], geom[a1][2][0]), math.atan2(geom[a2][2][1], geom[a2][2][0])) > math.radians(35):
                continue
            for b1, b2 in itertools.combinations(family_b, 2):
                if _angle_distance(math.atan2(geom[b1][2][1], geom[b1][2][0]), math.atan2(geom[b2][2][1], geom[b2][2][0])) > math.radians(35):
                    continue
                corners = [_line_intersection(geom[a1], geom[b1]), _line_intersection(geom[a1], geom[b2]),
                           _line_intersection(geom[a2], geom[b2]), _line_intersection(geom[a2], geom[b1])]
                if any(p is None or not np.isfinite(p).all() for p in corners):
                    continue
                poly = np.asarray(corners, dtype=np.float32)
                if not cv2.isContourConvex(poly.reshape(-1, 1, 2)):
                    continue
                area = abs(float(cv2.contourArea(poly)))
                if area < width * height * 0.015 or area > width * height * 30:
                    continue
                # A plain rectangle could be a wall, screen, or frame. Require
                # at least one additional court marking inside the candidate,
                # aligned to one of the two independently supported court axes.
                boundary_indices = {a1, a2, b1, b2}
                internal_support = 0
                for k, g in enumerate(geom):
                    if g is None or k in boundary_indices:
                        continue
                    angle = math.atan2(g[2][1], g[2][0])
                    if min(_angle_distance(angle, math.atan2(geom[a1][2][1], geom[a1][2][0])),
                           _angle_distance(angle, math.atan2(geom[b1][2][1], geom[b1][2][0]))) > math.radians(35):
                        continue
                    inside_support = False
                    for source_segment in lines[k].get("_segments", [lines[k]]):
                        segment = _segment(source_segment)
                        inside = 0
                        for t in np.linspace(0.05, 0.95, 11):
                            p = segment[0] * (1 - t) + segment[1] * t
                            if cv2.pointPolygonTest(poly.reshape(-1, 1, 2), tuple(p), True) > max(5.0, min(width, height) * 0.01):
                                inside += 1
                        if inside >= 5:
                            inside_support = True
                            break
                    if inside_support:
                        internal_support += 1
                if internal_support == 0:
                    continue
                # Polygon edges run along family A, family B, family A,
                # family B respectively.
                selected = [lines[a1], lines[b2], lines[a2], lines[b1]]
                supports = [_visible_support(line, (poly[i], poly[(i + 1) % 4]), width, height) for i, line in enumerate(selected)]
                # Each inferred court boundary needs visible evidence. Permit
                # clipped edges and absent corners, but not wholly invented sides.
                if min(supports) < 0.16:
                    continue
                # Favor balanced edge evidence and well-conditioned corners.
                score = min(supports) + 0.25 * float(np.mean(supports)) + min(1.0, area / (width * height)) * 0.35
                candidates.append((score, poly, supports, internal_support))

    if not candidates:
        return None, {"reliable": False, "reason": "independent_supported_boundaries_not_found", "support": []}
    candidates.sort(key=lambda item: item[0], reverse=True)
    score, polygon, supports, internal_support = candidates[0]
    # Similar scoring for substantially different regions means the visible
    # line network does not uniquely constrain the playing area.
    for other in candidates[1:]:
        if score - other[0] > 0.025:
            break
        variants = [np.roll(other[1], shift, axis=0) for shift in range(4)]
        variants += [np.roll(other[1][::-1], shift, axis=0) for shift in range(4)]
        disagreement = min(float(np.mean(np.linalg.norm(polygon - variant, axis=1))) for variant in variants)
        if disagreement > math.hypot(width, height) * 0.04:
            return None, {"reliable": False, "reason": "multiple_competing_court_regions", "support": []}
    return polygon.astype(float).tolist(), {
        "reliable": True,
        "reason": "four_supported_lines_two_direction_families",
        "support": [round(float(v), 3) for v in supports],
        "internal_markings": int(internal_support),
        "corners_visible": all(0 <= p[0] < width and 0 <= p[1] < height for p in polygon),
    }
