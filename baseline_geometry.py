"""Conservative geometry helpers for an explicitly user-confirmed baseline."""
import math


def normalize_baseline_config(config):
    if not isinstance(config, dict) or config.get("type") != "baseline_calibration":
        return None
    points = config.get("points")
    court_type = config.get("court_type")
    if court_type not in ("singles", "doubles") or not isinstance(points, list) or len(points) != 3:
        return None
    normalized = []
    try:
        for point in points:
            x, y = float(point["x"]), float(point["y"])
            if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x <= 1 and 0 <= y <= 1):
                return None
            normalized.append((x, y))
    except (KeyError, TypeError, ValueError):
        return None
    start, end, inside = normalized
    if math.dist(start, end) < 0.15:
        return None
    cross = (end[0] - start[0]) * (inside[1] - start[1]) - (end[1] - start[1]) * (inside[0] - start[0])
    if abs(cross) / math.dist(start, end) < 0.04:
        return None
    return {"type": "baseline_calibration", "points": normalized, "court_type": court_type,
            "previously_locked": config.get("previously_locked") is True}


def baseline_pixels(config, width, height):
    if not config or width <= 0 or height <= 0:
        return None
    return [(x * (width - 1), y * (height - 1)) for x, y in config["points"]]


def match_baseline_observation(lines, config, width, height):
    """Return conservative Hough support for the line explicitly marked by the user."""
    points = baseline_pixels(config, width, height)
    if points is None:
        return {"valid": False, "confidence": 0.0, "reason": "baseline_calibration_missing"}
    start, end, _inside = points
    vx, vy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(vx, vy)
    if length < 1:
        return {"valid": False, "confidence": 0.0, "reason": "baseline_line_too_short"}
    ux, uy = vx / length, vy / length
    tolerance = max(5.0, min(width, height) * 0.008)
    intervals = []
    for line in lines or []:
        try:
            p0 = (float(line["x1"]), float(line["y1"]))
            p1 = (float(line["x2"]), float(line["y2"]))
        except (KeyError, TypeError, ValueError):
            continue
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        segment_length = math.hypot(dx, dy)
        if segment_length < 8:
            continue
        alignment = abs((dx * ux + dy * uy) / segment_length)
        if alignment < math.cos(math.radians(6)):
            continue
        d0 = abs(ux * (p0[1] - start[1]) - uy * (p0[0] - start[0]))
        d1 = abs(ux * (p1[1] - start[1]) - uy * (p1[0] - start[0]))
        if max(d0, d1) > tolerance:
            continue
        t0 = (p0[0] - start[0]) * ux + (p0[1] - start[1]) * uy
        t1 = (p1[0] - start[0]) * ux + (p1[1] - start[1]) * uy
        lo, hi = max(0.0, min(t0, t1)), min(length, max(t0, t1))
        if hi > lo:
            intervals.append((lo, hi))
    if not intervals:
        return {"valid": False, "confidence": 0.0, "reason": "marked_baseline_not_visible"}
    intervals.sort()
    covered = 0.0
    lo, hi = intervals[0]
    for next_lo, next_hi in intervals[1:]:
        if next_lo <= hi + 4:
            hi = max(hi, next_hi)
        else:
            covered += hi - lo
            lo, hi = next_lo, next_hi
    covered += hi - lo
    coverage = min(1.0, covered / length)
    valid = coverage >= 0.35
    return {"valid": valid, "confidence": round(coverage, 3),
            "reason": "marked_baseline_supported" if valid else "insufficient_marked_baseline_support"}


def classify_baseline_landing(landing, config, width, height, tracking_confidence, threshold=0.65):
    """Classify only a supported landing whose projection falls on the marked baseline segment."""
    unknown = {"result": "UNKNOWN", "inside": None, "reason": "baseline_landing_evidence_insufficient"}
    try:
        confidence = float(tracking_confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    if not landing or not config or not math.isfinite(confidence) or confidence < threshold:
        return unknown
    if landing.get("method") not in ("downward_phase_bounce", "downward_phase_settle"):
        return {**unknown, "reason": "landing_method_not_confident"}
    points = baseline_pixels(config, width, height)
    if points is None:
        return {**unknown, "reason": "baseline_geometry_unavailable"}
    start, end, inside = points
    vx, vy = end[0] - start[0], end[1] - start[1]
    length_sq = vx * vx + vy * vy
    if length_sq <= 0:
        return {**unknown, "reason": "baseline_geometry_invalid"}
    try:
        px, py = float(landing["x"]), float(landing["y"])
    except (KeyError, TypeError, ValueError):
        return {**unknown, "reason": "landing_point_invalid"}
    if not math.isfinite(px) or not math.isfinite(py) or not (0 <= px < width and 0 <= py < height):
        return {**unknown, "reason": "landing_point_outside_frame"}
    t = ((px - start[0]) * vx + (py - start[1]) * vy) / length_sq
    if t < -0.05 or t > 1.05:
        return {**unknown, "reason": "landing_outside_marked_baseline_extent"}
    cross = vx * (py - start[1]) - vy * (px - start[0])
    inside_cross = vx * (inside[1] - start[1]) - vy * (inside[0] - start[0])
    is_inside = cross * inside_cross >= 0
    result = "IN" if is_inside else "OUT"
    return {"result": result, "inside": is_inside, "reason": "landing_on_or_inside_marked_baseline" if is_inside else "landing_beyond_marked_baseline"}
