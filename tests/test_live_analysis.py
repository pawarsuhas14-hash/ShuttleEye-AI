import builtins
import json
import math
import os

import cv2
import numpy as np
from fastapi.testclient import TestClient

import main


def test_geometry_change_detects_translation_rotation_scale_and_perspective():
    reference = np.array([[80, 60], [560, 65], [550, 360], [90, 350]], dtype=np.float32)
    assert not main._court_geometry_changed(reference, reference + [3, 2], (400, 640, 3))
    assert main._court_geometry_changed(reference, reference + [30, 0], (400, 640, 3))

    center = reference.mean(axis=0)
    angle = math.radians(4)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    rotated = (reference - center) @ rotation.T + center
    assert main._court_geometry_changed(reference, rotated, (400, 640, 3))
    assert main._court_geometry_changed(reference, center + (reference - center) * 1.08, (400, 640, 3))
    perspective = reference.copy()
    perspective[0] += [0, 25]
    assert main._court_geometry_changed(reference, perspective, (400, 640, 3))


def test_phase_correlation_detects_translation_in_portrait_and_landscape():
    rng = np.random.default_rng(7)
    for height, width in ((240, 320), (320, 240)):
        image = rng.integers(0, 255, (height, width), dtype=np.uint8)
        shifted = cv2.warpAffine(image, np.float32([[1, 0, 9], [0, 1, 0]]), (width, height))
        assert main._global_translation_detected(image, shifted)
        assert not main._global_translation_detected(image, image.copy())


def test_landing_bounds_use_actual_portrait_and_landscape_frame_dimensions():
    trajectory = [
        {"frame": i, "x": 1150 + i * 8, "y": y}
        for i, y in enumerate((300, 400, 500, 580, 650, 580))
    ]
    assert main.choose_landing_point(trajectory, 8, frame_size=(720, 1280)) is not None

    portrait = [
        {"frame": i, "x": 500 + i * 4, "y": y}
        for i, y in enumerate((300, 550, 800, 1100, 1450, 1350))
    ]
    assert main.choose_landing_point(portrait, 8, frame_size=(1700, 800)) is not None


def _patch_live_session(monkeypatch, confidence=0.0, decision="UNKNOWN"):
    corners = {
        "top_left": [20, 20],
        "top_right": [220, 20],
        "bottom_right": [220, 180],
        "bottom_left": [20, 180],
    }
    monkeypatch.setattr(main, "get_court_corners", lambda frame: corners)
    monkeypatch.setattr(main, "build_court_region", lambda frame, found: (corners, "geometry_polygon"))
    monkeypatch.setattr(main, "detect_court", lambda frame: {"lines": []})
    monkeypatch.setattr(main, "build_static_line_mask", lambda frame, analysis: np.zeros(frame.shape[:2], dtype=np.uint8))
    monkeypatch.setattr(
        main,
        "extract_frame_candidates",
        lambda frame, previous, previous_previous, static_line_mask=None: (np.zeros((20, 20), dtype=np.uint8), []),
    )
    trajectory = [
        {"frame": i, "x": 60 + i * 10, "y": y}
        for i, y in enumerate((30, 60, 90, 130, 160, 120))
    ]
    monkeypatch.setattr(main, "track_shuttle", lambda history: (trajectory, confidence))
    monkeypatch.setattr(main, "choose_landing_point", lambda *args, **kwargs: {"frame": 0, "x": 60, "y": 30})
    monkeypatch.setattr(
        main,
        "check_landing_inside_court",
        lambda point, region: {"result": decision, "inside": decision == "IN"},
    )
    return corners


def test_geometry_is_reused_between_scheduled_checks(monkeypatch):
    corners = _patch_live_session(monkeypatch)
    calls = 0

    def detect(frame):
        nonlocal calls
        calls += 1
        return corners

    monkeypatch.setattr(main, "get_court_corners", detect)
    session = main.LiveAnalysisSession("test-session")
    frame = np.zeros((200, 240, 3), dtype=np.uint8)
    for index in range(4):
        session.process_frame(frame, index)
    assert calls == 1
    session.process_frame(frame, 4)
    assert calls == 2
    assert session.candidate_frames.maxlen == 16


def test_camera_translation_invalidates_tracking_and_emits_recalibrating(monkeypatch):
    corners = _patch_live_session(monkeypatch)
    session = main.LiveAnalysisSession("moving-camera")
    frame = np.zeros((200, 240, 3), dtype=np.uint8)
    for index in range(4):
        session.process_frame(frame, index)
    for point in corners.values():
        point[0] += 30
    update = session.process_frame(frame, 4)
    assert update["state"] == "RECALIBRATING"
    assert update["decision"]["result"] == "UNKNOWN"
    assert session.reference_corners is not None


def test_insufficient_tracking_never_emits_in_or_out(monkeypatch):
    _patch_live_session(monkeypatch, confidence=0.2, decision="IN")
    session = main.LiveAnalysisSession("safe-session")
    frame = np.zeros((200, 240, 3), dtype=np.uint8)
    result = session.process_frame(frame, 0)
    assert result["state"] != "IN"
    assert result["decision"]["result"] == "UNKNOWN"


def test_landing_event_id_is_stable_for_repeated_frames(monkeypatch):
    _patch_live_session(monkeypatch, confidence=0.8, decision="IN")
    session = main.LiveAnalysisSession("stable-session")
    frame = np.zeros((200, 240, 3), dtype=np.uint8)
    first = session.process_frame(frame, 50)
    second = session.process_frame(frame, 51)
    assert first["decision"]["event_id"] == "stable-session:50"
    assert second["decision"]["event_id"] == first["decision"]["event_id"]


def test_websocket_malformed_frame_disconnect_and_reconnect():
    client = TestClient(main.app)
    with client.websocket_connect("/ws/analyze-live") as websocket:
        assert websocket.receive_json()["state"] == "SEARCHING_FOR_COURT"
        websocket.send_text(json.dumps({"type": "session", "session_id": "stable-client", "next_frame_id": 40}))
        websocket.send_bytes(b"not a JPEG")
        update = websocket.receive_json()
        assert update["state"] == "UNKNOWN"
        assert update["decision"]["result"] == "UNKNOWN"
        assert update["frame_id"] == 40

    with client.websocket_connect("/ws/analyze-live") as websocket:
        websocket.receive_json()
        websocket.send_text(json.dumps({"type": "session", "session_id": "stable-client", "next_frame_id": 41}))
        websocket.send_bytes(b"not a JPEG")
        update = websocket.receive_json()
        assert update["state"] == "UNKNOWN"
        assert update["frame_id"] == 41


def test_websocket_valid_jpeg_returns_json_analysis_update(monkeypatch):
    _patch_live_session(monkeypatch)
    ok, encoded = cv2.imencode(".jpg", np.zeros((200, 240, 3), dtype=np.uint8))
    assert ok

    client = TestClient(main.app)
    with client.websocket_connect("/ws/analyze-live") as websocket:
        initial = websocket.receive_json()
        assert initial["state"] == "SEARCHING_FOR_COURT"

        websocket.send_bytes(encoded.tobytes())
        update = websocket.receive_json()

    assert update["state"] == "LANDING_ANALYSIS"
    assert update["frame_id"] == 0
    assert update["geometry"]["ready"] is True
    assert update["shuttle_tracking"]["tracked_points"] == 6
    assert update["decision"]["result"] == "UNKNOWN"


def test_health_endpoint_is_available():
    response = TestClient(main.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_video_route_keeps_multipart_field_and_decision_contract():
    parameters = main.analyze_video.__annotations__
    assert "video" in parameters
    source = open(main.__file__, encoding="utf-8").read()
    video_handler = source[source.index('async def analyze_video'):]
    assert '"decision": {' in video_handler
    assert '"result"' in video_handler


def test_analyze_video_accepts_multipart_generated_video_and_returns_decision(
    monkeypatch, tmp_path
):
    video_path = tmp_path / "small-test.avi"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (64, 48)
    )
    assert writer.isOpened(), "OpenCV MJPG video writer is required for this integration test"
    for index in range(3):
        frame = np.full((48, 64, 3), index * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    # The endpoint uses /tmp, which is not a writable temp directory on Windows.
    # Map only its generated upload path to pytest's isolated temporary directory.
    actual_open = builtins.open

    def open_in_test_temp(path, *args, **kwargs):
        if isinstance(path, (str, os.PathLike)) and os.fspath(path).startswith("/tmp/"):
            path = tmp_path / os.path.basename(os.fspath(path))
        return actual_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", open_in_test_temp)
    actual_video_capture = cv2.VideoCapture

    def capture_uploaded_video(path, *args, **kwargs):
        path = os.fspath(path)
        if path.startswith("/tmp/"):
            path = str(tmp_path / os.path.basename(path))
        return actual_video_capture(path, *args, **kwargs)

    monkeypatch.setattr(main.cv2, "VideoCapture", capture_uploaded_video)
    corners = {
        "top_left": [5, 5],
        "top_right": [58, 5],
        "bottom_right": [58, 42],
        "bottom_left": [5, 42],
    }
    monkeypatch.setattr(
        main,
        "detect_court",
        lambda frame: {"court_detected": True, "court_lines_detected": 4, "lines": []},
    )
    monkeypatch.setattr(main, "get_court_corners", lambda frame: corners)
    monkeypatch.setattr(
        main,
        "build_court_region",
        lambda frame, found: (corners, "geometry_polygon"),
    )
    monkeypatch.setattr(
        main,
        "build_static_line_mask",
        lambda frame, analysis: np.zeros(frame.shape[:2], dtype=np.uint8),
    )
    monkeypatch.setattr(
        main,
        "extract_frame_candidates",
        lambda frame, previous, previous_previous, static_line_mask=None: (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), []
        ),
    )
    monkeypatch.setattr(main, "track_shuttle", lambda history: ([], 0.0))

    client = TestClient(main.app)
    with video_path.open("rb") as video_file:
        response = client.post(
            "/analyze-video",
            files={"video": ("small-test.avi", video_file, "video/x-msvideo")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["video_info"]["total_frames"] == 3
    assert payload["video_info"]["resolution"] == {"width": 64, "height": 48}
    assert payload["decision"]["result"] == "UNKNOWN"
    assert "confidence" in payload["decision"]
