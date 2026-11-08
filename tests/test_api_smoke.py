import json
import os
import socket
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch
from websockets.sync.client import connect as websocket_connect
from websockets.exceptions import ConnectionClosed

import cv2
import numpy as np
import uvicorn

import main
from main import app


class ApiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"))
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()
        deadline = time.time() + 10
        while not cls.server.started and time.time() < deadline:
            time.sleep(0.05)
        if not cls.server.started:
            raise RuntimeError("Uvicorn did not start")
        cls.port = cls.server.servers[0].sockets[0].getsockname()[1]
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.thread.join(timeout=5)

    def request(self, path, data=None, headers=None):
        req = urllib.request.Request(self.base + path, data=data, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=20) as response:
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.headers, exc.read()

    def test_health(self):
        status, _, body = self.request("/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "healthy")

    def test_live_websocket_accepts_frame_and_returns_state(self):
        with websocket_connect(f"ws://127.0.0.1:{self.port}/ws/analyze-live", open_timeout=5) as ws:
            initial = json.loads(ws.recv(timeout=5))
            self.assertEqual(initial["state"], "SEARCHING_FOR_COURT")
            ok, encoded = cv2.imencode(".jpg", np.zeros((240, 320, 3), dtype=np.uint8))
            self.assertTrue(ok)
            ws.send(encoded.tobytes())
            update = json.loads(ws.recv(timeout=15))
            self.assertEqual(update["state"], "SEARCHING_FOR_COURT")
            self.assertEqual(update["decision"]["result"], "UNKNOWN")

    def test_live_websocket_reports_invalid_baseline_handshake(self):
        with websocket_connect(f"ws://127.0.0.1:{self.port}/ws/analyze-live", open_timeout=5) as ws:
            initial = json.loads(ws.recv(timeout=5))
            self.assertEqual(initial["state"], "SEARCHING_FOR_COURT")
            ws.send("not a binary JPEG frame")
            update = json.loads(ws.recv(timeout=5))
            self.assertEqual(update["state"], "BASELINE_SEARCH")
            self.assertEqual(update["decision"]["reason"], "baseline_confirmation_invalid")

    def test_live_websocket_baseline_handshake_then_binary_frame(self):
        config = {"type": "baseline_calibration", "court_type": "singles", "points": [
            {"x": 0.1, "y": 0.5}, {"x": 0.9, "y": 0.5}, {"x": 0.5, "y": 0.75},
        ]}
        with websocket_connect(f"ws://127.0.0.1:{self.port}/ws/analyze-live", open_timeout=5) as ws:
            initial = json.loads(ws.recv(timeout=5))
            self.assertEqual(initial["state"], "SEARCHING_FOR_COURT")
            ws.send(json.dumps(config))
            handshake = json.loads(ws.recv(timeout=5))
            self.assertEqual(handshake["mode"], "baseline_live")
            self.assertEqual(handshake["decision"]["reason"], "baseline_confirmation_started")
            ok, encoded = cv2.imencode(".jpg", np.zeros((240, 320, 3), dtype=np.uint8))
            self.assertTrue(ok)
            ws.send(encoded.tobytes())
            update = json.loads(ws.recv(timeout=15))
            self.assertEqual(update["mode"], "baseline_live")
            self.assertEqual(update["frame_index"], 1)
            self.assertEqual(update["analysis_frame_index"], 1)

    def test_live_websocket_invalid_jpeg_does_not_desynchronize_frame_indices(self):
        ok, encoded = cv2.imencode(".jpg", np.zeros((240, 320, 3), dtype=np.uint8))
        self.assertTrue(ok)
        with websocket_connect(f"ws://127.0.0.1:{self.port}/ws/analyze-live", open_timeout=5) as ws:
            ws.recv(timeout=5)
            ws.send(b"not a JPEG")
            rejected = json.loads(ws.recv(timeout=5))
            self.assertEqual(rejected["decision"]["reason"], "frame_decode_failed")
            self.assertEqual(rejected["frame_index"], 1)
            self.assertIsNone(rejected["analysis_frame_index"])
            ws.send(encoded.tobytes())
            analyzed = json.loads(ws.recv(timeout=15))
            self.assertEqual(analyzed["frame_index"], 2)
            self.assertEqual(analyzed["analysis_frame_index"], 1)

    def test_live_websocket_client_disconnect_keeps_service_available(self):
        with websocket_connect(f"ws://127.0.0.1:{self.port}/ws/analyze-live", open_timeout=5) as ws:
            self.assertEqual(json.loads(ws.recv(timeout=5))["state"], "SEARCHING_FOR_COURT")
            ws.close()
        status, _, body = self.request("/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "healthy")

    def test_live_websocket_logs_processing_exception_and_closes_safely(self):
        ok, encoded = cv2.imencode(".jpg", np.zeros((240, 320, 3), dtype=np.uint8))
        self.assertTrue(ok)
        with patch.object(main.LiveFrameAnalyzer, "process_frame", side_effect=RuntimeError("controlled test failure")):
            with websocket_connect(f"ws://127.0.0.1:{self.port}/ws/analyze-live", open_timeout=5) as ws:
                ws.recv(timeout=5)
                ws.send(encoded.tobytes())
                with patch.object(main.logger, "exception", wraps=main.logger.exception) as log_exception:
                    with self.assertRaises(ConnectionClosed) as closed:
                        ws.recv(timeout=5)
                log_exception.assert_called_once_with("Live WebSocket frame processing failed")
                self.assertEqual(closed.exception.rcvd.code, 1011)
                self.assertEqual(closed.exception.rcvd.reason, "Live frame processing failed")

    def test_malformed_and_empty_uploads(self):
        boundary = "smokeboundary"
        malformed = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"video\"; filename=\"bad.mp4\"\r\n"
                     "Content-Type: video/mp4\r\n\r\nnot a video\r\n"
                     f"--{boundary}--\r\n").encode()
        status, _, body = self.request("/analyze-video", malformed, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        self.assertEqual(status, 400, body.decode(errors="replace"))
        empty = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"video\"; filename=\"empty.mp4\"\r\n"
                 "Content-Type: video/mp4\r\n\r\n\r\n"
                 f"--{boundary}--\r\n").encode()
        status, _, body = self.request("/analyze-video", empty, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        self.assertEqual(status, 400, body.decode(errors="replace"))

    def test_static_clips_at_30_and_60_fps_return_unknown(self):
        for fps in (30, 60):
            with tempfile.TemporaryDirectory() as temp:
                clip = os.path.join(temp, "static.mp4")
                writer = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
                self.assertTrue(writer.isOpened(), "MP4 encoder unavailable")
                for _ in range(12):
                    writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
                writer.release()
                with open(clip, "rb") as source:
                    video = source.read()
            boundary = "smokeboundary"
            body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"video\"; filename=\"static.mp4\"\r\n"
                    "Content-Type: video/mp4\r\n\r\n").encode() + video + f"\r\n--{boundary}--\r\n".encode()
            status, _, response = self.request("/analyze-video", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
            self.assertEqual(status, 200, response.decode(errors="replace"))
            result = json.loads(response)
            self.assertEqual(result["data"]["decision"]["result"], "UNKNOWN")
            self.assertEqual(result["decision"]["result"], "UNKNOWN")
            self.assertFalse(result["court_detection"]["diagnostics"]["reliable"])


if __name__ == "__main__":
    unittest.main()
