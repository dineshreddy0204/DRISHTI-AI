"""Native Windows Webcam Bridge for DRISHTI AI V2 (CAM-001).

Captures frames from the local Windows webcam using OpenCV (DirectShow/MSMF)
and serves a local-only MJPEG/JPEG HTTP stream on 127.0.0.1:8765 so Docker/WSL2
or local tools can pull live frames via http://host.docker.internal:8765.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

try:
    import cv2
except ImportError as exc:
    raise SystemExit(
        "OpenCV is required for the native webcam bridge. "
        "Install it with: pip install opencv-python-headless"
    ) from exc


class WebcamCaptureManager:
    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480, fps: int = 15):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self._cap = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_jpeg: bytes | None = None
        self._last_ts: float = 0.0
        self._frames_captured: int = 0
        self._reconnects: int = 0
        self._last_error: str | None = None

    @staticmethod
    def list_devices(max_index: int = 5) -> list[dict]:
        found = []
        for idx in range(max_index):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                found.append({"index": idx, "name": f"Windows Camera {idx}", "width": w, "height": h})
                cap.release()
        return found

    def start(self, device_index: int | None = None) -> dict:
        with self._lock:
            if device_index is not None:
                self.device_index = device_index
            if self._running:
                return self.status()
            self._running = True
            self._last_error = None
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self.status()

    def stop(self) -> dict:
        with self._lock:
            self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
        return self.status()

    def _open_camera(self):
        cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.device_index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
        return cap

    def _loop(self) -> None:
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                self._cap = self._open_camera()
                if not self._cap.isOpened():
                    self._last_error = f"Cannot open webcam index {self.device_index}"
                    self._reconnects += 1
                    time.sleep(1.5)
                    continue
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self._last_error = "Webcam frame read failed; reconnecting"
                self._reconnects += 1
                self._cap.release()
                self._cap = None
                time.sleep(0.5)
                continue
            ok_enc, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok_enc:
                with self._lock:
                    self._last_jpeg = buf.tobytes()
                    self._last_ts = time.time()
                    self._frames_captured += 1
                    self._last_error = None
            time.sleep(max(0.01, 1.0 / self.fps))

    def latest_jpeg(self) -> tuple[bytes | None, float]:
        with self._lock:
            return self._last_jpeg, self._last_ts

    def status(self) -> dict:
        with self._lock:
            return {
                "camera_id": "CAM-001",
                "camera_name": "Laptop Webcam",
                "source_type": "LIVE WEBCAM",
                "running": self._running,
                "device_index": self.device_index,
                "frames_captured": self._frames_captured,
                "last_frame_ts": self._last_ts,
                "reconnects": self._reconnects,
                "error": self._last_error,
            }


def make_handler(mgr: WebcamCaptureManager):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/health":
                self._send_json(200, mgr.status())
            elif path == "/devices":
                self._send_json(200, {"devices": mgr.list_devices()})
            elif path == "/snapshot.jpg":
                jpg, ts = mgr.latest_jpeg()
                if jpg is None:
                    self._send_json(503, {"error": "No frame captured yet"})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("X-Frame-Timestamp", str(ts))
                self.send_header("Content-Length", str(len(jpg)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(jpg)
            elif path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        jpg, _ = mgr.latest_jpeg()
                        if jpg:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                        time.sleep(max(0.03, 1.0 / mgr.fps))
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self._send_json(404, {"error": "Not found"})

        def do_POST(self):
            path = urlparse(self.path).path
            if path == "/start":
                self._send_json(200, mgr.start())
            elif path == "/stop":
                self._send_json(200, mgr.stop())
            else:
                self._send_json(404, {"error": "Not found"})

        def log_message(self, format, *args):
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    mgr = WebcamCaptureManager(device_index=args.device)
    mgr.start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(mgr))
    print(f"DRISHTI Webcam Bridge listening on http://{args.host}:{args.port} (CAM-001)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        mgr.stop()
        server.server_close()


if __name__ == "__main__":
    main()

