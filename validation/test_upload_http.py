"""Exercise background uploads over a real loopback socket using the image dependencies."""

import hashlib
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import uvicorn
from fastapi import FastAPI

from frigate.api import drishti_video
from frigate.config import FrigateConfig


class BackgroundUploadHTTPTest(unittest.TestCase):
    def test_upload_acknowledged_and_downloadable_after_reconnect(self):
        app = FastAPI()
        app.frigate_config = FrigateConfig.load()
        app.include_router(drishti_video.router)
        assets = Path(__file__).resolve().parents[1] / "demo"
        route = "/drishti/analysis"
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(drishti_video, "ROOT", Path(temp)),
            patch.object(drishti_video, "ASSETS", assets),
            socket.socket() as listener,
        ):
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            base = f"http://127.0.0.1:{listener.getsockname()[1]}"
            server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
            thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]})
            thread.start()
            try:
                deadline = time.monotonic() + 20
                while not server.started and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(server.started)
                clip = (assets / "hyderabad.mp4").read_bytes()
                # These are the headers supplied by the authenticated reverse proxy.
                headers = {"remote-role": "admin"}
                with httpx.Client(
                    base_url=base, headers=headers, trust_env=False
                ) as client:
                    ack = client.post(
                        route,
                        params={"background": "true"},
                        files={"file": ("demo.mp4", clip)},
                    )
                    self.assertEqual(ack.status_code, 202)
                    key = ack.json()["id"]
                with httpx.Client(
                    base_url=base, headers=headers, trust_env=False
                ) as client:
                    deadline = time.monotonic() + 180
                    while time.monotonic() < deadline:
                        state = client.get(route + "/status").json()
                        self.assertEqual(state["id"], key)
                        if state["status"] != "running":
                            break
                        time.sleep(0.1)
                    self.assertEqual(state["status"], "completed")
                    self.assertEqual(client.get(route).json()[0]["id"], key)
                    report = client.get(f"{route}/{key}/report.json").json()
                    video = client.get(f"{route}/{key}/review.mp4")
                    self.assertEqual(
                        report["input_sha256"], hashlib.sha256(clip).hexdigest()
                    )
                    self.assertEqual(report["sample_count"], 40)
                    self.assertGreater(report["box_count"], 0)
                    self.assertEqual(video.status_code, 200)
                    self.assertEqual(video.content[4:8], b"ftyp")
            finally:
                server.should_exit = True
                thread.join(timeout=20)
                self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
