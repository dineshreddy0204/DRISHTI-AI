"""Run against the release image's real auth dependency, model, FFmpeg and router."""

import fcntl
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from frigate.api import drishti_video
from frigate.config import FrigateConfig


class UploadIntegrationTest(unittest.TestCase):
    def test_role_validation_and_model_outputs(self):
        app = FastAPI()
        app.frigate_config = FrigateConfig.load()
        app.include_router(drishti_video.router)
        root = Path(__file__).resolve().parents[1]
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(drishti_video, "ROOT", Path(temp) / "results"),
            patch.object(drishti_video, "ASSETS", root / "demo"),
            TestClient(app) as client,
        ):
            # nginx normally supplies the authenticated role.
            route = "/drishti/analysis"
            self.assertEqual(client.get(route).status_code, 403)
            self.assertEqual(
                client.get(route, headers={"remote-role": "viewer"}).status_code, 403
            )
            client.headers.update({"remote-role": "admin"})
            self.assertEqual(client.get(route + "/status").json()["status"], "idle")
            self.assertEqual(client.get(route).json(), [])
            self.assertEqual(
                client.post(
                    route, files={"file": ("bad.mp4", io.BytesIO(b"bad"))}
                ).status_code,
                422,
            )
            with (root / "demo/hyderabad.mp4").open("rb") as f:
                response = client.post(
                    route, files={"file": ("../sample.mp4", f, "video/mp4")}
                )
            self.assertEqual(response.status_code, 201, response.text[:200])
            summary = response.json()
            self.assertEqual(summary["name"], "sample.mp4")
            self.assertEqual(summary["sample_count"], 40)
            self.assertGreater(summary["box_count"], 0)
            self.assertEqual(client.get(route).json()[0]["id"], summary["id"])
            base = f"{route}/{summary['id']}/"
            self.assertEqual(client.get(base + "review.mp4").content[4:8], b"ftyp")
            self.assertEqual(
                len(client.get(base + "report.json").json()["samples"]), 40
            )
            self.assertEqual(client.get(base + "source.mp4").status_code, 422)
            with (root / "demo/hyderabad.mp4").open("rb") as f:
                ack = client.post(
                    route + "?background=true", files={"file": ("replay.mp4", f)}
                )
            self.assertEqual(ack.status_code, 202)
            self.assertEqual(ack.json()["status"], "running")
            key = ack.json()["id"]
            self.assertEqual(
                client.get(route + "/status").json(), {"id": key, "status": "completed"}
            )
            history = client.get(route).json()
            self.assertEqual(len(history), 2)
            self.assertEqual(history[0]["sample_count"], 40)
            with (drishti_video.ROOT / ".lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                self.assertEqual(
                    client.post(
                        route + "?background=true",
                        files={"file": ("busy.mp4", b"busy")},
                    ).status_code,
                    409,
                )
            drishti_video.write_state("10000000-0000-4000-8000-000000000000", "running")
            self.assertEqual(
                client.get(route + "/status").json()["status"], "interrupted"
            )


if __name__ == "__main__":
    unittest.main()
