"""Security, authorization, path-traversal, and test-mode isolation regression tests."""

import unittest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from frigate.api import auth as frigate_auth
from frigate.api import drishti_intelligence, drishti_video


def _make_test_app(role: str) -> TestClient:
    app = FastAPI()
    app.include_router(drishti_video.router)
    app.include_router(drishti_intelligence.router)
    app.dependency_overrides[frigate_auth.get_current_user] = lambda: {
        "username": f"test_{role}",
        "role": role,
    }
    return TestClient(app)


class DrishtiSecurityTests(unittest.TestCase):
    def test_viewer_blocked_from_admin_routes(self):
        client = _make_test_app("viewer")
        r1 = client.post(
            "/drishti/v2/contacts",
            json={"name": "Test", "channel": "sms", "target": "+919999999999"},
        )
        self.assertEqual(r1.status_code, 403)

        r2 = client.post(
            "/drishti/v2/cameras/test",
            json={"url": "rtsp://127.0.0.1:554/stream"},
        )
        self.assertEqual(r2.status_code, 403)

        r3 = client.get("/drishti/v2/audit-log")
        self.assertEqual(r3.status_code, 403)

    def test_path_traversal_rejected_on_artifacts_and_evidence(self):
        client = _make_test_app("admin")
        r1 = client.get("/drishti/analysis/not-a-uuid/report.json")
        self.assertEqual(r1.status_code, 422)

        r2 = client.get("/drishti/v2/evidence/..%2F..%2Fetc%2Fpasswd.jpg")
        self.assertIn(r2.status_code, (404, 422))

    def test_test_mode_cannot_be_bypassed_on_manual_sos(self):
        client = _make_test_app("admin")
        r = client.post(
            "/drishti/v2/sos/manual",
            json={"camera_id": "CAM-001", "note": "Security test SOS"},
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["test_mode"])
        self.assertEqual(body["sos"]["notification_status"], "Test only")
        sos_id = body["sos"]["sos_id"]
        c = client.post(f"/drishti/v2/sos/{sos_id}/respond", json={"action": "cancel"})
        self.assertEqual(c.status_code, 200)

