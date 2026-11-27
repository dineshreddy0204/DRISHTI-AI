"""Unit & integration tests for MoveNet pose, temporal fall detection, SOS test mode, and public safety."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from frigate.drishti.fall_detection import FallState, TemporalFallEngine
from frigate.drishti.pose_estimator import MoveNetPoseEstimator
from frigate.drishti.public_safety import PublicSafetyEngine
from frigate.drishti.sos_engine import IncidentSOSStore, RiskLevel


class DrishtiHumanSafetyTests(unittest.TestCase):
    def test_movenet_pose_on_real_assets_if_present(self):
        assets = Path("/opt/drishti/demo")
        if not (assets / "movenet_lightning.onnx").is_file():
            self.skipTest("movenet_lightning.onnx not present")
        est = MoveNetPoseEstimator(assets)
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        res = est.estimate(blank)
        self.assertEqual(len(res["keypoints"]), 17)
        self.assertEqual(res["reliability"], "insufficient_evidence")

    def test_fall_engine_suppresses_sitting_and_bending(self):
        engine = TemporalFallEngine()
        # Gradual sitting transition (hip_y_norm changes slowly, torso angle stays < 40 deg)
        for i in range(6):
            t = i * 0.25
            res = engine.update(
                "CAM-001-T1",
                t,
                {
                    "posture": "standing" if i < 2 else "sitting",
                    "torso_angle_deg": 10.0 + i * 3.0,
                    "hip_y_norm": 0.45 + i * 0.015,
                    "reliability": "good",
                },
                320.0,
                240.0 + i * 4.0,
            )
            self.assertEqual(res["fall_state"], FallState.NORMAL.value)

    def test_fall_engine_detects_rapid_fall_and_recovery(self):
        engine = TemporalFallEngine(sos_trigger_inactivity_s=0.4, confirm_frames=2)
        seq = [
            (0.0, "standing", 8.0, 0.35, 320.0, 180.0),
            (0.25, "standing", 12.0, 0.37, 320.0, 185.0),
            (0.6, "lying_horizontal", 72.0, 0.72, 322.0, 330.0),
            (0.9, "lying_horizontal", 75.0, 0.74, 322.0, 331.0),
            (1.2, "lying_horizontal", 76.0, 0.74, 322.0, 331.0),
        ]
        last = {}
        for ts, post, ang, hip, cx, cy in seq:
            last = engine.update(
                "CAM-001-T1",
                ts,
                {
                    "posture": post,
                    "torso_angle_deg": ang,
                    "hip_y_norm": hip,
                    "reliability": "good",
                },
                cx,
                cy,
            )
        self.assertIn(
            last["fall_state"],
            (FallState.SUSPECTED_FALL.value, FallState.SOS_PENDING.value),
        )
        # Recovery back to standing
        rec = engine.update(
            "CAM-001-T1",
            2.0,
            {
                "posture": "standing",
                "torso_angle_deg": 10.0,
                "hip_y_norm": 0.38,
                "reliability": "good",
            },
            322.0,
            190.0,
        )
        self.assertEqual(rec["fall_state"], FallState.RECOVERED.value)

    def test_sos_store_persistence_and_test_mode_enforcement(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "v2.db"
            store = IncidentSOSStore(db_path=db)
            cams = store.list_cameras()
            self.assertEqual(cams[0]["camera_id"], "CAM-001")
            self.assertEqual(cams[0]["source_type"], "LIVE WEBCAM")

            sos = store.trigger_sos(
                camera_id="CAM-001",
                event_type="possible_human_fall",
                evidence=["rapid_posture_transition"],
                countdown_seconds=15,
            )
            self.assertEqual(sos["state"], "COUNTDOWN")
            self.assertEqual(sos["notification_status"], "Test only")

            resp = store.respond_sos(sos["sos_id"], "i_am_okay")
            self.assertEqual(resp["state"], "RECOVERED")

            # Re-open store to verify SQLite persistence
            store2 = IncidentSOSStore(db_path=db)
            incidents = store2.list_incidents()
            self.assertEqual(len(incidents), 1)
            self.assertEqual(incidents[0]["review_status"], "RECOVERED")

    def test_public_safety_tampering_and_disabled_models(self):
        ps = PublicSafetyEngine("CAM-001")
        caps = ps.capabilities()
        self.assertFalse(caps["fire_and_smoke"]["enabled"])
        self.assertFalse(caps["physical_altercation"]["enabled"])

        covered = np.zeros((240, 320, 3), dtype=np.uint8)
        alerts = ps.check_tampering(covered)
        self.assertTrue(any(a["event_type"] == "camera_tampering_covered" for a in alerts))

