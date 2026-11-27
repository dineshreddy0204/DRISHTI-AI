"""Comprehensive regression tests for DRISHTI AI V2 modules."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from frigate.drishti.accident_engine import AccidentConfig, TemporalAccidentEngine
from frigate.drishti.camera_health import classify_camera_source, redact_stream_url
from frigate.drishti.collision_risk import CalibrationConfig, CollisionRiskEngine
from frigate.drishti.indian_traffic import map_detection_label
from frigate.drishti.model_registry import MODEL_REGISTRY, list_models
from frigate.drishti.safety_analytics import SafetyAnalyticsEngine, SafetyConfig
from frigate.drishti.tracker_analytics import TrackAnalyticsEngine
from frigate.util.model import post_process_yolo26, post_process_yolo_v8_11


class DrishtiV2Tests(unittest.TestCase):
    def test_indian_taxonomy_honest_mapping_and_unknown_vehicle(self):
        # Never turn car into auto_rickshaw
        m1 = map_detection_label("car", 0.85, score_threshold=0.35)
        self.assertEqual(m1.label, "car")
        self.assertFalse(m1.is_uncertain)

        # Low-confidence vehicle becomes unknown_vehicle
        m2 = map_detection_label(
            "car", 0.28, score_threshold=0.35, uncertain_floor=0.25
        )
        self.assertEqual(m2.label, "unknown_vehicle")
        self.assertTrue(m2.is_uncertain)

    def test_yolo_v8_11_and_yolo26_decoders(self):
        # (1, 84, 2) synthetic YOLOv8/11 output
        preds = np.zeros((1, 84, 2), dtype=np.float32)
        preds[0, :4, 0] = [200.0, 200.0, 80.0, 60.0]
        preds[0, 6, 0] = 0.9  # class 2 (car)
        out = post_process_yolo_v8_11(preds, 640, 640, score_threshold=0.4)
        self.assertEqual(int(out[0, 0]), 2)
        self.assertAlmostEqual(float(out[0, 1]), 0.9, places=4)

        # (1, 2, 6) synthetic YOLO26 NMS-free output
        preds26 = np.array(
            [[[100.0, 100.0, 200.0, 200.0, 0.88, 3.0], [0, 0, 0, 0, 0.1, 0]]],
            dtype=np.float32,
        )
        out26 = post_process_yolo26(preds26, 640, 640, score_threshold=0.4)
        self.assertEqual(int(out26[0, 0]), 3)
        self.assertAlmostEqual(float(out26[0, 1]), 0.88, places=4)

    def test_model_registry_marks_missing_weights_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            models = {m["id"]: m for m in list_models(Path(tmp))}
            self.assertEqual(models["yolox-nano-416"]["availability"]["status"], "unavailable_weights_missing")
            self.assertIn("auto_rickshaw", MODEL_REGISTRY["yolox-nano-416"].unavailable_labels)

    def test_tracker_dedup_and_loop_isolation(self):
        trk = TrackAnalyticsEngine(camera_id="cam1", fps=5.0, min_confirmed_frames=2)
        # Duplicate overlapping boxes in frame 1
        dets = [
            {"label": "car", "score": 0.9, "xyxy": [100, 100, 200, 180]},
            {"label": "car", "score": 0.6, "xyxy": [102, 102, 198, 178]},
        ]
        r1 = trk.update(dets, 0.0)
        self.assertEqual(r1["per_frame_detections"], 1)
        self.assertEqual(r1["current_visible_objects"], 0)  # unconfirmed on frame 1

        r2 = trk.update(
            [{"label": "car", "score": 0.88, "xyxy": [110, 100, 210, 180]}], 0.2
        )
        self.assertEqual(r2["current_visible_objects"], 1)
        self.assertEqual(r2["unique_tracks_total"], 1)

        trk.reset_for_loop(1)
        self.assertEqual(trk.unique_confirmed_ids, {})

    def test_accident_engine_suppresses_overlap_only_and_normal_traffic(self):
        trk = TrackAnalyticsEngine(camera_id="cam1", fps=5.0, min_confirmed_frames=2)
        acc = TemporalAccidentEngine(camera_id="cam1", config=AccidentConfig())

        # Two cars passing steadily at constant speed with brief overlap
        for step in range(6):
            t = step * 0.2
            dets = [
                {"label": "car", "score": 0.9, "xyxy": [100 + step * 5, 100, 160 + step * 5, 150]},
                {"label": "car", "score": 0.9, "xyxy": [130 + step * 5, 110, 190 + step * 5, 160]},
            ]
            trk.update(dets, t)
            res = acc.evaluate(list(trk.tracks.values()), t)
            self.assertEqual(res["state"], "Normal")
        self.assertEqual(len(acc.incidents), 0)

    def test_accident_engine_detects_convergence_contact_and_sudden_stop(self):
        trk = TrackAnalyticsEngine(camera_id="cam1", fps=5.0, min_confirmed_frames=2)
        acc = TemporalAccidentEngine(camera_id="cam1", config=AccidentConfig())

        # Car A moves right steadily toward Car B, then collides and stops abruptly
        positions_a = [
            [40, 100, 100, 150],
            [56, 100, 116, 150],
            [72, 100, 132, 150],
            [88, 100, 148, 150],
            [104, 100, 164, 150],
            [120, 100, 180, 150],
            [140, 100, 200, 150],  # contact + decel
            [140, 100, 200, 150],
            [140, 100, 200, 150],
            [140, 100, 200, 150],
        ]
        positions_b = [
            [235, 100, 295, 150],
            [223, 100, 283, 150],
            [211, 100, 271, 150],
            [199, 100, 259, 150],
            [187, 100, 247, 150],
            [175, 100, 235, 150],
            [175, 100, 235, 150],
            [175, 100, 235, 150],
            [175, 100, 235, 150],
            [175, 100, 235, 150],
        ]
        for step, (ba, bb) in enumerate(zip(positions_a, positions_b)):
            t = step * 0.2
            trk.update(
                [
                    {"label": "car", "score": 0.9, "xyxy": ba},
                    {"label": "car", "score": 0.9, "xyxy": bb},
                ],
                t,
            )
            acc.evaluate(list(trk.tracks.values()), t)

        self.assertGreaterEqual(len(acc.incidents), 1)
        self.assertIn(
            acc.incidents[0].state,
            ("Suspected collision", "Probable accident"),
        )

    def test_collision_risk_uncalibrated_vs_calibrated(self):
        risk_uncal = CollisionRiskEngine("cam1", CalibrationConfig(calibrated=False))
        self.assertFalse(risk_uncal.is_calibrated)

        cal = CalibrationConfig(
            calibrated=True,
            image_points=[[0, 0], [640, 0], [640, 360], [0, 360]],
            ground_points_m=[[0, 0], [20, 0], [20, 12], [0, 12]],
        )
        risk_cal = CollisionRiskEngine("cam1", cal)
        self.assertTrue(risk_cal.is_calibrated)

    def test_label_smoothing_rejects_single_frame_car_flip(self):
        trk = TrackAnalyticsEngine(
            camera_id="cam1", fps=5.0, min_confirmed_frames=2
        )
        trk.update(
            [{"label": "car", "score": 0.9, "xyxy": [100, 100, 200, 180]}],
            0.0,
        )
        trk.update(
            [{"label": "car", "score": 0.88, "xyxy": [102, 100, 202, 180]}],
            0.2,
        )
        trk.update(
            [{"label": "truck", "score": 0.91, "xyxy": [102, 100, 202, 180]}],
            0.4,
        )
        state = next(iter(trk.tracks.values()))
        self.assertEqual(state.label, "car")

        trk.update(
            [{"label": "car", "score": 0.95, "xyxy": [104, 100, 204, 180]}],
            0.6,
        )
        state = next(iter(trk.tracks.values()))
        self.assertEqual(state.label, "car")

    def test_accident_engine_suppresses_moving_camera_looming_false_positive(self):
        trk = TrackAnalyticsEngine(
            camera_id="cam1", fps=5.0, min_confirmed_frames=2
        )
        acc = TemporalAccidentEngine(
            camera_id="cam1",
            config=AccidentConfig(confirm_frames=2),
        )
        for step in range(16):
            extent = 100 + step * 22
            trk.update(
                [{"label": "car", "score": 0.9, "xyxy": [0, 0, extent, min(360, 80 + step * 8)]}],
                step * 0.2,
            )
            acc.evaluate(
                list(trk.tracks.values()),
                step * 0.2,
                frame_area=640.0 * 360.0,
                camera_motion={
                    "dx": 12.0,
                    "dy": 12.0,
                    "shift_mag": 17.0,
                    "response": 0.9,
                },
            )
        self.assertEqual(len(acc.incidents), 0)

    def test_safety_capabilities_honest_reporting(self):
        saf = SafetyAnalyticsEngine("cam1", SafetyConfig())
        caps = saf.feature_capabilities()
        self.assertFalse(caps["helmet_detection"]["enabled"])
        self.assertFalse(caps["wrong_way_driving"]["enabled"])
        self.assertTrue(caps["triple_riding"]["enabled"])

    def test_camera_source_classification_and_redaction(self):
        replay_cfg = SimpleNamespace(
            ffmpeg=SimpleNamespace(
                inputs=[SimpleNamespace(path="/opt/drishti/demo/hyderabad.mp4", input_args="-re -stream_loop -1")]
            )
        )
        self.assertEqual(classify_camera_source(replay_cfg, 3.0), "VIDEO REPLAY")
        self.assertEqual(
            redact_stream_url("rtsp://admin:secret123@192.168.1.50:554/stream1"),
            "rtsp://***:***@192.168.1.50:554/stream1",
        )
