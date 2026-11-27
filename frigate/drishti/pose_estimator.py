"""Genuine MoveNet SinglePose Lightning ONNX human pose estimator."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from frigate.drishti.model_registry import get_model_spec

KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 5),
    (0, 6),
)


class MoveNetPoseEstimator:
    """CPU ONNX Runtime MoveNet SinglePose Lightning (192x192 int32 RGB)."""

    SIZE = 192

    def __init__(self, assets_dir: Path, intra_op_threads: int = 2) -> None:
        self.spec = get_model_spec("movenet-lightning-192")
        avail = self.spec.availability(assets_dir)
        if not avail["available"]:
            raise RuntimeError(f"Pose model unavailable: {avail['status']}")
        self.model_sha256 = avail["actual_sha256"]
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, intra_op_threads)
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            avail["path"], sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def estimate(
        self,
        frame_bgr: np.ndarray,
        person_box: list[float] | None = None,
        kp_conf_threshold: float = 0.25,
    ) -> dict[str, Any]:
        """Run pose estimation on full frame or padded person crop.

        Returns genuine keypoints (never fabricated), torso angle, posture,
        and reliability status.
        """
        fh, fw = frame_bgr.shape[:2]
        if person_box is not None:
            x1, y1, x2, y2 = person_box
            bw = max(16.0, x2 - x1)
            bh = max(16.0, y2 - y1)
            pad_x, pad_y = bw * 0.15, bh * 0.15
            cx1 = max(0, int(x1 - pad_x))
            cy1 = max(0, int(y1 - pad_y))
            cx2 = min(fw, int(x2 + pad_x))
            cy2 = min(fh, int(y2 + pad_y))
        else:
            cx1, cy1, cx2, cy2 = 0, 0, fw, fh

        crop = frame_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return {
                "keypoints": [],
                "skeleton_edges": SKELETON_EDGES,
                "reliability": "insufficient_evidence",
                "posture": "insufficient_evidence",
                "torso_angle_deg": None,
                "hip_y_norm": None,
                "mean_confidence": 0.0,
                "visible_keypoints": 0,
                "latency_ms": 0.0,
            }

        ch, cw = crop.shape[:2]
        scale = self.SIZE / max(ch, cw)
        rw, rh = max(1, int(cw * scale)), max(1, int(ch * scale))
        canvas = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(cv2.resize(crop, (rw, rh)), cv2.COLOR_BGR2RGB)
        canvas[:rh, :rw] = rgb
        tensor = canvas[None].astype(np.int32)

        t0 = time.perf_counter()
        raw = self.session.run(None, {self.input_name: tensor})[0]
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        kps_raw = raw[0, 0]  # (17, 3): [y, x, score]
        keypoints: list[dict[str, Any]] = []
        visible = 0
        conf_sum = 0.0
        for idx, (yn, xn, conf) in enumerate(kps_raw):
            px = cx1 + (float(xn) * self.SIZE) / scale
            py = cy1 + (float(yn) * self.SIZE) / scale
            px = float(np.clip(px, 0, fw))
            py = float(np.clip(py, 0, fh))
            c = float(conf)
            is_vis = c >= kp_conf_threshold
            if is_vis:
                visible += 1
                conf_sum += c
            keypoints.append(
                {
                    "name": KEYPOINT_NAMES[idx],
                    "x": round(px, 2),
                    "y": round(py, 2),
                    "confidence": round(c, 4),
                    "visible": is_vis,
                }
            )

        mean_conf = round(conf_sum / visible, 4) if visible else 0.0
        if visible >= 8 and mean_conf >= 0.35:
            reliability = "good"
        elif visible >= 4:
            reliability = "reduced"
        else:
            reliability = "insufficient_evidence"

        torso_angle_deg: float | None = None
        hip_y_norm: float | None = None
        posture = "insufficient_evidence"

        ls, rs = keypoints[5], keypoints[6]
        lh, rh_kp = keypoints[11], keypoints[12]
        lk, rk = keypoints[13], keypoints[14]

        if (ls["visible"] or rs["visible"]) and (lh["visible"] or rh_kp["visible"]):
            sx = (
                (ls["x"] + rs["x"]) * 0.5
                if (ls["visible"] and rs["visible"])
                else (ls["x"] if ls["visible"] else rs["x"])
            )
            sy = (
                (ls["y"] + rs["y"]) * 0.5
                if (ls["visible"] and rs["visible"])
                else (ls["y"] if ls["visible"] else rs["y"])
            )
            hx = (
                (lh["x"] + rh_kp["x"]) * 0.5
                if (lh["visible"] and rh_kp["visible"])
                else (lh["x"] if lh["visible"] else rh_kp["x"])
            )
            hy = (
                (lh["y"] + rh_kp["y"]) * 0.5
                if (lh["visible"] and rh_kp["visible"])
                else (lh["y"] if lh["visible"] else rh_kp["y"])
            )
            dx = abs(sx - hx)
            dy = abs(hy - sy)
            torso_angle_deg = round(math.degrees(math.atan2(dx, max(dy, 1e-3))), 1)
            hip_y_norm = round(hy / max(fh, 1), 4)

            if torso_angle_deg >= 58.0:
                posture = "lying_horizontal"
            elif torso_angle_deg >= 32.0:
                posture = "bending"
            elif (lk["visible"] or rk["visible"]) and person_box is not None:
                ky = (
                    (lk["y"] + rk["y"]) * 0.5
                    if (lk["visible"] and rk["visible"])
                    else (lk["y"] if lk["visible"] else rk["y"])
                )
                bh = max(1.0, person_box[3] - person_box[1])
                if abs(ky - hy) / bh < 0.18:
                    posture = "sitting"
                else:
                    posture = "standing"
            else:
                posture = "standing"

        return {
            "keypoints": keypoints,
            "skeleton_edges": SKELETON_EDGES,
            "reliability": reliability,
            "posture": posture,
            "torso_angle_deg": torso_angle_deg,
            "hip_y_norm": hip_y_norm,
            "mean_confidence": mean_conf,
            "visible_keypoints": visible,
            "latency_ms": latency_ms,
        }

