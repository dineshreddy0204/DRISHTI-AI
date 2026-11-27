"""DRISHTI AI V2 — Predictive Collision Risk & Near-Miss Engine (with homography support)."""

from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from frigate.drishti.indian_traffic import VEHICLE_CLASSES
from frigate.drishti.tracker_analytics import TrackState


@dataclass
class CalibrationConfig:
    """Optional 4-point perspective homography from image pixels to ground meters."""

    calibrated: bool = False
    image_points: list[list[float]] | None = None
    ground_points_m: list[list[float]] | None = None

    def build_homography(self) -> np.ndarray | None:
        if (
            not self.calibrated
            or not self.image_points
            or not self.ground_points_m
            or len(self.image_points) < 4
            or len(self.ground_points_m) < 4
        ):
            return None
        src = np.asarray(self.image_points, dtype=np.float32)
        dst = np.asarray(self.ground_points_m, dtype=np.float32)
        H, _ = cv2.findHomography(src, dst)
        return H


@dataclass
class CollisionRiskEvent:
    risk_id: str
    camera_id: str
    timestamp_s: float
    track_a: str
    track_b: str
    calibrated: bool
    unit_space: str
    relative_speed: float
    relative_distance: float
    min_projected_separation: float
    ttc_s: float | None
    pet_s: float | None
    risk_level: str
    near_miss: bool
    uncertainty_note: str
    contributing_evidence: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CollisionRiskEngine:
    """Estimates trajectory intersection, TTC, PET, and near-miss risk."""

    def __init__(
        self,
        camera_id: str,
        calibration: CalibrationConfig | None = None,
        ttc_alert_s: float = 2.0,
        near_miss_gap_px: float = 38.0,
        near_miss_gap_m: float = 1.8,
        cooldown_s: float = 6.0,
    ) -> None:
        self.camera_id = camera_id
        self.calibration = calibration or CalibrationConfig()
        self.H = self.calibration.build_homography()
        self.is_calibrated = self.H is not None
        self.ttc_alert_s = ttc_alert_s
        self.near_miss_gap_px = near_miss_gap_px
        self.near_miss_gap_m = near_miss_gap_m
        self.cooldown_s = cooldown_s
        self._cooldowns: dict[str, float] = {}
        self.risk_events: list[CollisionRiskEvent] = []

    def reset_for_loop(self) -> None:
        self._cooldowns.clear()

    def _project_point(self, x: float, y: float) -> tuple[float, float]:
        if self.H is None:
            return x, y
        pt = np.array([[[x, y]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.H)[0][0]
        return float(out[0]), float(out[1])

    def _track_kinematics(
        self, t: TrackState
    ) -> tuple[float, float, float, float] | None:
        if len(t.trajectory) < 2:
            return None
        p0, p1 = t.trajectory[-2], t.trajectory[-1]
        dt = max(1e-3, p1.timestamp_s - p0.timestamp_s)
        # Bottom-center ground contact point
        x0, y0 = p0.cx, p0.xyxy[3]
        x1, y1 = p1.cx, p1.xyxy[3]
        gx0, gy0 = self._project_point(x0, y0)
        gx1, gy1 = self._project_point(x1, y1)
        vx = (gx1 - gx0) / dt
        vy = (gy1 - gy0) / dt
        return gx1, gy1, vx, vy

    def evaluate(
        self, tracks: list[TrackState], timestamp_s: float
    ) -> dict[str, Any]:
        active = [
            t
            for t in tracks
            if t.confirmed
            and t.missed_frames == 0
            and (t.label in VEHICLE_CLASSES or t.label == "person")
            and len(t.trajectory) >= 2
        ]
        gap_thresh = self.near_miss_gap_m if self.is_calibrated else self.near_miss_gap_px
        unit_space = "ground_meters" if self.is_calibrated else "image_pixels_uncalibrated"
        new_risks: list[dict[str, Any]] = []

        for i in range(len(active)):
            ka = self._track_kinematics(active[i])
            if ka is None:
                continue
            xa, ya, vxa, vya = ka
            for j in range(i + 1, len(active)):
                kb = self._track_kinematics(active[j])
                if kb is None:
                    continue
                if active[i].label == "person" and active[j].label == "person":
                    continue
                xb, yb, vxb, vyb = kb

                rx = xb - xa
                ry = yb - ya
                rvx = vxb - vxa
                rvy = vyb - vya

                dist = math.hypot(rx, ry)
                rel_speed = math.hypot(rvx, rvy)
                if rel_speed < (1.2 if self.is_calibrated else 18.0):
                    continue

                # Closest approach time t* = - (r . v) / |v|^2
                rv_dot = rx * rvx + ry * rvy
                if rv_dot >= 0:
                    # Moving apart, not converging
                    continue

                ttc = -rv_dot / (rel_speed * rel_speed)
                if ttc > 5.0:
                    continue

                min_sep = math.hypot(rx + rvx * ttc, ry + rvy * ttc)
                if min_sep > gap_thresh:
                    continue

                # Estimate PET if trajectories cross near current midpoint
                pet_s = round(max(0.0, ttc * 0.85), 3) if min_sep < gap_thresh else None
                near_miss = ttc <= self.ttc_alert_s and min_sep <= gap_thresh
                risk_level = (
                    "high"
                    if ttc <= 1.0 and min_sep <= gap_thresh * 0.6
                    else "medium"
                    if near_miss
                    else "low"
                )
                cd_key = f"{active[i].track_id}:{active[j].track_id}"
                if timestamp_s - self._cooldowns.get(cd_key, -1e9) < self.cooldown_s:
                    continue
                self._cooldowns[cd_key] = timestamp_s

                note = (
                    "Homography-calibrated ground-plane estimate; subject to perspective error."
                    if self.is_calibrated
                    else (
                        "Uncalibrated image-space estimate (pixels). No physical km/h, "
                        "meters, or exact collision time claimed without camera homography."
                    )
                )
                ev = CollisionRiskEvent(
                    risk_id=f"risk-{uuid.uuid4().hex[:10]}",
                    camera_id=self.camera_id,
                    timestamp_s=round(timestamp_s, 3),
                    track_a=active[i].track_id,
                    track_b=active[j].track_id,
                    calibrated=self.is_calibrated,
                    unit_space=unit_space,
                    relative_speed=round(rel_speed, 2),
                    relative_distance=round(dist, 2),
                    min_projected_separation=round(min_sep, 2),
                    ttc_s=round(ttc, 3),
                    pet_s=pet_s,
                    risk_level=risk_level,
                    near_miss=near_miss,
                    uncertainty_note=note,
                    contributing_evidence=[
                        f"converging_rel_speed={rel_speed:.1f}",
                        f"ttc={ttc:.2f}s",
                        f"min_sep={min_sep:.1f}",
                    ],
                )
                self.risk_events.append(ev)
                new_risks.append(ev.to_dict())

        return {
            "calibrated": self.is_calibrated,
            "unit_space": unit_space,
            "new_risks": new_risks,
            "total_risks": len(self.risk_events),
        }

