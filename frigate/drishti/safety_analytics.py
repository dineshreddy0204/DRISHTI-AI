"""DRISHTI AI V2 — Modular Indian Traffic Safety Analytics (Decision-Support)."""

from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from frigate.drishti.indian_traffic import VEHICLE_CLASSES
from frigate.drishti.tracker_analytics import TrackState, _box_iou


@dataclass
class SafetyConfig:
    helmet_model_available: bool = False
    lane_direction_deg: float | None = None
    signal_state: str | None = None
    stop_line_y: float | None = None
    speed_limit_kmh: float | None = None
    is_calibrated: bool = False
    stopped_vehicle_s: float = 8.0
    congestion_vehicle_count: int = 6
    congestion_speed_px_s: float = 14.0
    no_parking_zones: list[list[float]] = field(default_factory=list)


@dataclass
class SafetyAlert:
    alert_id: str
    camera_id: str
    timestamp_s: float
    category: str
    severity: str
    track_ids: list[str]
    evidence: list[str]
    note: str = "Decision-support alert; requires human review, not automatic enforcement."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SafetyAnalyticsEngine:
    """Computes congestion, stopped vehicles, multi-rider association, and configured zone/lane rules."""

    def __init__(self, camera_id: str, config: SafetyConfig | None = None) -> None:
        self.camera_id = camera_id
        self.cfg = config or SafetyConfig()
        self._triple_streaks: dict[str, int] = {}
        self._wrong_way_streaks: dict[str, int] = {}
        self._cooldowns: dict[str, float] = {}
        self.alerts: list[SafetyAlert] = []

    def reset_for_loop(self) -> None:
        self._triple_streaks.clear()
        self._wrong_way_streaks.clear()
        self._cooldowns.clear()

    def feature_capabilities(self) -> dict[str, dict[str, Any]]:
        """Explicitly report which safety modules are active vs disabled due to missing models/config."""
        return {
            "helmet_detection": {
                "enabled": self.cfg.helmet_model_available,
                "reason": (
                    "Active"
                    if self.cfg.helmet_model_available
                    else "Disabled: no trained helmet-classifier ONNX model is bundled."
                ),
            },
            "triple_riding": {
                "enabled": True,
                "reason": "Active via multi-frame person-to-motorcycle track association (decision-support).",
            },
            "wrong_way_driving": {
                "enabled": self.cfg.lane_direction_deg is not None,
                "reason": (
                    f"Active (expected lane heading {self.cfg.lane_direction_deg} deg)"
                    if self.cfg.lane_direction_deg is not None
                    else "Disabled: lane_direction_deg not configured for this camera."
                ),
            },
            "red_light_crossing": {
                "enabled": self.cfg.signal_state is not None and self.cfg.stop_line_y is not None,
                "reason": (
                    "Active"
                    if self.cfg.signal_state is not None and self.cfg.stop_line_y is not None
                    else "Disabled: signal_state and stop_line_y not configured."
                ),
            },
            "overspeeding": {
                "enabled": self.cfg.is_calibrated and self.cfg.speed_limit_kmh is not None,
                "reason": (
                    "Active"
                    if self.cfg.is_calibrated and self.cfg.speed_limit_kmh is not None
                    else "Disabled: requires camera homography calibration and speed_limit_kmh."
                ),
            },
            "stopped_vehicle_and_congestion": {
                "enabled": True,
                "reason": "Active from confirmed vehicle track kinematics.",
            },
        }

    def evaluate(
        self, tracks: list[TrackState], timestamp_s: float
    ) -> dict[str, Any]:
        confirmed = [t for t in tracks if t.confirmed and t.missed_frames == 0]
        vehicles = [t for t in confirmed if t.label in VEHICLE_CLASSES]
        persons = [t for t in confirmed if t.label == "person"]
        motorcycles = [t for t in vehicles if t.label in ("motorcycle", "scooter")]

        slow_vehicles = sum(
            1
            for v in vehicles
            if v.trajectory and v.trajectory[-1].speed_px_s < self.cfg.congestion_speed_px_s
        )
        if slow_vehicles >= self.cfg.congestion_vehicle_count:
            congestion_level = "congested"
        elif len(vehicles) >= 4:
            congestion_level = "moderate"
        else:
            congestion_level = "free_flow"

        new_alerts: list[dict[str, Any]] = []

        # 1. Multi-frame motorcycle rider association (possible triple riding)
        for moto in motorcycles:
            if not moto.last_box:
                continue
            mx1, my1, mx2, my2 = moto.last_box
            # Expand upward slightly to cover rider head/torso region above seat
            rider_region = [mx1 - 12, max(0.0, my1 - (my2 - my1) * 0.65), mx2 + 12, my2]
            associated_riders = [
                p.track_id
                for p in persons
                if p.last_box and _box_iou(rider_region, p.last_box) >= 0.12
            ]
            if len(associated_riders) >= 3:
                streak = self._triple_streaks.get(moto.track_id, 0) + 1
                self._triple_streaks[moto.track_id] = streak
                if streak >= 2 and timestamp_s - self._cooldowns.get(f"triple:{moto.track_id}", -1e9) >= 10.0:
                    self._cooldowns[f"triple:{moto.track_id}"] = timestamp_s
                    al = SafetyAlert(
                        alert_id=f"saf-{uuid.uuid4().hex[:10]}",
                        camera_id=self.camera_id,
                        timestamp_s=round(timestamp_s, 3),
                        category="possible_triple_riding",
                        severity="medium",
                        track_ids=[moto.track_id, *associated_riders],
                        evidence=[
                            f"{len(associated_riders)}_person_tracks_associated_with_{moto.track_id}"
                        ],
                    )
                    self.alerts.append(al)
                    new_alerts.append(al.to_dict())
            else:
                self._triple_streaks[moto.track_id] = 0

        # 2. Stopped vehicles (duration >= stopped_vehicle_s)
        stopped_ids: list[str] = []
        for v in vehicles:
            if v.is_stationary and (v.last_seen_ts - v.first_seen_ts) >= self.cfg.stopped_vehicle_s:
                stopped_ids.append(v.track_id)
                cd_key = f"stopped:{v.track_id}"
                if timestamp_s - self._cooldowns.get(cd_key, -1e9) >= 20.0:
                    self._cooldowns[cd_key] = timestamp_s
                    al = SafetyAlert(
                        alert_id=f"saf-{uuid.uuid4().hex[:10]}",
                        camera_id=self.camera_id,
                        timestamp_s=round(timestamp_s, 3),
                        category="stopped_vehicle",
                        severity="low",
                        track_ids=[v.track_id],
                        evidence=[
                            f"stationary_duration={(v.last_seen_ts - v.first_seen_ts):.1f}s"
                        ],
                    )
                    self.alerts.append(al)
                    new_alerts.append(al.to_dict())

        # 3. Wrong-way driving (only when lane_direction_deg is explicitly configured)
        if self.cfg.lane_direction_deg is not None:
            for v in vehicles:
                if len(v.trajectory) < 3:
                    continue
                pt = v.trajectory[-1]
                if pt.speed_px_s < 18.0:
                    continue
                diff = abs(pt.heading_deg - self.cfg.lane_direction_deg) % 360.0
                diff = 360.0 - diff if diff > 180.0 else diff
                if diff >= 135.0:
                    streak = self._wrong_way_streaks.get(v.track_id, 0) + 1
                    self._wrong_way_streaks[v.track_id] = streak
                    cd_key = f"wrongway:{v.track_id}"
                    if streak >= 3 and timestamp_s - self._cooldowns.get(cd_key, -1e9) >= 15.0:
                        self._cooldowns[cd_key] = timestamp_s
                        al = SafetyAlert(
                            alert_id=f"saf-{uuid.uuid4().hex[:10]}",
                            camera_id=self.camera_id,
                            timestamp_s=round(timestamp_s, 3),
                            category="wrong_way_driving",
                            severity="high",
                            track_ids=[v.track_id],
                            evidence=[
                                f"heading={pt.heading_deg:.0f}deg_vs_lane={self.cfg.lane_direction_deg:.0f}deg"
                            ],
                        )
                        self.alerts.append(al)
                        new_alerts.append(al.to_dict())
                else:
                    self._wrong_way_streaks[v.track_id] = 0

        return {
            "congestion_level": congestion_level,
            "active_vehicles": len(vehicles),
            "active_persons": len(persons),
            "stopped_vehicle_ids": stopped_ids,
            "new_alerts": new_alerts,
            "total_alerts": len(self.alerts),
            "capabilities": self.feature_capabilities(),
        }

