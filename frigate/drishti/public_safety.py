"""DRISHTI AI V2 — Outdoor Public Safety & Camera Tampering Intelligence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from frigate.drishti.tracker_analytics import TrackState


def _point_in_poly(x: float, y: float, poly: list[list[float]]) -> bool:
    if len(poly) < 3:
        return False
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
        ):
            inside = not inside
        j = i
    return inside


@dataclass
class PublicSafetyConfig:
    loitering_seconds: float = 20.0
    abandoned_object_seconds: float = 25.0
    crowd_threshold: int = 8
    restricted_zones: list[dict[str, Any]] = field(default_factory=list)
    fire_smoke_model_available: bool = False
    altercation_model_available: bool = False


class PublicSafetyEngine:
    """Observable outdoor public safety & camera tampering checks (no criminal profiling)."""

    RULE_VERSION = "drishti-public-safety-v2.0"

    def __init__(self, camera_id: str, config: PublicSafetyConfig | None = None) -> None:
        self.camera_id = camera_id
        self.cfg = config or PublicSafetyConfig()
        self._zone_entry_ts: dict[tuple[str, str], float] = {}
        self._last_frame_hash: str | None = None
        self._frozen_count: int = 0

    def capabilities(self) -> dict[str, dict[str, Any]]:
        return {
            "loitering_prolonged_presence": {
                "enabled": True,
                "reason": "Active via anonymous track duration (neutral activity label).",
            },
            "restricted_area_intrusion": {
                "enabled": len(self.cfg.restricted_zones) > 0,
                "reason": (
                    f"Active ({len(self.cfg.restricted_zones)} zones)"
                    if self.cfg.restricted_zones
                    else "Disabled: no restricted polygon zones configured."
                ),
            },
            "abandoned_object": {
                "enabled": True,
                "reason": "Active for stationary non-person objects (possible_unattended_object).",
            },
            "camera_tampering": {
                "enabled": True,
                "reason": "Active (blackout/cover, severe blur, frozen-frame checks).",
            },
            "crowd_safety": {
                "enabled": True,
                "reason": "Active via aggregate anonymous person count.",
            },
            "fire_and_smoke": {
                "enabled": self.cfg.fire_smoke_model_available,
                "reason": (
                    "Active"
                    if self.cfg.fire_smoke_model_available
                    else "Disabled: no trained fire/smoke ONNX model bundled."
                ),
            },
            "physical_altercation": {
                "enabled": self.cfg.altercation_model_available,
                "reason": (
                    "Active"
                    if self.cfg.altercation_model_available
                    else "Disabled: no validated temporal altercation ONNX model bundled."
                ),
            },
        }

    def check_tampering(self, frame_bgr: np.ndarray) -> list[dict[str, Any]]:
        """Detect camera covering/blackout, severe blur, or frozen frame."""
        alerts: list[dict[str, Any]] = []
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mean_val = float(gray.mean())
        std_val = float(gray.std())
        if mean_val < 12.0 and std_val < 6.0:
            alerts.append(
                {
                    "event_type": "camera_tampering_covered",
                    "risk_level": "HIGH",
                    "evidence": [f"mean_luma={mean_val:.1f}", f"std={std_val:.1f}"],
                }
            )
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_var < 15.0 and mean_val >= 12.0:
            alerts.append(
                {
                    "event_type": "camera_tampering_severe_blur",
                    "risk_level": "LOW",
                    "evidence": [f"laplacian_var={lap_var:.1f}"],
                }
            )
        small = cv2.resize(gray, (32, 32)).tobytes()
        h = hashlib.md5(small).hexdigest()
        if h == self._last_frame_hash:
            self._frozen_count += 1
        else:
            self._frozen_count = 0
            self._last_frame_hash = h
        if self._frozen_count >= 15:
            alerts.append(
                {
                    "event_type": "camera_stream_frozen",
                    "risk_level": "HIGH",
                    "evidence": [f"identical_frames={self._frozen_count}"],
                }
            )
        return alerts

    def evaluate_tracks(
        self, tracks: list[TrackState], timestamp_s: float
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        confirmed = [t for t in tracks if t.confirmed and t.missed_frames == 0]
        persons = [t for t in confirmed if t.label == "person"]

        if len(persons) >= self.cfg.crowd_threshold:
            events.append(
                {
                    "event_type": "crowd_congestion",
                    "risk_level": "LOW",
                    "track_ids": [p.track_id for p in persons],
                    "evidence": [f"anonymous_person_count={len(persons)}"],
                }
            )

        for t in confirmed:
            if not t.last_box:
                continue
            cx = (t.last_box[0] + t.last_box[2]) * 0.5
            cy = t.last_box[3]
            duration = max(0.0, t.last_seen_ts - t.first_seen_ts)

            if t.label == "person" and duration >= self.cfg.loitering_seconds:
                events.append(
                    {
                        "event_type": "prolonged_presence",
                        "risk_level": "LOW",
                        "track_ids": [t.track_id],
                        "evidence": [f"observed_duration={duration:.1f}s"],
                    }
                )

            if (
                t.label in ("backpack", "suitcase", "handbag", "road_obstacle")
                and t.is_stationary
                and duration >= self.cfg.abandoned_object_seconds
            ):
                events.append(
                    {
                        "event_type": "possible_unattended_object",
                        "risk_level": "LOW",
                        "track_ids": [t.track_id],
                        "evidence": [
                            f"label={t.label}",
                            f"stationary_duration={duration:.1f}s",
                        ],
                    }
                )

            for zone in self.cfg.restricted_zones:
                zname = zone.get("name", "restricted")
                poly = zone.get("polygon", [])
                if _point_in_poly(cx, cy, poly):
                    key = (t.track_id, zname)
                    if key not in self._zone_entry_ts:
                        self._zone_entry_ts[key] = timestamp_s
                    zdur = timestamp_s - self._zone_entry_ts[key]
                    events.append(
                        {
                            "event_type": "restricted_area_intrusion",
                            "risk_level": "HIGH",
                            "track_ids": [t.track_id],
                            "evidence": [f"zone={zname}", f"duration={zdur:.1f}s"],
                        }
                    )

        return events

