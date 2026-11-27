"""DRISHTI AI V2 — Temporal Accident Detection, Camera-Motion Compensation & Evidence Fusion."""

from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import cv2
import numpy as np

from frigate.drishti.indian_traffic import VEHICLE_CLASSES
from frigate.drishti.tracker_analytics import TrackState, _box_iou


class IncidentState(str, Enum):
    NORMAL = "Normal"
    POTENTIAL_RISK = "Potential risk"
    SUSPECTED_COLLISION = "Suspected collision"
    PROBABLE_ACCIDENT = "Probable accident"
    HUMAN_CONFIRMED = "Human-confirmed accident"
    DISMISSED = "Dismissed incident"


@dataclass
class AccidentConfig:
    min_history_frames: int = 7
    min_moving_px_s: float = 38.0
    max_sustained_px_s: float = 220.0
    decel_drop_ratio: float = 0.70
    heading_change_deg: float = 65.0
    contact_iou_threshold: float = 0.14
    contact_gap_px: float = 10.0
    min_closing_px_s: float = 30.0
    looming_area_ratio: float = 2.1
    looming_min_frame_frac: float = 0.18
    congestion_vehicle_count: int = 6
    congestion_speed_px_s: float = 14.0
    confirm_frames: int = 3
    cooldown_s: float = 12.0
    stop_zones: list[list[float]] = field(default_factory=list)


@dataclass
class IncidentRecord:
    event_id: str
    camera_id: str
    source_type: str
    timestamp_s: float
    category: str
    state: str
    severity: str
    evidence_score: float
    evidence_score_note: str
    model_version: str
    rule_version: str
    involved_track_ids: list[str]
    contributing_evidence: list[str]
    bbox_union: list[float]
    review_status: str = "unreviewed"
    snapshot_file: str | None = None
    clip_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_camera_motion(
    prev_gray: np.ndarray | None, curr_gray: np.ndarray
) -> dict[str, float]:
    """Estimate global camera translation (dx, dy) and shock magnitude via phase correlation."""
    if prev_gray is None or prev_gray.shape != curr_gray.shape:
        return {"dx": 0.0, "dy": 0.0, "shift_mag": 0.0, "response": 0.0}
    f1 = cv2.resize(prev_gray, (160, 90)).astype(np.float32)
    f2 = cv2.resize(curr_gray, (160, 90)).astype(np.float32)
    (dx, dy), response = cv2.phaseCorrelate(f1, f2)
    scale_x = curr_gray.shape[1] / 160.0
    scale_y = curr_gray.shape[0] / 90.0
    dx_px = float(dx * scale_x)
    dy_px = float(dy * scale_y)
    return {
        "dx": round(dx_px, 2),
        "dy": round(dy_px, 2),
        "shift_mag": round(math.hypot(dx_px, dy_px), 2),
        "response": round(float(response), 3),
    }


def _angle_diff_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return 360.0 - d if d > 180.0 else d


def _box_edge_gap(a: list[float], b: list[float]) -> float:
    dx = max(0.0, max(a[0] - b[2], b[0] - a[2]))
    dy = max(0.0, max(a[1] - b[3], b[1] - a[3]))
    return math.hypot(dx, dy)


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


def _smoothed_speed(pts: list[Any]) -> float:
    if len(pts) < 2:
        return 0.0
    dt = max(1e-3, pts[-1].timestamp_s - pts[0].timestamp_s)
    return math.hypot(pts[-1].cx - pts[0].cx, pts[-1].cy - pts[0].cy) / dt


def _is_sustained_motion(pts: list[Any]) -> bool:
    """Distinguish true multi-frame motion from a 1-frame tracker ID-jump spike."""
    if len(pts) < 3:
        return False
    step1 = math.hypot(pts[1].cx - pts[0].cx, pts[1].cy - pts[0].cy)
    step2 = math.hypot(pts[2].cx - pts[1].cx, pts[2].cy - pts[1].cy)
    # Require motion across both steps, not a single giant teleport
    return min(step1, step2) >= 4.0 and max(step1, step2) <= 95.0


def _box_area(b: list[float]) -> float:
    return max(0.0, (b[2] - b[0]) * (b[3] - b[1]))


class TemporalAccidentEngine:
    """Multi-frame kinematic + looming/camera-motion evidence fusion."""

    RULE_VERSION = "drishti-accident-v2.2"

    def __init__(
        self,
        camera_id: str,
        source_type: str = "VIDEO REPLAY",
        model_version: str = "YOLOX-Nano",
        config: AccidentConfig | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.source_type = source_type
        self.model_version = model_version
        self.cfg = config or AccidentConfig()
        self._pair_streaks: dict[tuple[str, str], int] = {}
        self._single_streaks: dict[str, int] = {}
        self._looming_streaks: dict[str, int] = {}
        self._cooldowns: dict[str, float] = {}
        self._camera_shifts: list[float] = []
        self.incidents: list[IncidentRecord] = []
        self.current_state: IncidentState = IncidentState.NORMAL

    def reset_for_loop(self) -> None:
        self._pair_streaks.clear()
        self._single_streaks.clear()
        self._looming_streaks.clear()
        self._cooldowns.clear()
        self._camera_shifts.clear()
        self.current_state = IncidentState.NORMAL

    def _in_stop_zone(self, cx: float, cy: float) -> bool:
        return any(_point_in_poly(cx, cy, poly) for poly in self.cfg.stop_zones)

    def _is_crowded_slow_scene(self, tracks: list[TrackState]) -> bool:
        slow_vehicles = 0
        for t in tracks:
            if t.confirmed and t.label in VEHICLE_CLASSES and len(t.trajectory) >= 3:
                if _smoothed_speed(t.trajectory[-3:]) < self.cfg.congestion_speed_px_s:
                    slow_vehicles += 1
        return slow_vehicles >= self.cfg.congestion_vehicle_count

    def _single_track_cues(self, t: TrackState) -> tuple[float, list[str]]:
        if t.missed_frames > 0 or len(t.trajectory) < self.cfg.min_history_frames:
            return 0.0, []
        pts = t.trajectory[-max(6, self.cfg.min_history_frames) :]
        pre_pts = pts[:3]
        post_pts = pts[-3:]
        pre_speed = _smoothed_speed(pre_pts)
        post_speed = _smoothed_speed(post_pts)
        cues: list[str] = []
        score = 0.0

        # Sustained motion (excluding 1-frame tracker ID-jump spikes) followed by abrupt deceleration
        if (
            self.cfg.min_moving_px_s <= pre_speed <= self.cfg.max_sustained_px_s
            and _is_sustained_motion(pre_pts)
        ):
            drop = (pre_speed - post_speed) / max(pre_speed, 1e-6)
            if drop >= self.cfg.decel_drop_ratio and post_speed < 14.0:
                if not self._in_stop_zone(pts[-1].cx, pts[-1].cy):
                    score += 0.30
                    cues.append(
                        f"sudden_deceleration({pre_speed:.1f}->{post_speed:.1f}px/s)"
                    )

        # Abrupt heading change while moving
        if pre_speed >= 25.0 and _is_sustained_motion(pre_pts) and len(pts) >= 4:
            pre_hd = math.degrees(
                math.atan2(pts[2].cy - pts[0].cy, pts[2].cx - pts[0].cx)
            )
            post_hd = math.degrees(
                math.atan2(pts[-1].cy - pts[-3].cy, pts[-1].cx - pts[-3].cx)
            )
            if post_speed >= 15.0:
                hd = _angle_diff_deg(pre_hd, post_hd)
                if hd >= self.cfg.heading_change_deg:
                    score += 0.22
                    cues.append(f"abrupt_heading_change({hd:.0f}deg)")

        # Vehicle rotation / aspect ratio inversion or rollover / rider fall
        b0 = pts[0].xyxy
        b1 = pts[-1].xyxy
        ar0 = (b0[2] - b0[0]) / max(1.0, b0[3] - b0[1])
        ar1 = (b1[2] - b1[0]) / max(1.0, b1[3] - b1[1])
        if ar0 > 0 and (ar1 / ar0 >= 1.65 or ar0 / ar1 >= 1.65) and pre_speed >= 28.0:
            score += 0.28
            cues.append(f"rotation_or_aspect_shift({ar0:.2f}->{ar1:.2f})")

        return min(1.0, score), cues

    def evaluate(
        self,
        tracks: list[TrackState],
        timestamp_s: float,
        frame_area: float = 640.0 * 360.0,
        camera_motion: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Evaluate multi-frame tracks for vehicle collisions, ego-impacts, or rollovers."""
        if camera_motion is not None:
            shift = (
                camera_motion.get("shift_mag", 0.0)
                if camera_motion.get("response", 0.0) >= 0.35
                else 0.0
            )
            self._camera_shifts.append(shift)
            if len(self._camera_shifts) > 15:
                self._camera_shifts = self._camera_shifts[-15:]
        camera_moving = (
            len(self._camera_shifts) >= 3
            and sum(self._camera_shifts[-3:]) / 3.0 >= 5.0
        )

        # Allow missed_frames <= 1 so brief collision occlusion doesn't drop tracks
        confirmed = [
            t
            for t in tracks
            if t.confirmed
            and t.missed_frames <= 1
            and len(t.trajectory) >= self.cfg.min_history_frames
        ]
        crowded = self._is_crowded_slow_scene(confirmed)
        new_events: list[dict[str, Any]] = []
        highest_state = IncidentState.NORMAL

        # 1. Pairwise vehicle-to-vehicle convergence + contact + kinematic/rotation cues
        for i in range(len(confirmed)):
            a = confirmed[i]
            if a.label not in VEHICLE_CLASSES:
                continue
            for j in range(i + 1, len(confirmed)):
                b = confirmed[j]
                if b.label not in VEHICLE_CLASSES:
                    continue

                pa, pb = a.trajectory[-1], b.trajectory[-1]
                pa0, pb0 = a.trajectory[-3], b.trajectory[-3]
                dist_now = math.hypot(pa.cx - pb.cx, pa.cy - pb.cy)
                dist_prev = math.hypot(pa0.cx - pb0.cx, pa0.cy - pb0.cy)
                dt = max(1e-3, pa.timestamp_s - pa0.timestamp_s)
                closing_speed = (dist_prev - dist_now) / dt

                iou = _box_iou(pa.xyxy, pb.xyxy)
                gap = _box_edge_gap(pa.xyxy, pb.xyxy)
                contact = (
                    iou >= self.cfg.contact_iou_threshold
                    or gap <= self.cfg.contact_gap_px
                )

                if not contact:
                    self._pair_streaks.pop((a.track_id, b.track_id), None)
                    continue

                sa, cues_a = self._single_track_cues(a)
                sb, cues_b = self._single_track_cues(b)
                has_decel = any(
                    "sudden_deceleration" in c for c in (cues_a + cues_b)
                )
                kinematic_evidence = has_decel

                if not kinematic_evidence or crowded:
                    self._pair_streaks[(a.track_id, b.track_id)] = 0
                    continue

                cues: list[str] = []
                score = 0.0
                if closing_speed >= self.cfg.min_closing_px_s:
                    score += 0.30
                    cues.append(f"trajectory_convergence({closing_speed:.1f}px/s)")
                score += 0.25
                cues.append(f"proximity_contact(iou={iou:.2f},gap={gap:.1f}px)")
                score += 0.5 * (sa + sb)
                cues.extend(f"{a.track_id}:{c}" for c in cues_a)
                cues.extend(f"{b.track_id}:{c}" for c in cues_b)

                post_a = _smoothed_speed(a.trajectory[-3:])
                post_b = _smoothed_speed(b.trajectory[-3:])
                if has_decel and (post_a < 10.0 or post_b < 10.0):
                    score += 0.20
                    cues.append("unexpected_post_contact_stop")

                score = min(0.95, round(score, 3))
                pair_key = (a.track_id, b.track_id)
                if score >= 0.55 and (
                    closing_speed >= self.cfg.min_closing_px_s
                    or (post_a < 10.0 and post_b < 10.0)
                    or (sa > 0 and sb > 0)
                ):
                    streak = self._pair_streaks.get(pair_key, 0) + 1
                    self._pair_streaks[pair_key] = streak
                else:
                    self._pair_streaks[pair_key] = 0
                    continue

                if streak < self.cfg.confirm_frames:
                    highest_state = IncidentState.POTENTIAL_RISK
                    continue

                if score >= 0.72:
                    state = IncidentState.PROBABLE_ACCIDENT
                    severity = "high"
                else:
                    state = IncidentState.SUSPECTED_COLLISION
                    severity = "medium"

                highest_state = state
                cd_key = f"pair:{a.track_id}:{b.track_id}"
                if timestamp_s - self._cooldowns.get(cd_key, -1e9) >= self.cfg.cooldown_s:
                    self._cooldowns[cd_key] = timestamp_s
                    union_box = [
                        round(min(pa.xyxy[0], pb.xyxy[0]), 2),
                        round(min(pa.xyxy[1], pb.xyxy[1]), 2),
                        round(max(pa.xyxy[2], pb.xyxy[2]), 2),
                        round(max(pa.xyxy[3], pb.xyxy[3]), 2),
                    ]
                    rec = IncidentRecord(
                        event_id=f"inc-{uuid.uuid4().hex[:10]}",
                        camera_id=self.camera_id,
                        source_type=self.source_type,
                        timestamp_s=round(timestamp_s, 3),
                        category="collision_incident",
                        state=state.value,
                        severity=severity,
                        evidence_score=score,
                        evidence_score_note=(
                            "Weighted kinematic evidence score (0-1); "
                            "not a calibrated accident probability."
                        ),
                        model_version=self.model_version,
                        rule_version=self.RULE_VERSION,
                        involved_track_ids=[a.track_id, b.track_id],
                        contributing_evidence=cues,
                        bbox_union=union_box,
                    )
                    self.incidents.append(rec)
                    new_events.append(rec.to_dict())

        # 2. Ego-vehicle / dashcam looming impact (16-frame lookback + rapid expansion + post-impact slowdown)
        for t in confirmed:
            if t.label not in VEHICLE_CLASSES or len(t.trajectory) < 7:
                continue
            pts = t.trajectory[-16:]
            early_count = max(3, min(6, len(pts) // 2))
            early_areas = sorted(_box_area(p.xyxy) for p in pts[:early_count])
            a0 = float(early_areas[len(early_areas) // 2])
            a_now = _box_area(pts[-1].xyxy)
            area_frac = a_now / max(frame_area, 1.0)
            a0_frac = a0 / max(frame_area, 1.0)
            ratio = a_now / max(a0, 1.0)
            post_speed = _smoothed_speed(pts[-3:])
            recent_areas = [_box_area(p.xyxy) for p in pts[-max(3, len(pts) // 3):]]
            sustained_growth = sum(
                recent_areas[i] > recent_areas[i - 1] * 1.02
                for i in range(1, len(recent_areas))
            )
            moving_camera_suppressed = camera_moving and area_frac > 0.60
            if (
                a0_frac >= 0.06
                and ratio >= self.cfg.looming_area_ratio
                and 0.40 <= area_frac <= 0.85
                and post_speed <= 58.0
                and sustained_growth >= 1
                and not moving_camera_suppressed
            ):
                streak = self._looming_streaks.get(t.track_id, 0) + 1
                self._looming_streaks[t.track_id] = streak
                if streak >= self.cfg.confirm_frames:
                    cd_key = f"looming:{t.track_id}"
                    if timestamp_s - self._cooldowns.get(cd_key, -1e9) >= self.cfg.cooldown_s:
                        self._cooldowns[cd_key] = timestamp_s
                        cues = [
                            f"rapid_looming_expansion({ratio:.2f}x)",
                            f"frame_occupancy({area_frac * 100:.1f}%)",
                            f"post_approach_speed({post_speed:.1f}px/s)",
                        ]
                        if camera_moving:
                            cues.append("moving_camera_ego_motion")
                        score = round(min(0.92, 0.45 + min(0.3, (ratio - 1.5) * 0.15) + min(0.2, area_frac * 0.4)), 3)
                        state = (
                            IncidentState.PROBABLE_ACCIDENT
                            if score >= 0.72
                            else IncidentState.SUSPECTED_COLLISION
                        )
                        rec = IncidentRecord(
                            event_id=f"inc-{uuid.uuid4().hex[:10]}",
                            camera_id=self.camera_id,
                            source_type=self.source_type,
                            timestamp_s=round(timestamp_s, 3),
                            category="ego_or_looming_collision",
                            state=state.value,
                            severity="high" if state == IncidentState.PROBABLE_ACCIDENT else "medium",
                            evidence_score=score,
                            evidence_score_note=(
                                "Weighted looming + kinematic evidence score (0-1); "
                                "not a calibrated accident probability."
                            ),
                            model_version=self.model_version,
                            rule_version=self.RULE_VERSION,
                            involved_track_ids=[t.track_id],
                            contributing_evidence=cues,
                            bbox_union=t.last_box,
                        )
                        self.incidents.append(rec)
                        new_events.append(rec.to_dict())
                        highest_state = state
            else:
                self._looming_streaks.pop(t.track_id, None)

        # 3. Single-vehicle rollover or rider fall (requires both aspect shift + decel)
        for t in confirmed:
            s_score, s_cues = self._single_track_cues(t)
            has_rollover = any("rotation_or_aspect_shift" in c for c in s_cues)
            has_decel = any("sudden_deceleration" in c for c in s_cues)
            if has_rollover and has_decel and not crowded:
                streak = self._single_streaks.get(t.track_id, 0) + 1
                self._single_streaks[t.track_id] = streak
                if streak >= self.cfg.confirm_frames:
                    cd_key = f"single:{t.track_id}"
                    if timestamp_s - self._cooldowns.get(cd_key, -1e9) >= self.cfg.cooldown_s:
                        self._cooldowns[cd_key] = timestamp_s
                        rec = IncidentRecord(
                            event_id=f"inc-{uuid.uuid4().hex[:10]}",
                            camera_id=self.camera_id,
                            source_type=self.source_type,
                            timestamp_s=round(timestamp_s, 3),
                            category="fall_or_rollover",
                            state=IncidentState.SUSPECTED_COLLISION.value,
                            severity="medium",
                            evidence_score=round(min(0.9, s_score), 3),
                            evidence_score_note=(
                                "Weighted kinematic evidence score (0-1); "
                                "not a calibrated accident probability."
                            ),
                            model_version=self.model_version,
                            rule_version=self.RULE_VERSION,
                            involved_track_ids=[t.track_id],
                            contributing_evidence=s_cues,
                            bbox_union=t.last_box,
                        )
                        self.incidents.append(rec)
                        new_events.append(rec.to_dict())
                        highest_state = IncidentState.SUSPECTED_COLLISION
            else:
                self._single_streaks.pop(t.track_id, None)

        self.current_state = highest_state
        return {
            "state": highest_state.value,
            "camera_moving": camera_moving,
            "crowded_suppression_active": crowded,
            "new_incidents": new_events,
            "total_incidents": len(self.incidents),
        }
