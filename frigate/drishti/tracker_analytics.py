"""DRISHTI AI V2 — Temporal tracking, duplicate suppression, and separated metrics."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from norfair import Detection, Tracker


def _box_iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    denom = area_a + area_b - inter
    return inter / denom if denom > 0.0 else 0.0


def _normalized_box_distance(det: Detection, tracked_obj: Any) -> float:
    est = tracked_obj.estimate
    pts = det.points
    est_w = max(1.0, float(est[1][0] - est[0][0]))
    est_h = max(1.0, float(est[1][1] - est[0][1]))
    det_w = max(1.0, float(pts[1][0] - pts[0][0]))
    det_h = max(1.0, float(pts[1][1] - pts[0][1]))

    est_cx = (est[0][0] + est[1][0]) * 0.5
    est_cy = float(est[1][1])
    det_cx = (pts[0][0] + pts[1][0]) * 0.5
    det_cy = float(pts[1][1])

    dx = (det_cx - est_cx) / est_w
    dy = (det_cy - est_cy) / est_h
    w_ratio = max(est_w, det_w) / min(est_w, det_w) - 1.0
    h_ratio = max(est_h, det_h) / min(est_h, det_h) - 1.0
    return float(math.sqrt(dx * dx + dy * dy + w_ratio * w_ratio + h_ratio * h_ratio))


@dataclass
class TrajectoryPoint:
    timestamp_s: float
    xyxy: list[float]
    cx: float
    cy: float
    vx_px_s: float
    vy_px_s: float
    speed_px_s: float
    heading_deg: float
    score: float


@dataclass
class TrackState:
    track_id: str
    raw_norfair_id: int
    label: str
    confirmed: bool = False
    hit_count: int = 0
    missed_frames: int = 0
    is_stationary: bool = False
    last_seen_ts: float = 0.0
    first_seen_ts: float = 0.0
    last_box: list[float] = field(default_factory=list)
    last_score: float = 0.0
    class_votes: list[tuple[str, float]] = field(default_factory=list)
    trajectory: list[TrajectoryPoint] = field(default_factory=list)
    entered: bool = False
    exited: bool = False


class TrackAnalyticsEngine:
    """Per-camera tracker with temporal class smoothing, duplicate suppression, and replay isolation."""

    def __init__(
        self,
        camera_id: str,
        fps: float = 10.0,
        distance_threshold: float = 2.5,
        min_confirmed_frames: int = 2,
        max_disappeared_frames: int = 12,
        duplicate_iou_threshold: float = 0.65,
        stationary_px_threshold: float = 6.0,
        trajectory_maxlen: int = 90,
    ) -> None:
        self.camera_id = camera_id
        self.fps = max(0.5, fps)
        self.distance_threshold = distance_threshold
        self.min_confirmed_frames = max(1, min_confirmed_frames)
        self.max_disappeared_frames = max(2, max_disappeared_frames)
        self.duplicate_iou_threshold = duplicate_iou_threshold
        self.stationary_px_threshold = stationary_px_threshold
        self.trajectory_maxlen = trajectory_maxlen

        self.loop_index: int = 0
        self._init_tracker()

    def _init_tracker(self) -> None:
        self.tracker = Tracker(
            distance_function=_normalized_box_distance,
            distance_threshold=self.distance_threshold,
            initialization_delay=0,
            hit_counter_max=self.max_disappeared_frames,
        )
        self.tracks: dict[int, TrackState] = {}
        self.unique_confirmed_ids: dict[str, str] = {}
        self.per_frame_detection_count: int = 0
        self.entry_count: int = 0
        self.exit_count: int = 0
        self.id_switches: int = 0

    def reset_for_loop(self, loop_index: int | None = None) -> None:
        """Isolate analytics across prerecorded replay loop boundaries."""
        self.loop_index = (self.loop_index + 1) if loop_index is None else loop_index
        self._init_tracker()

    def _smooth_label(
        self, votes: list[tuple[str, float]], current_label: str
    ) -> str:
        weights: dict[str, float] = defaultdict(float)
        recent_votes = votes[-9:]
        for lbl, sc in recent_votes:
            weights[lbl] += max(0.05, sc)
        if not weights:
            return "unknown_vehicle"
        winner, _ = max(weights.items(), key=lambda kv: kv[1])
        winner_count = sum(1 for lbl, _ in recent_votes if lbl == winner)
        if winner_count < 3:
            current_count = sum(
                1 for lbl, _ in recent_votes if lbl == current_label
            )
            if current_count >= 2:
                return current_label
            return winner
        return winner

    def _suppress_duplicate_detections(
        self, detections: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        ordered = sorted(detections, key=lambda d: d["score"], reverse=True)
        kept: list[dict[str, Any]] = []
        for det in ordered:
            if any(
                _box_iou(det["xyxy"], prev["xyxy"]) >= self.duplicate_iou_threshold
                for prev in kept
            ):
                continue
            kept.append(det)
        return kept

    def update(
        self, detections: list[dict[str, Any]], timestamp_s: float
    ) -> dict[str, Any]:
        """Update tracks for one frame and return separated live/unique/trajectory metrics."""
        deduped = self._suppress_duplicate_detections(detections)
        self.per_frame_detection_count += len(deduped)

        norfair_dets = [
            Detection(
                points=np.array(
                    [
                        [d["xyxy"][0], d["xyxy"][1]],
                        [d["xyxy"][2], d["xyxy"][3]],
                    ],
                    dtype=np.float32,
                ),
                scores=np.array([d["score"], d["score"]], dtype=np.float32),
                data=d,
            )
            for d in deduped
        ]

        tracked = self.tracker.update(detections=norfair_dets)
        active_raw_ids: set[int] = set()
        visible_tracks: list[dict[str, Any]] = []

        for tobj in tracked:
            raw_id = int(tobj.id)
            active_raw_ids.add(raw_id)
            det_data = (
                tobj.last_detection.data
                if tobj.last_detection is not None and tobj.last_detection.data
                else {}
            )
            box = (
                det_data.get("xyxy")
                or [
                    round(float(tobj.estimate[0][0]), 2),
                    round(float(tobj.estimate[0][1]), 2),
                    round(float(tobj.estimate[1][0]), 2),
                    round(float(tobj.estimate[1][1]), 2),
                ]
            )
            score = float(det_data.get("score", 0.0))
            raw_label = str(det_data.get("label", "unknown_vehicle"))

            state = self.tracks.get(raw_id)
            if state is None:
                anon_id = f"{self.camera_id}-L{self.loop_index}-T{raw_id:04d}"
                state = TrackState(
                    track_id=anon_id,
                    raw_norfair_id=raw_id,
                    label=raw_label,
                    first_seen_ts=timestamp_s,
                    last_seen_ts=timestamp_s,
                    last_box=list(box),
                    last_score=score,
                )
                self.tracks[raw_id] = state

            was_visible = tobj.live_points.any()
            if was_visible and det_data:
                state.hit_count += 1
                state.missed_frames = 0
                state.class_votes.append((raw_label, score))
                state.label = self._smooth_label(state.class_votes, state.label)
                state.last_score = score
            else:
                state.missed_frames += 1

            cx = (box[0] + box[2]) * 0.5
            cy = (box[1] + box[3]) * 0.5
            vx, vy, speed, heading = 0.0, 0.0, 0.0, 0.0
            if state.trajectory:
                prev = state.trajectory[-1]
                dt = max(1e-3, timestamp_s - prev.timestamp_s)
                vx = (cx - prev.cx) / dt
                vy = (cy - prev.cy) / dt
                speed = math.hypot(vx, vy)
                if speed > 1.0:
                    heading = math.degrees(math.atan2(vy, vx))

            pt = TrajectoryPoint(
                timestamp_s=round(timestamp_s, 3),
                xyxy=[round(float(v), 2) for v in box],
                cx=round(cx, 2),
                cy=round(cy, 2),
                vx_px_s=round(vx, 2),
                vy_px_s=round(vy, 2),
                speed_px_s=round(speed, 2),
                heading_deg=round(heading, 2),
                score=round(score, 4),
            )
            if was_visible:
                state.trajectory.append(pt)
                if len(state.trajectory) > self.trajectory_maxlen:
                    state.trajectory = state.trajectory[-self.trajectory_maxlen :]
                state.last_seen_ts = timestamp_s
                state.last_box = pt.xyxy

                if len(state.trajectory) >= 4:
                    recent = state.trajectory[-4:]
                    disp = math.hypot(
                        recent[-1].cx - recent[0].cx, recent[-1].cy - recent[0].cy
                    )
                    state.is_stationary = disp < self.stationary_px_threshold

            if state.hit_count >= self.min_confirmed_frames:
                if not state.confirmed:
                    state.confirmed = True
                    state.entered = True
                    self.entry_count += 1
                self.unique_confirmed_ids[state.track_id] = state.label

            if state.confirmed and state.missed_frames <= 2:
                visible_tracks.append(
                    {
                        "track_id": state.track_id,
                        "label": state.label,
                        "score": round(state.last_score, 4),
                        "xyxy": state.last_box,
                        "speed_px_s": pt.speed_px_s,
                        "heading_deg": pt.heading_deg,
                        "is_stationary": state.is_stationary,
                        "hit_count": state.hit_count,
                        "missed_frames": state.missed_frames,
                        "trajectory": [
                            [p.cx, p.cy] for p in state.trajectory[-15:]
                        ],
                    }
                )

        # Suppress overlapping active tracks (duplicate suppression across track IDs)
        visible_tracks.sort(key=lambda t: (t["hit_count"], t["score"]), reverse=True)
        deduped_visible: list[dict[str, Any]] = []
        for trk in visible_tracks:
            if any(
                _box_iou(trk["xyxy"], kept["xyxy"]) >= self.duplicate_iou_threshold
                for kept in deduped_visible
            ):
                continue
            deduped_visible.append(trk)

        # Expire vanished tracks and record exits
        for raw_id in list(self.tracks.keys()):
            if raw_id not in active_raw_ids:
                st = self.tracks[raw_id]
                if st.confirmed and not st.exited:
                    st.exited = True
                    self.exit_count += 1
                del self.tracks[raw_id]

        unique_by_label = dict(Counter(self.unique_confirmed_ids.values()))
        visible_by_label = dict(Counter(t["label"] for t in deduped_visible))

        return {
            "timestamp_s": round(timestamp_s, 3),
            "loop_index": self.loop_index,
            "current_visible_objects": len(deduped_visible),
            "visible_by_label": visible_by_label,
            "per_frame_detections": len(deduped),
            "cumulative_frame_detections": self.per_frame_detection_count,
            "unique_tracks_total": len(self.unique_confirmed_ids),
            "unique_tracks_by_label": unique_by_label,
            "entry_count": self.entry_count,
            "exit_count": self.exit_count,
            "tracks": deduped_visible,
        }
