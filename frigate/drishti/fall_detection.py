"""DRISHTI AI V2 — Temporal Human Fall Detection Engine (Non-Diagnostic)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FallState(str, Enum):
    NORMAL = "NORMAL"
    POTENTIAL_FALL = "POTENTIAL_FALL"
    SUSPECTED_FALL = "SUSPECTED_FALL"
    PROLONGED_INACTIVITY = "PROLONGED_INACTIVITY"
    SOS_PENDING = "SOS_PENDING"
    SOS_ACTIVATED = "SOS_ACTIVATED"
    RECOVERED = "RECOVERED"
    HUMAN_CONFIRMED_EMERGENCY = "HUMAN_CONFIRMED_EMERGENCY"
    DISMISSED = "DISMISSED"


@dataclass
class PoseObservation:
    timestamp_s: float
    posture: str
    torso_angle_deg: float | None
    hip_y_norm: float | None
    reliability: str
    cx: float
    cy: float


@dataclass
class TrackFallState:
    state: FallState = FallState.NORMAL
    history: deque[PoseObservation] = field(default_factory=lambda: deque(maxlen=40))
    horizontal_since_ts: float | None = None
    last_motion_ts: float = 0.0
    fall_streak: int = 0
    inactivity_s: float = 0.0
    last_evidence: list[str] = field(default_factory=list)


class TemporalFallEngine:
    """Multi-frame pose-transition & inactivity fall detector.

    Distinguishes sudden falls from sitting, bending, walking, or intentional
    lying down. Horizontal posture alone never triggers a confirmed fall.
    Never claims medical diagnoses.
    """

    RULE_VERSION = "drishti-fall-v2.0"

    def __init__(
        self,
        rapid_drop_norm_per_s: float = 0.22,
        horizontal_angle_deg: float = 55.0,
        inactivity_alert_s: float = 8.0,
        sos_trigger_inactivity_s: float = 5.0,
        confirm_frames: int = 2,
    ) -> None:
        self.rapid_drop_norm_per_s = rapid_drop_norm_per_s
        self.horizontal_angle_deg = horizontal_angle_deg
        self.inactivity_alert_s = inactivity_alert_s
        self.sos_trigger_inactivity_s = sos_trigger_inactivity_s
        self.confirm_frames = confirm_frames
        self.tracks: dict[str, TrackFallState] = {}

    def reset(self) -> None:
        self.tracks.clear()

    def update(
        self,
        track_id: str,
        timestamp_s: float,
        pose: dict[str, Any],
        cx: float,
        cy: float,
    ) -> dict[str, Any]:
        st = self.tracks.get(track_id)
        if st is None:
            st = TrackFallState(last_motion_ts=timestamp_s)
            self.tracks[track_id] = st

        obs = PoseObservation(
            timestamp_s=timestamp_s,
            posture=pose.get("posture", "insufficient_evidence"),
            torso_angle_deg=pose.get("torso_angle_deg"),
            hip_y_norm=pose.get("hip_y_norm"),
            reliability=pose.get("reliability", "insufficient_evidence"),
            cx=cx,
            cy=cy,
        )

        # Motion check vs previous observation
        moved = False
        if st.history:
            prev = st.history[-1]
            disp = ((cx - prev.cx) ** 2 + (cy - prev.cy) ** 2) ** 0.5
            if disp >= 6.0:
                moved = True
                st.last_motion_ts = timestamp_s
        else:
            st.last_motion_ts = timestamp_s

        st.history.append(obs)
        st.inactivity_s = round(max(0.0, timestamp_s - st.last_motion_ts), 2)

        if obs.reliability == "insufficient_evidence":
            return {
                "track_id": track_id,
                "fall_state": st.state.value,
                "posture": obs.posture,
                "inactivity_s": st.inactivity_s,
                "reliability": obs.reliability,
                "evidence": ["insufficient_pose_visibility"],
            }

        # Check recovery if person returns upright
        if st.state in (
            FallState.SUSPECTED_FALL,
            FallState.POTENTIAL_FALL,
            FallState.PROLONGED_INACTIVITY,
            FallState.SOS_PENDING,
        ):
            if obs.posture == "standing" and (obs.torso_angle_deg or 0) < 28.0:
                st.state = FallState.RECOVERED
                st.fall_streak = 0
                st.horizontal_since_ts = None
                st.last_evidence = ["returned_to_upright_posture"]
                return {
                    "track_id": track_id,
                    "fall_state": st.state.value,
                    "posture": obs.posture,
                    "inactivity_s": st.inactivity_s,
                    "reliability": obs.reliability,
                    "evidence": st.last_evidence,
                }

        is_horizontal = (
            obs.posture == "lying_horizontal"
            or (obs.torso_angle_deg is not None and obs.torso_angle_deg >= self.horizontal_angle_deg)
        )
        if is_horizontal:
            if st.horizontal_since_ts is None:
                st.horizontal_since_ts = timestamp_s
        else:
            st.horizontal_since_ts = None

        # Look back 0.4s - 2.5s for an upright -> horizontal transition + downward hip drop
        evidence: list[str] = []
        rapid_transition = False
        if is_horizontal and len(st.history) >= 3 and obs.hip_y_norm is not None:
            for prev in st.history:
                dt = timestamp_s - prev.timestamp_s
                if 0.3 <= dt <= 2.5 and prev.hip_y_norm is not None:
                    drop_rate = (obs.hip_y_norm - prev.hip_y_norm) / dt
                    was_upright = prev.posture in ("standing", "walking") or (
                        prev.torso_angle_deg is not None and prev.torso_angle_deg < 30.0
                    )
                    if was_upright and drop_rate >= self.rapid_drop_norm_per_s:
                        rapid_transition = True
                        evidence.append(
                            f"rapid_posture_transition({prev.posture}->horizontal)"
                        )
                        evidence.append(f"downward_hip_rate={drop_rate:.2f}/s")
                        if obs.torso_angle_deg is not None:
                            evidence.append(f"torso_angle={obs.torso_angle_deg:.1f}deg")
                        break

        if rapid_transition:
            st.fall_streak += 1
        elif not is_horizontal:
            st.fall_streak = 0

        if st.fall_streak >= self.confirm_frames and is_horizontal:
            if st.inactivity_s >= self.sos_trigger_inactivity_s and not moved:
                st.state = FallState.SOS_PENDING
                evidence.append(f"post_fall_inactivity={st.inactivity_s:.1f}s")
            else:
                st.state = FallState.SUSPECTED_FALL
        elif st.fall_streak == 1 and is_horizontal:
            st.state = FallState.POTENTIAL_FALL
        elif st.inactivity_s >= self.inactivity_alert_s and obs.posture in (
            "lying_horizontal",
            "sitting",
        ):
            st.state = FallState.PROLONGED_INACTIVITY
            evidence.append(f"prolonged_inactivity={st.inactivity_s:.1f}s")
        elif st.state == FallState.RECOVERED and not is_horizontal:
            st.state = FallState.NORMAL

        st.last_evidence = evidence
        return {
            "track_id": track_id,
            "fall_state": st.state.value,
            "posture": obs.posture,
            "torso_angle_deg": obs.torso_angle_deg,
            "inactivity_s": st.inactivity_s,
            "reliability": obs.reliability,
            "evidence": evidence,
        }

