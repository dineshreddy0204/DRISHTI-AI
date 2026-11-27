"""DRISHTI AI V2 — Centralized SQLite Incident Store, Risk Levels & Emergency SOS Engine."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

from frigate.const import BASE_DIR

DB_PATH = Path(BASE_DIR) / "drishti_v2.db"
EVIDENCE_DIR = Path(BASE_DIR) / "drishti-evidence"


class RiskLevel(str, Enum):
    NORMAL = "NORMAL"
    LOW = "LOW"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class NotificationStatus(str, Enum):
    NOT_CONFIGURED = "Not configured"
    PENDING = "Pending"
    SENDING = "Sending"
    SENT = "Sent"
    DELIVERED = "Delivered"
    FAILED = "Failed"
    TEST_ONLY = "Test only"


def is_test_mode() -> bool:
    """TEST MODE is enabled by default and enforced at the backend notification layer."""
    return os.environ.get("DRISHTI_TEST_MODE", "true").lower() != "false"


class IncidentSOSStore:
    """Thread-safe SQLite store for multi-camera incidents, SOS state, and emergency contacts."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS cameras_v2 (
                    camera_id TEXT PRIMARY KEY,
                    camera_number INTEGER NOT NULL,
                    camera_name TEXT NOT NULL,
                    location TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    stream_source TEXT NOT NULL,
                    enabled_modules TEXT NOT NULL,
                    zones TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS incidents (
                    incident_id TEXT PRIMARY KEY,
                    camera_id TEXT NOT NULL,
                    camera_number INTEGER NOT NULL,
                    camera_name TEXT NOT NULL,
                    camera_location TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    contributing_evidence TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    rule_version TEXT NOT NULL,
                    evidence_reliability TEXT NOT NULL,
                    track_ids TEXT NOT NULL,
                    snapshot_path TEXT,
                    pre_clip_path TEXT,
                    post_clip_path TEXT,
                    review_status TEXT NOT NULL DEFAULT 'unreviewed',
                    sos_status TEXT NOT NULL DEFAULT 'NONE',
                    notification_status TEXT NOT NULL,
                    is_test_mode INTEGER NOT NULL DEFAULT 1,
                    is_historical INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS sos_sessions (
                    sos_id TEXT PRIMARY KEY,
                    incident_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    countdown_seconds INTEGER NOT NULL,
                    started_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    test_mode INTEGER NOT NULL DEFAULT 1,
                    notification_status TEXT NOT NULL,
                    resolved_at REAL,
                    resolution_action TEXT
                );
                CREATE TABLE IF NOT EXISTS emergency_contacts (
                    contact_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    target_masked TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL
                );
                """
            )
            # Seed CAM-001 (Laptop Webcam) and CAM-002 (Hyderabad Replay) if absent
            conn.execute(
                """
                INSERT OR IGNORE INTO cameras_v2
                (camera_id, camera_number, camera_name, location, source_type, stream_source, enabled_modules, zones)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "CAM-001",
                    1,
                    "Laptop Webcam",
                    "Local Test Environment",
                    "LIVE WEBCAM",
                    "local://webcam/0",
                    json.dumps(["human_detection", "pose_estimation", "fall_detection", "sos"]),
                    json.dumps([]),
                ),
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO cameras_v2
                (camera_id, camera_number, camera_name, location, source_type, stream_source, enabled_modules, zones)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "hyderabad_replay",
                    2,
                    "Hyderabad Replay",
                    "Demo Traffic Corridor",
                    "VIDEO REPLAY",
                    "/opt/drishti/demo/hyderabad.mp4",
                    json.dumps(["object_detection", "tracking", "accident_detection", "collision_risk"]),
                    json.dumps([]),
                ),
            )
            conn.commit()

    def list_cameras(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM cameras_v2 ORDER BY camera_number ASC").fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["enabled_modules"] = json.loads(d["enabled_modules"])
                d["zones"] = json.loads(d["zones"])
                out.append(d)
            return out

    def get_camera(self, camera_id: str) -> dict[str, Any]:
        for c in self.list_cameras():
            if c["camera_id"] == camera_id:
                return c
        return {
            "camera_id": camera_id,
            "camera_number": 99,
            "camera_name": camera_id,
            "location": "Unassigned",
            "source_type": "UPLOADED VIDEO",
            "stream_source": "",
            "enabled_modules": [],
            "zones": [],
        }

    def save_snapshot(self, incident_id: str, jpeg_bytes: bytes) -> str:
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        path = EVIDENCE_DIR / f"{incident_id}.jpg"
        path.write_bytes(jpeg_bytes)
        return str(path)

    def create_incident(
        self,
        camera_id: str,
        event_type: str,
        risk_level: RiskLevel,
        contributing_evidence: list[str],
        model_version: str,
        rule_version: str,
        evidence_reliability: str,
        track_ids: list[str],
        source_type: str | None = None,
        snapshot_bytes: bytes | None = None,
        sos_status: str = "NONE",
        is_historical: bool = False,
        dedup_window_s: float = 12.0,
    ) -> dict[str, Any]:
        cam = self.get_camera(camera_id)
        now = time.time()
        test_flag = is_test_mode() or is_historical
        notif_status = (
            NotificationStatus.TEST_ONLY.value
            if test_flag
            else NotificationStatus.NOT_CONFIGURED.value
        )

        with self._lock, self._connect() as conn:
            # Deduplicate identical recent event on same camera
            recent = conn.execute(
                """
                SELECT * FROM incidents
                WHERE camera_id = ? AND event_type = ? AND timestamp >= ?
                ORDER BY timestamp DESC LIMIT 1
                """,
                (camera_id, event_type, now - dedup_window_s),
            ).fetchone()
            if recent is not None:
                d = dict(recent)
                d["contributing_evidence"] = json.loads(d["contributing_evidence"])
                d["track_ids"] = json.loads(d["track_ids"])
                d["deduplicated"] = True
                return d

            inc_id = f"INC-{uuid.uuid4().hex[:10].upper()}"
            snap_path = self.save_snapshot(inc_id, snapshot_bytes) if snapshot_bytes else None
            conn.execute(
                """
                INSERT INTO incidents (
                    incident_id, camera_id, camera_number, camera_name, camera_location,
                    source_type, timestamp, event_type, risk_level, contributing_evidence,
                    model_version, rule_version, evidence_reliability, track_ids,
                    snapshot_path, review_status, sos_status, notification_status,
                    is_test_mode, is_historical
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    inc_id,
                    camera_id,
                    int(cam["camera_number"]),
                    cam["camera_name"],
                    cam["location"],
                    source_type or cam["source_type"],
                    now,
                    event_type,
                    risk_level.value,
                    json.dumps(contributing_evidence),
                    model_version,
                    rule_version,
                    evidence_reliability,
                    json.dumps(track_ids),
                    snap_path,
                    "unreviewed",
                    sos_status,
                    notif_status,
                    1 if test_flag else 0,
                    1 if is_historical else 0,
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM incidents WHERE incident_id = ?", (inc_id,)
            ).fetchone()
            d = dict(row)
            d["contributing_evidence"] = json.loads(d["contributing_evidence"])
            d["track_ids"] = json.loads(d["track_ids"])
            d["deduplicated"] = False
            return d

    def list_incidents(
        self,
        camera_id: str | None = None,
        risk_level: str | None = None,
        review_status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM incidents WHERE 1=1"
        params: list[Any] = []
        if camera_id:
            query += " AND camera_id = ?"
            params.append(camera_id)
        if risk_level:
            query += " AND risk_level = ?"
            params.append(risk_level)
        if review_status:
            query += " AND review_status = ?"
            params.append(review_status)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        with self._lock, self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["contributing_evidence"] = json.loads(d["contributing_evidence"])
                d["track_ids"] = json.loads(d["track_ids"])
                out.append(d)
            return out

    def update_incident_review(self, incident_id: str, status: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE incidents SET review_status = ? WHERE incident_id = ?",
                (status, incident_id),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM incidents WHERE incident_id = ?", (incident_id,)
            ).fetchone()
            if not row:
                return None
            d = dict(row)
            d["contributing_evidence"] = json.loads(d["contributing_evidence"])
            d["track_ids"] = json.loads(d["track_ids"])
            return d

    def trigger_sos(
        self,
        camera_id: str,
        event_type: str,
        evidence: list[str],
        countdown_seconds: int = 15,
        track_ids: list[str] | None = None,
        snapshot_bytes: bytes | None = None,
        manual: bool = False,
    ) -> dict[str, Any]:
        test_flag = is_test_mode()
        incident = self.create_incident(
            camera_id=camera_id,
            event_type=event_type,
            risk_level=RiskLevel.CRITICAL,
            contributing_evidence=evidence,
            model_version="MoveNet-Lightning + YOLOX-Nano",
            rule_version="drishti-sos-v2.0",
            evidence_reliability="manual_trigger" if manual else "temporal_pose_fusion",
            track_ids=track_ids or [],
            snapshot_bytes=snapshot_bytes,
            sos_status="SOS_PENDING",
            is_historical=False,
            dedup_window_s=0.0 if manual else 15.0,
        )
        now = time.time()
        sos_id = f"SOS-{uuid.uuid4().hex[:8].upper()}"
        notif_status = (
            NotificationStatus.TEST_ONLY.value
            if test_flag
            else NotificationStatus.NOT_CONFIGURED.value
        )
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM sos_sessions WHERE state IN ('COUNTDOWN', 'ACTIVATED') ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            if existing is not None and not manual:
                return dict(existing)
            conn.execute(
                """
                INSERT INTO sos_sessions (
                    sos_id, incident_id, camera_id, state, countdown_seconds,
                    started_at, expires_at, test_mode, notification_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sos_id,
                    incident["incident_id"],
                    camera_id,
                    "COUNTDOWN",
                    countdown_seconds,
                    now,
                    now + countdown_seconds,
                    1 if test_flag else 0,
                    notif_status,
                ),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM sos_sessions WHERE sos_id = ?", (sos_id,)).fetchone()
            return dict(row)

    def get_active_sos(self) -> dict[str, Any] | None:
        now = time.time()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sos_sessions WHERE state IN ('COUNTDOWN', 'ACTIVATED') ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            d = dict(row)
            if d["state"] == "COUNTDOWN" and now >= d["expires_at"]:
                # Transition to ACTIVATED locally; external dispatch stays blocked in TEST MODE
                notif = (
                    NotificationStatus.TEST_ONLY.value
                    if d["test_mode"]
                    else NotificationStatus.NOT_CONFIGURED.value
                )
                conn.execute(
                    "UPDATE sos_sessions SET state = 'ACTIVATED', notification_status = ? WHERE sos_id = ?",
                    (notif, d["sos_id"]),
                )
                conn.execute(
                    "UPDATE incidents SET sos_status = 'SOS_ACTIVATED', notification_status = ? WHERE incident_id = ?",
                    (notif, d["incident_id"]),
                )
                conn.commit()
                d["state"] = "ACTIVATED"
                d["notification_status"] = notif
            d["remaining_s"] = max(0, round(d["expires_at"] - now, 1)) if d["state"] == "COUNTDOWN" else 0
            return d

    def respond_sos(self, sos_id: str, action: str) -> dict[str, Any] | None:
        state_map = {
            "i_am_okay": ("RECOVERED", "RECOVERED"),
            "cancel": ("CANCELLED", "DISMISSED"),
            "acknowledge": ("ACKNOWLEDGED", "ACKNOWLEDGED"),
            "confirm_emergency": ("HUMAN_CONFIRMED", "HUMAN_CONFIRMED_EMERGENCY"),
        }
        if action not in state_map:
            raise ValueError(f"Unsupported SOS action: {action}")
        sos_state, inc_review = state_map[action]
        now = time.time()
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM sos_sessions WHERE sos_id = ?", (sos_id,)).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE sos_sessions SET state = ?, resolved_at = ?, resolution_action = ? WHERE sos_id = ?",
                (sos_state, now, action, sos_id),
            )
            conn.execute(
                "UPDATE incidents SET sos_status = ?, review_status = ? WHERE incident_id = ?",
                (sos_state, inc_review, row["incident_id"]),
            )
            conn.commit()
            updated = conn.execute("SELECT * FROM sos_sessions WHERE sos_id = ?", (sos_id,)).fetchone()
            return dict(updated)

    def list_contacts(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM emergency_contacts ORDER BY created_at DESC").fetchall()
            return [dict(r) for r in rows]

    def add_contact(self, name: str, channel: str, target: str) -> dict[str, Any]:
        cid = f"CNT-{uuid.uuid4().hex[:8].upper()}"
        masked = target[:2] + "***" + target[-2:] if len(target) > 4 else "***"
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO emergency_contacts (contact_id, name, channel, target_masked, enabled, created_at) VALUES (?, ?, ?, ?, 1, ?)",
                (cid, name, channel, masked, now),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM emergency_contacts WHERE contact_id = ?", (cid,)).fetchone()
            return dict(row)

