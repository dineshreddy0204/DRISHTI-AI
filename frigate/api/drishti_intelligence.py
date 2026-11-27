"""DRISHTI AI V2 — Traffic Intelligence, Camera Health, Calibration & Diagnostics API."""

from __future__ import annotations

import json
import os
import platform
import threading
import time
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import onnxruntime as ort
import psutil
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from frigate.api.auth import require_role
from frigate.api.defs.tags import Tags
from frigate.api.drishti_video import ROOT as ANALYSIS_ROOT
from frigate.const import BASE_DIR
from frigate.drishti.camera_health import (
    classify_camera_source,
    load_calibrations,
    redact_stream_url,
    save_camera_calibration,
    test_stream_connection,
)
from frigate.drishti.detector import UnifiedDetector
from frigate.drishti.fall_detection import FallState, TemporalFallEngine
from frigate.drishti.model_registry import list_models
from frigate.drishti.pose_estimator import MoveNetPoseEstimator
from frigate.drishti.public_safety import PublicSafetyEngine
from frigate.drishti.safety_analytics import SafetyAnalyticsEngine, SafetyConfig
from frigate.drishti.sos_engine import IncidentSOSStore, RiskLevel, is_test_mode
from frigate.drishti.tracker_analytics import TrackAnalyticsEngine

router = APIRouter(tags=[Tags.media])

ASSETS_DIR = Path("/opt/drishti/demo")
VALIDATION_DIR = Path("/opt/drishti/validation")
REVIEW_OVERRIDES_PATH = Path(BASE_DIR) / "drishti_incident_reviews.json"
AUDIT_LOG_PATH = Path(BASE_DIR) / "drishti_audit.jsonl"

_STORE = IncidentSOSStore()
_LIVE_LOCK = threading.Lock()
_SHARED_DETECTOR: UnifiedDetector | None = None
_SHARED_POSE: MoveNetPoseEstimator | None = None
_LIVE_TRACKER = TrackAnalyticsEngine(camera_id="CAM-001", fps=6.0, min_confirmed_frames=2)
_LIVE_FALL = TemporalFallEngine()
_LIVE_PUBLIC_SAFETY = PublicSafetyEngine(camera_id="CAM-001")
_LIVE_START_TS = time.monotonic()


def _get_live_models() -> tuple[UnifiedDetector, MoveNetPoseEstimator]:
    global _SHARED_DETECTOR, _SHARED_POSE
    if _SHARED_DETECTOR is None:
        _SHARED_DETECTOR = UnifiedDetector(
            assets_dir=ASSETS_DIR,
            model_id="yolox-nano-416",
            score_threshold=0.4,
            intra_op_threads=2,
        )
    if _SHARED_POSE is None:
        _SHARED_POSE = MoveNetPoseEstimator(assets_dir=ASSETS_DIR, intra_op_threads=2)
    return _SHARED_DETECTOR, _SHARED_POSE


def _append_audit(actor: str, action: str, target: str, detail: dict[str, Any]) -> None:
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": time.time(),
        "actor": actor,
        "action": action,
        "target": target,
        "detail": detail,
    }
    with AUDIT_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def _load_review_overrides() -> dict[str, Any]:
    if not REVIEW_OVERRIDES_PATH.is_file():
        return {}
    try:
        return json.loads(REVIEW_OVERRIDES_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_review_override(event_id: str, status: str, note: str, actor: str) -> dict[str, Any]:
    data = _load_review_overrides()
    data[event_id] = {
        "review_status": status,
        "note": note,
        "reviewed_by": actor,
        "reviewed_at": time.time(),
    }
    REVIEW_OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_OVERRIDES_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data[event_id]


def _load_completed_reports() -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    if not ANALYSIS_ROOT.exists():
        return reports
    for path in ANALYSIS_ROOT.glob("[!.]*/summary.json"):
        try:
            reports.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return sorted(reports, key=lambda r: r.get("completed_at", 0), reverse=True)


class StreamTestRequest(BaseModel):
    url: str = Field(..., min_length=5, max_length=512)


class CalibrationPayload(BaseModel):
    calibrated: bool = False
    image_points: list[list[float]] | None = None
    ground_points_m: list[list[float]] | None = None
    lane_direction_deg: float | None = None
    speed_limit_kmh: float | None = None
    stop_line_y: float | None = None


class IncidentReviewRequest(BaseModel):
    decision: Literal[
        "Human-confirmed accident",
        "Dismissed incident",
        "acknowledged",
    ]
    note: str = Field(default="", max_length=300)


@router.get("/drishti/v2/models")
def get_models() -> dict[str, Any]:
    """Return the verified model registry with live disk & checksum availability."""
    return {
        "assets_dir": str(ASSETS_DIR),
        "runtime_providers": ort.get_available_providers(),
        "models": list_models(ASSETS_DIR),
    }


@router.get("/drishti/v2/cameras")
def get_cameras_health(request: Request) -> dict[str, Any]:
    """Return honest source classification, FPS metrics, and calibration status for each camera."""
    cfg = request.app.frigate_config
    stats_emitter = getattr(request.app, "stats_emitter", None)
    latest_stats = {}
    if stats_emitter is not None:
        try:
            latest_stats = stats_emitter.get_latest_stats() or {}
        except Exception:
            latest_stats = {}

    calibrations = load_calibrations()
    cameras_stats = latest_stats.get("cameras", {})
    rows: list[dict[str, Any]] = []

    for name, cam_cfg in cfg.cameras.items():
        cstat = cameras_stats.get(name, {})
        cap_fps = float(cstat.get("camera_fps", 0.0) or 0.0)
        proc_fps = float(cstat.get("process_fps", 0.0) or 0.0)
        skip_fps = float(cstat.get("skipped_fps", 0.0) or 0.0)
        det_fps = float(cstat.get("detection_fps", 0.0) or 0.0)
        source_type = classify_camera_source(cam_cfg, cap_fps)
        inputs = [
            redact_stream_url(str(getattr(inp, "path", "")))
            for inp in (getattr(cam_cfg.ffmpeg, "inputs", []) or [])
        ]
        cal = calibrations.get(name, {})
        rows.append(
            {
                "camera_id": name,
                "source_type": source_type,
                "enabled": bool(getattr(cam_cfg, "enabled", True)),
                "inputs": inputs,
                "detect_resolution": [
                    cam_cfg.detect.width,
                    cam_cfg.detect.height,
                ],
                "target_fps": cam_cfg.detect.fps,
                "captured_fps": round(cap_fps, 2),
                "processed_fps": round(proc_fps, 2),
                "skipped_fps": round(skip_fps, 2),
                "effective_detection_fps": round(det_fps, 2),
                "offline_alert": source_type == "CAMERA OFFLINE",
                "calibrated": bool(cal.get("calibrated", False)),
                "calibration": cal,
            }
        )

    return {"cameras": rows}


@router.post(
    "/drishti/v2/cameras/test",
    dependencies=[Depends(require_role(["admin"]))],
)
def test_camera_stream(request: Request, body: StreamTestRequest) -> dict[str, Any]:
    """Test connectivity to an RTSP/HTTP stream using ffprobe."""
    ffprobe_path = request.app.frigate_config.ffmpeg.ffprobe_path
    result = test_stream_connection(body.url, ffprobe_path=ffprobe_path)
    _append_audit(
        actor="admin",
        action="camera_stream_test",
        target=result.get("url", ""),
        detail={"reachable": result.get("reachable", False)},
    )
    return result


@router.get("/drishti/v2/calibration/{camera_id}")
def get_calibration(camera_id: str) -> dict[str, Any]:
    calibrations = load_calibrations()
    return {
        "camera_id": camera_id,
        "calibration": calibrations.get(camera_id, {"calibrated": False}),
    }


@router.put(
    "/drishti/v2/calibration/{camera_id}",
    dependencies=[Depends(require_role(["admin"]))],
)
def put_calibration(camera_id: str, body: CalibrationPayload) -> dict[str, Any]:
    saved = save_camera_calibration(camera_id, body.model_dump())
    _append_audit(
        actor="admin",
        action="update_calibration",
        target=camera_id,
        detail={"calibrated": saved.get("calibrated", False)},
    )
    return {"camera_id": camera_id, "calibration": saved}


@router.get("/drishti/v2/incidents")
def get_incidents(
    camera_id: str | None = None,
    severity: str | None = None,
    review_status: str | None = None,
) -> dict[str, Any]:
    """Return real incident records from analyzed clips (empty if none detected)."""
    overrides = _load_review_overrides()
    incidents: list[dict[str, Any]] = []
    for rep in _load_completed_reports():
        for inc in rep.get("incidents", []):
            row = dict(inc)
            row["analysis_id"] = rep.get("id")
            row["analysis_name"] = rep.get("name")
            if row["event_id"] in overrides:
                ov = overrides[row["event_id"]]
                row["review_status"] = ov["review_status"]
                row["state"] = (
                    ov["review_status"]
                    if ov["review_status"]
                    in ("Human-confirmed accident", "Dismissed incident")
                    else row["state"]
                )
                row["review_note"] = ov.get("note", "")
            if camera_id and row.get("camera_id") != camera_id:
                continue
            if severity and row.get("severity") != severity:
                continue
            if review_status and row.get("review_status") != review_status:
                continue
            incidents.append(row)
    return {"incidents": incidents, "count": len(incidents)}


@router.post(
    "/drishti/v2/incidents/{event_id}/review",
    dependencies=[Depends(require_role(["admin"]))],
)
def review_incident(event_id: str, body: IncidentReviewRequest) -> dict[str, Any]:
    saved = _save_review_override(event_id, body.decision, body.note, actor="admin")
    _append_audit(
        actor="admin",
        action="incident_review",
        target=event_id,
        detail={"decision": body.decision, "note": body.note},
    )
    return {"event_id": event_id, **saved}


@router.get("/drishti/v2/risks")
def get_collision_risks() -> dict[str, Any]:
    """Return real predictive collision-risk and near-miss events from analyzed clips."""
    risks: list[dict[str, Any]] = []
    for rep in _load_completed_reports():
        for rk in rep.get("risk_events", []):
            row = dict(rk)
            row["analysis_id"] = rep.get("id")
            row["analysis_name"] = rep.get("name")
            risks.append(row)
    return {"risk_events": risks, "count": len(risks)}


@router.get("/drishti/v2/safety")
def get_safety_analytics() -> dict[str, Any]:
    """Return real safety alerts and explicit module capability flags."""
    alerts: list[dict[str, Any]] = []
    for rep in _load_completed_reports():
        for al in rep.get("safety_alerts", []):
            row = dict(al)
            row["analysis_id"] = rep.get("id")
            row["analysis_name"] = rep.get("name")
            alerts.append(row)
    default_caps = SafetyAnalyticsEngine("default", SafetyConfig()).feature_capabilities()
    return {
        "safety_alerts": alerts,
        "count": len(alerts),
        "capabilities": default_caps,
    }


@router.get("/drishti/v2/diagnostics")
def get_diagnostics() -> dict[str, Any]:
    """Return real hardware telemetry, ONNX Runtime state, and validation outputs."""
    mem = psutil.virtual_memory()
    validation_files: dict[str, Any] = {}
    for fname in (
        "coco20-results.json",
        "sample-summary.json",
        "replay-detector.json",
        "benchmark-results.json",
    ):
        fpath = VALIDATION_DIR / fname
        if fpath.is_file():
            try:
                validation_files[fname] = json.loads(fpath.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass

    return {
        "system": {
            "arch": platform.machine(),
            "python": platform.python_version(),
            "cpu_cores": os.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "ram_total_gb": round(mem.total / 1e9, 2),
            "ram_used_gb": round(mem.used / 1e9, 2),
            "ram_percent": mem.percent,
            "onnxruntime": ort.__version__,
            "ort_providers": ort.get_available_providers(),
        },
        "validation": validation_files,
        "completed_analyses": len(_load_completed_reports()),
    }


@router.get(
    "/drishti/v2/audit-log",
    dependencies=[Depends(require_role(["admin"]))],
)
def get_audit_log(limit: int = 100) -> dict[str, Any]:
    if not AUDIT_LOG_PATH.is_file():
        return {"entries": []}
    lines = AUDIT_LOG_PATH.read_text(encoding="utf-8").splitlines()[-limit:]
    entries = []
    for line in reversed(lines):
        try:
            entries.append(json.loads(line))
        except ValueError:
            continue
    return {"entries": entries}


class SosRespondRequest(BaseModel):
    action: Literal["i_am_okay", "cancel", "acknowledge", "confirm_emergency"]


class ManualSosRequest(BaseModel):
    camera_id: str = "CAM-001"
    note: str = "Manual Test SOS triggered by operator"


class ContactRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    channel: Literal["sms", "email", "whatsapp", "voice"] = "sms"
    target: str = Field(..., min_length=3, max_length=128)


class DbIncidentReviewRequest(BaseModel):
    review_status: Literal[
        "HUMAN_CONFIRMED_EMERGENCY",
        "DISMISSED",
        "ACKNOWLEDGED",
    ]


@router.post("/drishti/v2/human-safety/analyze-frame")
def analyze_live_webcam_frame(
    frame: UploadFile = File(...),
    camera_id: str = Query("CAM-001"),
    enable_pose: bool = Query(True),
) -> dict[str, Any]:
    """Run real ONNX detection, tracking, pose estimation & temporal fall detection on a live webcam frame."""
    t0 = time.perf_counter()
    raw_bytes = frame.file.read(4 * 1024 * 1024)
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(422, "Invalid JPEG frame")

    with _LIVE_LOCK:
        detector, pose_est = _get_live_models()
        ts = round(time.monotonic() - _LIVE_START_TS, 3)

        tamper_alerts = _LIVE_PUBLIC_SAFETY.check_tampering(img)
        detections, det_ms = detector.detect(img)
        trk_out = _LIVE_TRACKER.update(detections, ts)
        active_states = list(_LIVE_TRACKER.tracks.values())
        public_events = _LIVE_PUBLIC_SAFETY.evaluate_tracks(active_states, ts)

        persons_out: list[dict[str, Any]] = []
        pose_ms_total = 0.0
        highest_risk = RiskLevel.NORMAL
        overall_fall_state = FallState.NORMAL.value

        # Conditional pose estimation: only run MoveNet when a person track is visible
        person_tracks = [t for t in trk_out["tracks"] if t["label"] == "person"]
        for trk in person_tracks:
            pose_res: dict[str, Any] = {
                "keypoints": [],
                "skeleton_edges": [],
                "reliability": "disabled",
                "posture": "unknown",
                "torso_angle_deg": None,
                "latency_ms": 0.0,
            }
            fall_res: dict[str, Any] = {
                "fall_state": FallState.NORMAL.value,
                "inactivity_s": 0.0,
                "evidence": [],
            }
            if enable_pose:
                pose_res = pose_est.estimate(img, trk["xyxy"])
                pose_ms_total += float(pose_res.get("latency_ms", 0.0))
                x1, y1, x2, y2 = trk["xyxy"]
                fall_res = _LIVE_FALL.update(
                    trk["track_id"], ts, pose_res, (x1 + x2) * 0.5, (y1 + y2) * 0.5
                )

            fstate = fall_res["fall_state"]
            if fstate in (FallState.SOS_PENDING.value, FallState.SOS_ACTIVATED.value):
                highest_risk = RiskLevel.CRITICAL
                overall_fall_state = fstate
                _STORE.trigger_sos(
                    camera_id=camera_id,
                    event_type="possible_human_fall",
                    evidence=fall_res.get("evidence", ["temporal_fall_triggered"]),
                    countdown_seconds=15,
                    track_ids=[trk["track_id"]],
                    snapshot_bytes=raw_bytes,
                    manual=False,
                )
            elif fstate == FallState.SUSPECTED_FALL.value:
                if highest_risk != RiskLevel.CRITICAL:
                    highest_risk = RiskLevel.HIGH
                overall_fall_state = fstate
                _STORE.create_incident(
                    camera_id=camera_id,
                    event_type="suspected_human_fall",
                    risk_level=RiskLevel.HIGH,
                    contributing_evidence=fall_res.get("evidence", []),
                    model_version="MoveNet-Lightning + YOLOX-Nano",
                    rule_version=_LIVE_FALL.RULE_VERSION,
                    evidence_reliability=pose_res.get("reliability", "reduced"),
                    track_ids=[trk["track_id"]],
                    snapshot_bytes=raw_bytes,
                )
            elif fstate == FallState.PROLONGED_INACTIVITY.value:
                if highest_risk == RiskLevel.NORMAL:
                    highest_risk = RiskLevel.LOW
                overall_fall_state = fstate

            persons_out.append(
                {
                    "track_id": trk["track_id"],
                    "xyxy": trk["xyxy"],
                    "score": trk["score"],
                    "is_stationary": trk["is_stationary"],
                    "pose": pose_res,
                    "fall": fall_res,
                }
            )

        for ta in tamper_alerts:
            _STORE.create_incident(
                camera_id=camera_id,
                event_type=ta["event_type"],
                risk_level=RiskLevel(ta["risk_level"]),
                contributing_evidence=ta["evidence"],
                model_version="OpenCV-Tamper",
                rule_version=_LIVE_PUBLIC_SAFETY.RULE_VERSION,
                evidence_reliability="frame_statistics",
                track_ids=[],
                snapshot_bytes=raw_bytes,
            )

    end_to_end_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        "camera_id": camera_id,
        "camera_number": 1 if camera_id == "CAM-001" else 99,
        "source_type": "LIVE WEBCAM",
        "timestamp": time.time(),
        "test_mode": is_test_mode(),
        "risk_level": highest_risk.value,
        "fall_state": overall_fall_state,
        "detections": detections,
        "persons": persons_out,
        "visible_objects": trk_out["current_visible_objects"],
        "unique_tracks_total": trk_out["unique_tracks_total"],
        "tamper_alerts": tamper_alerts,
        "public_safety_events": public_events,
        "active_sos": _STORE.get_active_sos(),
        "latency": {
            "detection_ms": round(det_ms, 2),
            "pose_ms": round(pose_ms_total, 2),
            "end_to_end_ms": end_to_end_ms,
        },
    }


@router.post("/drishti/v2/human-safety/reset")
def reset_live_webcam_session() -> dict[str, Any]:
    with _LIVE_LOCK:
        _LIVE_TRACKER.reset_for_loop(0)
        _LIVE_FALL.reset()
    return {"reset": True}


@router.get("/drishti/v2/sos")
def get_sos_status() -> dict[str, Any]:
    return {
        "test_mode": is_test_mode(),
        "active_sos": _STORE.get_active_sos(),
    }


@router.post("/drishti/v2/sos/manual")
def trigger_manual_sos(body: ManualSosRequest) -> dict[str, Any]:
    sos = _STORE.trigger_sos(
        camera_id=body.camera_id,
        event_type="manual_sos_alert",
        evidence=[body.note],
        countdown_seconds=15,
        manual=True,
    )
    _append_audit("operator", "manual_sos", body.camera_id, {"sos_id": sos["sos_id"], "test_mode": is_test_mode()})
    return {"test_mode": is_test_mode(), "sos": sos}


@router.post("/drishti/v2/sos/{sos_id}/respond")
def respond_to_sos(sos_id: str, body: SosRespondRequest) -> dict[str, Any]:
    res = _STORE.respond_sos(sos_id, body.action)
    if res is None:
        raise HTTPException(404, "SOS session not found")
    _append_audit("operator", f"sos_{body.action}", sos_id, {"state": res["state"]})
    return {"sos": res}


@router.get("/drishti/v2/command-center")
def get_command_center(request: Request) -> dict[str, Any]:
    cam_health = get_cameras_health(request)
    mem = psutil.virtual_memory()
    return {
        "test_mode": is_test_mode(),
        "cameras_v2": _STORE.list_cameras(),
        "frigate_cameras": cam_health["cameras"],
        "active_sos": _STORE.get_active_sos(),
        "incidents": _STORE.list_incidents(limit=100),
        "public_safety_capabilities": _LIVE_PUBLIC_SAFETY.capabilities(),
        "contacts": _STORE.list_contacts(),
        "system": {
            "arch": platform.machine(),
            "cpu_percent": psutil.cpu_percent(interval=0.05),
            "ram_percent": mem.percent,
            "ram_used_gb": round(mem.used / 1e9, 2),
            "ram_total_gb": round(mem.total / 1e9, 2),
        },
    }


@router.post("/drishti/v2/command-center/incidents/{incident_id}/review")
def review_db_incident(incident_id: str, body: DbIncidentReviewRequest) -> dict[str, Any]:
    updated = _STORE.update_incident_review(incident_id, body.review_status)
    if updated is None:
        raise HTTPException(404, "Incident not found")
    _append_audit("operator", "db_incident_review", incident_id, {"status": body.review_status})
    return {"incident": updated}


@router.get("/drishti/v2/contacts")
def list_emergency_contacts() -> dict[str, Any]:
    return {"test_mode": is_test_mode(), "contacts": _STORE.list_contacts()}


@router.post(
    "/drishti/v2/contacts",
    dependencies=[Depends(require_role(["admin"]))],
)
def add_emergency_contact(body: ContactRequest) -> dict[str, Any]:
    row = _STORE.add_contact(body.name, body.channel, body.target)
    _append_audit("admin", "add_emergency_contact", row["contact_id"], {"channel": body.channel})
    return {"contact": row}


@router.get("/drishti/v2/evidence/{incident_id}.jpg")
def get_incident_snapshot(incident_id: str):
    path = Path(BASE_DIR) / "drishti-evidence" / f"{Path(incident_id).name}.jpg"
    if not path.is_file():
        raise HTTPException(404, "Snapshot unavailable")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, no-store"})


class ReanalyzeRequest(BaseModel):
    model_id: str = "yolox-s-640"
    sample_fps: int = Field(default=5, ge=1, le=15)


@router.post(
    "/drishti/v2/historical/reanalyze",
    dependencies=[Depends(require_role(["admin"]))],
)
def reanalyze_historical_demo(request: Request, body: ReanalyzeRequest) -> dict[str, Any]:
    """Admin-controlled reanalysis of the bundled traffic clip using a selected verified model."""
    from frigate.drishti_video import analyze

    demo_clip = ASSETS_DIR / "hyderabad.mp4"
    if not demo_clip.is_file():
        raise HTTPException(404, "Reference clip unavailable")
    ff_cfg = request.app.frigate_config.ffmpeg
    out_dir = ANALYSIS_ROOT / f"reanalyze-{int(time.time())}"
    try:
        report = analyze(
            source=demo_clip,
            dest=out_dir,
            assets=ASSETS_DIR,
            ffmpeg=ff_cfg.ffmpeg_path,
            ffprobe=ff_cfg.ffprobe_path,
            model_id=body.model_id,
            sample_fps=body.sample_fps,
        )
    except Exception as exc:
        raise HTTPException(422, f"Reanalysis failed: {exc}") from exc

    summary = {k: v for k, v in report.items() if k != "samples"}
    summary.update(
        id=out_dir.name,
        completed_at=time.time(),
        name=f"reanalyze-hyderabad-{body.model_id}.mp4",
    )
    (out_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    _append_audit(
        "admin",
        "historical_reanalyze",
        body.model_id,
        {"unique_tracks": summary.get("unique_tracks_total", 0)},
    )
    return {"reanalysis": summary}

