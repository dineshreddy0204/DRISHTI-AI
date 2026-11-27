"""DRISHTI AI V2 — Honest Camera Source Classification, Calibration Store & Stream Health."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from frigate.const import BASE_DIR

CALIBRATION_PATH = Path(BASE_DIR) / "drishti_calibration.json"


def redact_stream_url(url: str) -> str:
    """Mask embedded credentials in RTSP/HTTP camera URLs before exposing via API."""
    try:
        parsed = urlsplit(url)
        if parsed.username or parsed.password:
            host = parsed.hostname or ""
            port = f":{parsed.port}" if parsed.port else ""
            netloc = f"***:***@{host}{port}"
            return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, ""))
    except ValueError:
        pass
    return url


def classify_camera_source(camera_cfg: Any, camera_fps: float = 0.0) -> str:
    """Strictly distinguish LIVE CCTV from VIDEO REPLAY and CAMERA OFFLINE."""
    inputs = getattr(getattr(camera_cfg, "ffmpeg", None), "inputs", []) or []
    paths = [str(getattr(inp, "path", "")) for inp in inputs]
    args = " ".join(
        " ".join(getattr(inp, "input_args", []) or [])
        if isinstance(getattr(inp, "input_args", None), list)
        else str(getattr(inp, "input_args", "") or "")
        for inp in inputs
    )
    is_replay = any(
        p.endswith((".mp4", ".mkv", ".avi", ".mov", ".webm")) or p.startswith("/opt/drishti/demo")
        for p in paths
    ) or "-stream_loop" in args

    if camera_fps <= 0.01 and not is_replay:
        return "CAMERA OFFLINE"
    if is_replay:
        return "VIDEO REPLAY"
    if any(p.startswith(("rtsp://", "rtsps://", "http://", "https://", "/dev/video")) for p in paths):
        return "LIVE CCTV"
    return "VIDEO REPLAY" if paths else "CAMERA OFFLINE"


def test_stream_connection(url: str, ffprobe_path: str = "ffprobe") -> dict[str, Any]:
    """Probe an authorized RTSP/HTTP/local camera stream without storing credentials in plain logs."""
    safe_url = redact_stream_url(url)
    t0 = time.perf_counter()
    try:
        out = subprocess.check_output(
            [
                ffprobe_path,
                "-v",
                "error",
                "-rtsp_transport",
                "tcp",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,r_frame_rate",
                "-of",
                "json",
                url,
            ],
            stderr=subprocess.PIPE,
            timeout=10,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        meta = json.loads(out)
        stream = (meta.get("streams") or [{}])[0]
        return {
            "reachable": True,
            "url": safe_url,
            "latency_ms": latency_ms,
            "codec": stream.get("codec_name"),
            "width": stream.get("width"),
            "height": stream.get("height"),
            "r_frame_rate": stream.get("r_frame_rate"),
        }
    except (subprocess.SubprocessError, ValueError, OSError) as exc:
        return {
            "reachable": False,
            "url": safe_url,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "error": str(exc),
        }


def load_calibrations(path: Path = CALIBRATION_PATH) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_camera_calibration(
    camera_id: str, payload: dict[str, Any], path: Path = CALIBRATION_PATH
) -> dict[str, Any]:
    data = load_calibrations(path)
    data[camera_id] = payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data[camera_id]

