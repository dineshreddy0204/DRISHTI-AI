"""Inspect local surveillance clips using DRISHTI V2 multi-model detection, tracking & temporal incident intelligence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import onnxruntime as ort

from frigate.drishti.accident_engine import (
    AccidentConfig,
    TemporalAccidentEngine,
    estimate_camera_motion,
)
from frigate.drishti.collision_risk import CalibrationConfig, CollisionRiskEngine
from frigate.drishti.detector import UnifiedDetector
from frigate.drishti.fall_detection import FallState, TemporalFallEngine
from frigate.drishti.pose_estimator import MoveNetPoseEstimator
from frigate.drishti.public_safety import PublicSafetyEngine
from frigate.drishti.safety_analytics import SafetyAnalyticsEngine, SafetyConfig
from frigate.drishti.sos_engine import IncidentSOSStore, RiskLevel
from frigate.drishti.tracker_analytics import TrackAnalyticsEngine

MAX_BYTES = int(os.environ.get("DRISHTI_MAX_UPLOAD_BYTES", 32 * 1024 * 1024))
MAX_SECONDS = int(os.environ.get("DRISHTI_MAX_UPLOAD_SECONDS", 120))
DEFAULT_SAMPLE_FPS = 10
MODEL_SHA = "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d"


class InvalidVideo(ValueError):
    """The input exceeds the advertised media limits or cannot be decoded."""


class AnalysisCancelled(RuntimeError):
    """Raised when an administrator cancels an active analysis job."""


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def probe(path: Path, ffprobe: str) -> float:
    if not 0 < path.stat().st_size <= MAX_BYTES:
        raise InvalidVideo(
            f"Choose an MP4 file up to {MAX_BYTES // (1024 * 1024)} MiB."
        )
    with path.open("rb") as stream:
        if stream.read(12)[4:8] != b"ftyp":
            raise InvalidVideo("Choose an H.264 MP4 file.")
    try:
        meta = json.loads(
            subprocess.check_output(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-protocol_whitelist",
                    "file",
                    "-f",
                    "mov",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name,width,height:format=duration",
                    "-of",
                    "json",
                    str(path),
                ],
                stderr=subprocess.DEVNULL,
                timeout=12,
            )
        )
        stream = meta["streams"][0]
        duration = float(meta["format"]["duration"])
        w, h = int(stream["width"]), int(stream["height"])
    except (subprocess.SubprocessError, ValueError, KeyError, IndexError) as exc:
        raise InvalidVideo("Cannot read the video stream.") from exc
    if (
        stream["codec_name"] != "h264"
        or not 0 < duration <= MAX_SECONDS
        or not 0 < w <= 1920
        or not 0 < h <= 1920
        or w * h > 1920 * 1080
    ):
        raise InvalidVideo(
            f"Use H.264 MP4 up to 1080p, {MAX_SECONDS} seconds and {MAX_BYTES // (1024 * 1024)} MiB."
        )
    return duration


class FrameDetector(UnifiedDetector):
    """Backward-compatible wrapper around UnifiedDetector."""

    def __init__(
        self,
        assets: Path,
        model_id: str = "yolox-nano-416",
        input_size: int | None = None,
        score_threshold: float = 0.4,
        nms_iou: float = 0.4,
        intra_op_threads: int = 2,
    ) -> None:
        super().__init__(
            assets_dir=assets,
            model_id=model_id,
            input_size=input_size,
            score_threshold=score_threshold,
            nms_iou=nms_iou,
            intra_op_threads=intra_op_threads,
        )


def _write_csv_report(path: Path, samples: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "timestamp_s",
                "track_id",
                "label",
                "score",
                "x1",
                "y1",
                "x2",
                "y2",
                "speed_px_s",
                "heading_deg",
                "is_stationary",
            ]
        )
        for sample in samples:
            ts = sample["seconds"]
            for trk in sample.get("tracks", []):
                x1, y1, x2, y2 = trk["xyxy"]
                writer.writerow(
                    [
                        ts,
                        trk["track_id"],
                        trk["label"],
                        trk["score"],
                        x1,
                        y1,
                        x2,
                        y2,
                        trk["speed_px_s"],
                        trk["heading_deg"],
                        trk["is_stationary"],
                    ]
                )


def analyze(
    source: Path,
    dest: Path,
    assets: Path,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    model_id: str = "yolox-nano-416",
    sample_fps: int = DEFAULT_SAMPLE_FPS,
    input_size: int | None = None,
    score_threshold: float = 0.4,
    intra_op_threads: int = 2,
    source_profile: str = "auto",
    processing_profile: str = "detailed_offline",
    calibration: CalibrationConfig | None = None,
    progress_cb: Callable[[int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Publish report, CSV, and annotated MP4 with tracking, accident, and risk intelligence."""
    if dest.exists():
        raise FileExistsError(dest)
    sample_fps = max(1, min(15, int(sample_fps)))
    duration = probe(source, ffprobe)
    cv2.setNumThreads(1)
    t_wall0 = time.perf_counter()

    detector = FrameDetector(
        assets,
        model_id=model_id,
        input_size=input_size,
        score_threshold=score_threshold,
        intra_op_threads=intra_op_threads,
    )
    tracker = TrackAnalyticsEngine(camera_id="upload", fps=float(sample_fps))
    accident_engine = TemporalAccidentEngine(
        camera_id="upload",
        source_type="UPLOADED VIDEO",
        model_version=detector.spec.name,
        config=AccidentConfig(),
    )
    risk_engine = CollisionRiskEngine(camera_id="upload", calibration=calibration)
    safety_engine = SafetyAnalyticsEngine(
        camera_id="upload",
        config=SafetyConfig(is_calibrated=risk_engine.is_calibrated),
    )
    pose_estimator = (
        MoveNetPoseEstimator(assets, intra_op_threads=intra_op_threads)
        if (assets / "movenet_lightning.onnx").is_file()
        else None
    )
    fall_engine = TemporalFallEngine()
    public_safety = PublicSafetyEngine(camera_id="upload")

    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".processing-") as work:
        stage = Path(work)
        output = stage / "output"
        output.mkdir()
        pattern = str(stage / "%04d.jpg")
        if progress_cb:
            progress_cb(5, "decoding")
        try:
            subprocess.run(
                [
                    ffmpeg,
                    "-v",
                    "error",
                    "-nostdin",
                    "-xerror",
                    "-threads",
                    "2",
                    "-protocol_whitelist",
                    "file",
                    "-f",
                    "mov",
                    "-i",
                    str(source),
                    "-map",
                    "0:v:0",
                    "-filter_threads",
                    "1",
                    "-vf",
                    f"setpts=PTS-STARTPTS,fps={sample_fps}:start_time=0:round=up,scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,setsar=1",
                    "-frames:v",
                    str(MAX_SECONDS * sample_fps + 1),
                    "-q:v",
                    "2",
                    "-threads",
                    "1",
                    pattern,
                ],
                stderr=subprocess.DEVNULL,
                timeout=120,
                check=True,
            )
        except subprocess.SubprocessError as exc:
            raise InvalidVideo(
                "Cannot decode this clip within the supported limits."
            ) from exc

        frames = sorted(stage.glob("*.jpg"))
        if not 0 < len(frames) <= MAX_SECONDS * sample_fps + 1:
            raise InvalidVideo("Decoded duration exceeds supported limits.")

        samples: list[dict[str, Any]] = []
        times: list[float] = []
        counts: Counter[str] = Counter()
        fall_events: list[dict[str, Any]] = []
        public_safety_events: list[dict[str, Any]] = []
        total_frames = len(frames)
        prev_gray = None

        for index, path in enumerate(frames):
            if cancel_check and cancel_check():
                raise AnalysisCancelled("Analysis cancelled by administrator.")
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cam_motion = estimate_camera_motion(prev_gray, curr_gray)
            prev_gray = curr_gray
            ts = round(index / sample_fps, 3)
            items, ms = detector.detect(frame)
            times.append(ms)
            for item in items:
                counts[item["label"]] += 1

            trk_out = tracker.update(items, ts)
            active_states = list(tracker.tracks.values())
            acc_out = accident_engine.evaluate(
                active_states,
                ts,
                frame_area=float(frame.shape[0] * frame.shape[1]),
                camera_motion=cam_motion,
            )
            risk_out = risk_engine.evaluate(active_states, ts)
            saf_out = safety_engine.evaluate(active_states, ts)
            pub_evs = public_safety.evaluate_tracks(active_states, ts)
            for pe in pub_evs:
                public_safety_events.append({"timestamp_s": ts, **pe})

            if pose_estimator is not None:
                for trk in trk_out["tracks"]:
                    if trk["label"] != "person":
                        continue
                    pose_res = pose_estimator.estimate(frame, trk["xyxy"])
                    px1, py1, px2, py2 = trk["xyxy"]
                    fall_res = fall_engine.update(
                        trk["track_id"], ts, pose_res, (px1 + px2) * 0.5, (py1 + py2) * 0.5
                    )
                    if fall_res["fall_state"] in (
                        FallState.SUSPECTED_FALL.value,
                        FallState.SOS_PENDING.value,
                    ):
                        fall_events.append({"timestamp_s": ts, **fall_res})
                    kps = pose_res.get("keypoints", [])
                    for u, v in pose_res.get("skeleton_edges", []):
                        if u < len(kps) and v < len(kps) and kps[u]["visible"] and kps[v]["visible"]:
                            cv2.line(
                                frame,
                                (int(kps[u]["x"]), int(kps[u]["y"])),
                                (int(kps[v]["x"]), int(kps[v]["y"])),
                                (255, 215, 0),
                                1,
                                cv2.LINE_AA,
                            )

            # Draw confirmed tracks, trajectories, and incident banners
            for trk in trk_out["tracks"]:
                x1, y1, x2, y2 = (round(v) for v in trk["xyxy"])
                color = (49, 199, 201) if not trk["is_stationary"] else (240, 173, 78)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                short_id = trk["track_id"].split("-")[-1]
                cv2.putText(
                    frame,
                    f"{short_id} {trk['label']} {trk['score']:.2f}",
                    (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                pts = trk.get("trajectory", [])
                for k in range(1, len(pts)):
                    cv2.line(
                        frame,
                        (int(pts[k - 1][0]), int(pts[k - 1][1])),
                        (int(pts[k][0]), int(pts[k][1])),
                        (49, 199, 201),
                        1,
                        cv2.LINE_AA,
                    )

            if acc_out["state"] != "Normal":
                cv2.putText(
                    frame,
                    f"INCIDENT STATE: {acc_out['state']}",
                    (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 69, 255),
                    2,
                    cv2.LINE_AA,
                )

            if not cv2.imwrite(str(path), frame):
                raise OSError("Cannot save annotated frame")

            samples.append(
                {
                    "seconds": ts,
                    "objects": items,
                    "tracks": trk_out["tracks"],
                    "visible_count": trk_out["current_visible_objects"],
                    "unique_tracks_so_far": trk_out["unique_tracks_total"],
                    "incident_state": acc_out["state"],
                    "congestion_level": saf_out["congestion_level"],
                }
            )
            if progress_cb and (index % max(1, total_frames // 20) == 0):
                pct = 10 + int(75 * (index + 1) / total_frames)
                progress_cb(pct, "analyzing")

        if progress_cb:
            progress_cb(88, "encoding")
        subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-nostdin",
                "-framerate",
                str(sample_fps),
                "-i",
                pattern,
                "-threads",
                "2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output / "review.mp4"),
            ],
            timeout=120,
            check=True,
        )

        # Save snapshot + evidence clip + historical TEST SOS for qualifying accident incidents
        local_test_sos: dict[str, Any] | None = None
        for idx, inc in enumerate(accident_engine.incidents[:5]):
            frame_idx = min(len(frames) - 1, max(0, int(round(inc.timestamp_s * sample_fps))))
            snap_name = f"incident_{idx}.jpg"
            clip_name = f"incident_{idx}.mp4"
            snap_src = frames[frame_idx]
            if snap_src.is_file():
                (output / snap_name).write_bytes(snap_src.read_bytes())
                inc.snapshot_file = snap_name
            start_s = max(0.0, inc.timestamp_s - 2.0)
            try:
                subprocess.run(
                    [
                        ffmpeg,
                        "-v",
                        "error",
                        "-nostdin",
                        "-ss",
                        f"{start_s:.2f}",
                        "-t",
                        "4.0",
                        "-i",
                        str(output / "review.mp4"),
                        "-c",
                        "copy",
                        str(output / clip_name),
                    ],
                    timeout=30,
                    check=True,
                )
                inc.clip_file = clip_name
            except subprocess.SubprocessError:
                pass

        if accident_engine.incidents:
            try:
                store = IncidentSOSStore()
                first_inc = accident_engine.incidents[0]
                snap_bytes = (
                    (output / first_inc.snapshot_file).read_bytes()
                    if first_inc.snapshot_file and (output / first_inc.snapshot_file).is_file()
                    else None
                )
                rec = store.create_incident(
                    camera_id="upload",
                    event_type=first_inc.category,
                    risk_level=RiskLevel.CRITICAL,
                    contributing_evidence=first_inc.contributing_evidence,
                    model_version=detector.spec.name,
                    rule_version=accident_engine.RULE_VERSION,
                    evidence_reliability="temporal_kinematic_fusion",
                    track_ids=first_inc.involved_track_ids,
                    source_type="UPLOADED VIDEO",
                    snapshot_bytes=snap_bytes,
                    sos_status="TEST_ALERT_HISTORICAL",
                    is_historical=True,
                )
                local_test_sos = {
                    "incident_id": rec["incident_id"],
                    "status": "TEST_ALERT_HISTORICAL",
                    "notification_status": rec["notification_status"],
                    "note": "Uploaded video incident — external emergency dispatch is strictly blocked.",
                }
            except Exception:
                local_test_sos = None

        wall_s = round(time.perf_counter() - t_wall0, 3)
        unique_by_label = dict(
            Counter(tracker.unique_confirmed_ids.values())
        )
        report = {
            "source_type": "UPLOADED VIDEO",
            "source_profile": source_profile,
            "processing_profile": processing_profile,
            "is_historical": True,
            "sos_dispatch_blocked": True,
            "model": detector.spec.name,
            "model_id": detector.spec.id,
            "model_sha256": detector.model_sha256,
            "input_sha256": sha256(source),
            "provider": detector.session.get_providers()[0],
            "onnxruntime": ort.__version__,
            "arch": platform.machine(),
            "input_size": detector.size,
            "sample_fps": sample_fps,
            "duration_s": round(duration, 2),
            "frame_size": [640, 360],
            "sample_count": len(samples),
            "box_count": sum(counts.values()),
            "boxes_by_label": dict(counts),
            "filtered_non_traffic_counts": dict(detector.filtered_non_traffic_counts),
            "unique_tracks_total": len(tracker.unique_confirmed_ids),
            "unique_tracks_by_label": unique_by_label,
            "median_inference_ms": round(float(np.median(times)), 2) if times else 0.0,
            "p95_inference_ms": round(float(np.percentile(times, 95)), 2) if times else 0.0,
            "end_to_end_fps": round(len(samples) / max(wall_s, 1e-3), 2),
            "total_wall_s": wall_s,
            "score_threshold": score_threshold,
            "nms_iou": detector.nms_iou,
            "incidents": [inc.to_dict() for inc in accident_engine.incidents],
            "local_test_sos": local_test_sos,
            "fall_events": fall_events,
            "public_safety_events": public_safety_events,
            "risk_events": [rk.to_dict() for rk in risk_engine.risk_events],
            "safety_alerts": [al.to_dict() for al in safety_engine.alerts],
            "safety_capabilities": safety_engine.feature_capabilities(),
            "samples": samples,
        }
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        _write_csv_report(output / "report.csv", samples)
        output.rename(dest)
        if progress_cb:
            progress_cb(100, "completed")
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("dest", type=Path)
    parser.add_argument("--assets", type=Path, default=Path("demo"))
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--model-id", default="yolox-nano-416")
    parser.add_argument("--sample-fps", type=int, default=DEFAULT_SAMPLE_FPS)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.4)
    args = parser.parse_args()
    try:
        report = analyze(
            args.source,
            args.dest,
            args.assets,
            args.ffmpeg,
            args.ffprobe,
            model_id=args.model_id,
            sample_fps=args.sample_fps,
            input_size=args.input_size,
            score_threshold=args.score_threshold,
        )
    except InvalidVideo as exc:
        raise SystemExit(2) from exc
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
