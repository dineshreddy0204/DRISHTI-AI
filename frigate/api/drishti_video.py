"""Private, admin-only uploaded video analyses with progress, cancellation, and CSV/JSON export."""

import fcntl
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse

from frigate.api.auth import require_role
from frigate.api.defs.tags import Tags
from frigate.const import BASE_DIR
from frigate.drishti_video import DEFAULT_SAMPLE_FPS, MAX_BYTES, InvalidVideo, probe

router = APIRouter(tags=[Tags.media])
logger = logging.getLogger(__name__)
ROOT = Path(BASE_DIR) / "drishti-analysis"
ASSETS = Path("/opt/drishti/demo")
KEEP = int(os.environ.get("DRISHTI_MAX_RETAINED_ANALYSES", 12))


def summaries():
    """A completed directory becomes visible only after an atomic rename."""
    rows = []
    for path in ROOT.glob("[!.]*/summary.json"):
        try:
            rows.append(json.loads(path.read_text()))
        except FileNotFoundError:
            continue
    return sorted(rows, key=lambda r: r["completed_at"], reverse=True)


@router.get("/drishti/analysis", dependencies=[Depends(require_role(["admin"]))])
def saved_analyses() -> list[dict]:
    """List recent completed analyses."""
    return summaries()


def write_state(
    key: str,
    state: str,
    progress: int = 0,
    stage: str = "",
    model_id: str = "yolox-nano-416",
    sample_fps: int = DEFAULT_SAMPLE_FPS,
) -> None:
    ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=ROOT, delete=False) as pending:
        json.dump(
            {
                "id": key,
                "status": state,
                "progress": progress,
                "stage": stage,
                "model_id": model_id,
                "sample_fps": sample_fps,
            },
            pending,
        )
    os.replace(pending.name, ROOT / ".status.json")


@router.get("/drishti/analysis/status", dependencies=[Depends(require_role(["admin"]))])
def analysis_status() -> dict:
    """Report the last job and identify a worker lost in a service restart."""
    state_file = ROOT / ".status.json"
    if not state_file.exists():
        return {"status": "idle", "progress": 0}
    state = json.loads(state_file.read_text())
    if state["status"] == "running":
        with (ROOT / ".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return state
            state = json.loads(state_file.read_text())
            if state["status"] == "running":
                finished = (ROOT / str(UUID(state["id"])) / "summary.json").exists()
                state["status"] = "completed" if finished else "interrupted"
    return state


@router.post("/drishti/analysis/cancel", dependencies=[Depends(require_role(["admin"]))])
def cancel_analysis() -> dict:
    """Signal the active upload worker to cancel."""
    cancel_flag = ROOT / ".cancel"
    ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    cancel_flag.write_text("1")
    return {"cancelled": True}


def finish_analysis(
    lock,
    work: Path,
    key: str,
    name: str,
    config,
    model_id: str = "yolox-nano-416",
    sample_fps: int = DEFAULT_SAMPLE_FPS,
    input_size: int | None = None,
    score_threshold: float = 0.4,
) -> dict:
    cancel_flag = ROOT / ".cancel"
    cancel_flag.unlink(missing_ok=True)
    try:
        result = work / "result"
        command = [
            sys.executable,
            "-m",
            "frigate.drishti_video",
            str(work / "source.mp4"),
            str(result),
            "--assets",
            str(ASSETS),
            "--ffmpeg",
            config.ffmpeg_path,
            "--ffprobe",
            config.ffprobe_path,
            "--model-id",
            model_id,
            "--sample-fps",
            str(sample_fps),
            "--score-threshold",
            str(score_threshold),
        ]
        if input_size is not None:
            command.extend(["--input-size", str(input_size)])

        proc = subprocess.Popen(command, stdout=subprocess.DEVNULL)
        start_t = time.monotonic()
        while proc.poll() is None:
            if cancel_flag.exists():
                proc.terminate()
                proc.wait(timeout=5)
                write_state(key, "cancelled", 0, "cancelled", model_id, sample_fps)
                return {"id": key, "status": "cancelled"}
            if time.monotonic() - start_t > 1200:
                proc.kill()
                raise subprocess.TimeoutExpired(command, 1200)
            elapsed = time.monotonic() - start_t
            est_progress = min(90, 10 + int(elapsed * 2))
            write_state(key, "running", est_progress, "processing", model_id, sample_fps)
            time.sleep(1.0)

        if proc.returncode == 2:
            raise InvalidVideo(
                "The video could not be decoded within the supported limits."
            )
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, command)

        report = json.loads((result / "report.json").read_text())
        summary = {k: v for k, v in report.items() if k != "samples"}
        summary.update(id=key, completed_at=time.time(), name=name)
        (result / "summary.json").write_text(json.dumps(summary))
        result.rename(ROOT / key)
        try:
            write_state(key, "completed", 100, "completed", model_id, sample_fps)
            for row in summaries()[KEEP:]:
                shutil.rmtree(ROOT / str(UUID(row["id"])))
        except OSError:
            logger.exception(
                "Could not update bookkeeping for completed analysis %s", key
            )
        logger.info(
            "Completed video analysis %s (%s) with %d samples and %d unique tracks",
            key,
            model_id,
            summary["sample_count"],
            summary.get("unique_tracks_total", 0),
        )
        return summary
    except InvalidVideo:
        write_state(key, "invalid", 0, "invalid", model_id, sample_fps)
        raise
    except Exception:
        write_state(key, "failed", 0, "failed", model_id, sample_fps)
        logger.exception("Video analysis %s failed", key)
        raise
    finally:
        cancel_flag.unlink(missing_ok=True)
        shutil.rmtree(work, ignore_errors=True)
        lock.close()


def run_background(*args):
    try:
        finish_analysis(*args)
    except Exception:
        return


@router.post(
    "/drishti/analysis",
    status_code=201,
    responses={202: {"description": "Accepted for background analysis"}},
    dependencies=[Depends(require_role(["admin"]))],
)
def analyze_upload(
    request: Request,
    tasks: BackgroundTasks,
    file: UploadFile,
    background: bool = False,
    model_id: str = Query("yolox-nano-416"),
    sample_fps: int = Query(DEFAULT_SAMPLE_FPS, ge=1, le=15),
    input_size: int | None = Query(None),
    score_threshold: float = Query(0.4, ge=0.1, le=0.95),
) -> dict:
    """Stage a bounded local MP4 for CPU inference and private review."""
    if file.size is not None and file.size > MAX_BYTES:
        raise HTTPException(413, f"Maximum file size is {MAX_BYTES // (1024 * 1024)} MiB.")
    ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock = (ROOT / ".lock").open("a")
    work = None
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise HTTPException(409, "Another clip is being analyzed.") from None
        for stale in ROOT.glob(".work-*"):
            shutil.rmtree(stale)
        work = Path(tempfile.mkdtemp(dir=ROOT, prefix=".work-"))
        total = 0
        source = work / "source.mp4"
        with source.open("xb") as stream:
            while chunk := file.file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_BYTES:
                    raise HTTPException(
                        413, f"Maximum file size is {MAX_BYTES // (1024 * 1024)} MiB."
                    )
                stream.write(chunk)
        config = request.app.frigate_config.ffmpeg
        try:
            probe(source, config.ffprobe_path)
        except InvalidVideo as exc:
            raise HTTPException(422, str(exc)) from exc
        key = str(uuid4())
        name = Path((file.filename or "clip.mp4").replace(chr(92), "/")).name[:96]
        args = (
            lock,
            work,
            key,
            name,
            config,
            model_id,
            sample_fps,
            input_size,
            score_threshold,
        )
        write_state(key, "running", 5, "queued", model_id, sample_fps)
        if background:
            tasks.add_task(run_background, *args)
            return JSONResponse(
                {"id": key, "status": "running", "progress": 5}, status_code=202
            )
    except Exception:
        if work is not None:
            shutil.rmtree(work, ignore_errors=True)
        lock.close()
        raise
    try:
        return finish_analysis(*args)
    except InvalidVideo as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get(
    "/drishti/analysis/{key}/{artifact}",
    dependencies=[Depends(require_role(["admin"]))],
)
def analysis_artifact(
    key: UUID,
    artifact: Literal[
        "review.mp4",
        "report.json",
        "report.csv",
        "incident_0.jpg",
        "incident_0.mp4",
        "incident_1.jpg",
        "incident_1.mp4",
        "incident_2.jpg",
        "incident_2.mp4",
        "incident_3.jpg",
        "incident_3.mp4",
        "incident_4.jpg",
        "incident_4.mp4",
    ],
):
    """Return a review clip, incident evidence, JSON report, or CSV export."""
    path = ROOT / str(key) / artifact
    if not path.is_file():
        raise HTTPException(404, "Result unavailable.")
    media_type = (
        "video/mp4"
        if artifact.endswith(".mp4")
        else "image/jpeg"
        if artifact.endswith(".jpg")
        else "text/csv"
        if artifact == "report.csv"
        else "application/json"
    )
    return FileResponse(
        path,
        media_type=media_type,
        headers={"Cache-Control": "private, no-store"},
    )
