"""Install the bundled replay configuration safely with pre-migration backups and start Frigate."""
import fcntl
import hashlib
import os
import shutil
import stat
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

# Exact, pristine config.yml files shipped in the earlier DRISHTI releases.
PREVIOUS_DEFAULTS = {
    "2735b9c59677f9903b8f8fe3d2ae385475a819531de1f37cfb8b4fbcd4fea279",
    "56408645cd7a3056e7cb61af1657fd996ec965a8b7b83f8c023d8d9c7acf027a",
}


def backup_existing_state(directory: Path) -> None:
    """Create timestamped backups of config.yml and frigate.db before startup/migration."""
    backup_dir = directory / "backups"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for fname in ("config.yml", "frigate.db"):
        src = directory / fname
        if src.is_file():
            backup_dir.mkdir(exist_ok=True, parents=True)
            dst = backup_dir / f"{fname}.{stamp}.bak"
            if not dst.exists():
                shutil.copy2(src, dst)
    # Keep at most 10 backups per file type
    if backup_dir.is_dir():
        for fname in ("config.yml", "frigate.db"):
            old = sorted(backup_dir.glob(f"{fname}.*.bak"))[:-10]
            for item in old:
                item.unlink(missing_ok=True)


def install_replay_config(directory: Path, bundled: Path) -> None:
    directory.mkdir(exist_ok=True, parents=True)
    target = directory / "config.yml"
    # Frigate's config editor uses this same advisory lock on its YAML file.
    with (directory / "config.yml.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (directory / "config.yaml").exists() or target.is_symlink():
            return
        existing = None
        if target.exists():
            with target.open("rb") as source:
                if hashlib.file_digest(source, "sha256").hexdigest() not in PREVIOUS_DEFAULTS:
                    # Preserve custom user configuration untouched
                    backup_existing_state(directory)
                    return
                existing = os.fstat(source.fileno())
            backup_existing_state(directory)
        with NamedTemporaryFile(dir=directory, delete=False) as pending:
            staging = Path(pending.name)
            try:
                with bundled.open("rb") as source:
                    shutil.copyfileobj(source, pending)
                if existing:
                    os.fchown(pending.fileno(), existing.st_uid, existing.st_gid)
                mode = stat.S_IMODE(existing.st_mode) if existing else 0o644
                os.fchmod(pending.fileno(), mode)
                pending.flush()
                os.fsync(pending.fileno())
                os.replace(staging, target)
            finally:
                staging.unlink(missing_ok=True)


if __name__ == "__main__":
    install_replay_config(Path("/config"), Path("/opt/drishti/config.yml"))
    os.execv("/init", ["/init"])
