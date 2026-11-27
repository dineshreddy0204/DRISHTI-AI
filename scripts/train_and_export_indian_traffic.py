"""Reproducible training, validation, and ONNX export for Indian traffic models.

Requires `ultralytics` in a training environment (x86_64/GPU workstation).
Exports ONNX with SHA-256 checksum for registration in `model_registry.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to idd_drishti.yaml")
    parser.add_argument("--weights", default="yolo11n.pt", help="Base checkpoint")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640, choices=[416, 512, 640])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", type=Path, default=Path("runs/drishti_idd"))
    parser.add_argument("--export-dir", type=Path, default=Path("demo"))
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Install ultralytics in your training environment: pip install ultralytics"
        ) from exc

    model = YOLO(args.weights)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project),
        name="train",
    )
    val_metrics = model.val(data=str(args.data), imgsz=args.imgsz, split="test")
    exported = Path(model.export(format="onnx", imgsz=args.imgsz, simplify=True, opset=17))
    args.export_dir.mkdir(parents=True, exist_ok=True)
    target = args.export_dir / "drishti_idd_yolo11n.onnx"
    target.write_bytes(exported.read_bytes())

    summary = {
        "onnx_path": str(target),
        "sha256": sha256_file(target),
        "imgsz": args.imgsz,
        "metrics": {
            "map50": float(getattr(val_metrics.box, "map50", 0.0)),
            "map50_95": float(getattr(val_metrics.box, "map", 0.0)),
            "precision": float(getattr(val_metrics.box, "mp", 0.0)),
            "recall": float(getattr(val_metrics.box, "mr", 0.0)),
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

