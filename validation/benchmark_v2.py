"""Benchmark available ONNX models on COCO-20 and Hyderabad traffic clip on ARM64 CPU."""

from __future__ import annotations

import glob
import json
import os
import platform
import tempfile
import time
from collections import Counter
from pathlib import Path
import sys

import cv2
import numpy as np
import onnxruntime as ort
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frigate.drishti.detector import UnifiedDetector
from frigate.drishti.model_registry import list_models
from frigate.drishti_video import analyze

COCO_CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck")


def _overlap(box, truth, crowd=False):
    x1, y1, x2, y2 = box
    gx, gy, gw, gh = truth
    inter = max(0, min(x2, gx + gw) - max(x1, gx)) * max(
        0, min(y2, gy + gh) - max(y1, gy)
    )
    area = (x2 - x1) * (y2 - y1)
    denom = area if crowd else area + gw * gh - inter
    return inter / denom if denom else 0.0


def eval_coco_subset(assets_dir: Path, model_id: str, score_threshold: float = 0.4):
    data = json.loads((ROOT / "validation/coco20-boxes.json").read_text())
    det = UnifiedDetector(
        assets_dir=assets_dir,
        model_id=model_id,
        score_threshold=score_threshold,
        nms_iou=0.4,
        intra_op_threads=2,
    )
    classes = {c["id"]: c["name"] for c in data["categories"]}
    counts = {c: Counter(tp=0, fp=0, fn=0, ignored_crowd=0) for c in COCO_CLASSES}
    latencies = []

    for img in data["images"]:
        frame = cv2.imread(str(ROOT / "docker/rockchip/COCO/subset" / img["file_name"]))
        boxes, ms = det.detect(frame)
        latencies.append(ms)
        for name, row in counts.items():
            ann = [
                a
                for a in data["annotations"]
                if a["image_id"] == img["id"] and classes[a["category_id"]] == name
            ]
            unused = {a["id"]: a["bbox"] for a in ann if not a["iscrowd"]}
            crowds = [a["bbox"] for a in ann if a["iscrowd"]]
            for obj in sorted(
                (o for o in boxes if o["label"] == name), key=lambda o: -o["score"]
            ):
                box = obj["xyxy"]
                best = max(unused, key=lambda k: _overlap(box, unused[k]), default=None)
                if best is not None and _overlap(box, unused[best]) >= 0.5:
                    row["tp"] += 1
                    del unused[best]
                elif any(_overlap(box, b, True) >= 0.5 for b in crowds):
                    row["ignored_crowd"] += 1
                else:
                    row["fp"] += 1
            row["fn"] += len(unused)

    totals = sum(counts.values(), Counter())
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    return {
        "model_id": model_id,
        "input_size": det.size,
        "images": len(data["images"]),
        "total": dict(totals),
        "micro_precision": round(tp / (tp + fp), 4) if (tp + fp) else 0.0,
        "micro_recall": round(tp / (tp + fn), 4) if (tp + fn) else 0.0,
        "median_inference_ms": round(float(np.median(latencies)), 2),
        "p95_inference_ms": round(float(np.percentile(latencies, 95)), 2),
    }


def main():
    assets = ROOT / "demo"
    ff = (glob.glob("/usr/lib/ffmpeg/*/bin/ffmpeg") or ["ffmpeg"])[0]
    fp = (glob.glob("/usr/lib/ffmpeg/*/bin/ffprobe") or ["ffprobe"])[0]
    available_models = [
        m["id"] for m in list_models(assets) if m["availability"]["available"]
    ]
    results = {
        "arch": platform.machine(),
        "onnxruntime": ort.__version__,
        "cpu_cores": os.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "coco20_benchmarks": [],
        "hyderabad_video_benchmarks": [],
    }
    for mid in available_models:
        results["coco20_benchmarks"].append(eval_coco_subset(assets, mid))
        with tempfile.TemporaryDirectory() as tmp:
            rep = analyze(
                ROOT / "demo/hyderabad.mp4",
                Path(tmp) / "out",
                assets,
                ffmpeg=ff,
                ffprobe=fp,
                model_id=mid,
                sample_fps=5,
            )
            results["hyderabad_video_benchmarks"].append(
                {
                    "model_id": mid,
                    "sample_fps": rep["sample_fps"],
                    "sample_count": rep["sample_count"],
                    "box_count": rep["box_count"],
                    "unique_tracks_total": rep["unique_tracks_total"],
                    "unique_tracks_by_label": rep["unique_tracks_by_label"],
                    "median_inference_ms": rep["median_inference_ms"],
                    "p95_inference_ms": rep["p95_inference_ms"],
                    "end_to_end_fps": rep["end_to_end_fps"],
                    "total_wall_s": rep["total_wall_s"],
                    "incidents_detected": len(rep["incidents"]),
                    "risk_events_detected": len(rep["risk_events"]),
                }
            )
    out_path = ROOT / "validation/benchmark-results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

