"""Small ground-truth evaluation of full-frame road-user detection, at fixed thresholds."""

from collections import Counter
import json
from pathlib import Path
import sys

import cv2
import onnxruntime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from frigate.drishti_video import FrameDetector, MODEL_SHA

CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck")


def overlap(box, truth, crowd=False):
    x1, y1, x2, y2 = box
    gx, gy, gw, gh = truth
    intersection = max(0, min(x2, gx + gw) - max(x1, gx)) * max(
        0, min(y2, gy + gh) - max(y1, gy)
    )
    area = (x2 - x1) * (y2 - y1)
    divisor = area if crowd else area + gw * gh - intersection
    return intersection / divisor if divisor else 0


def evaluate():
    data = json.loads((ROOT / "validation/coco20-boxes.json").read_text())
    model = FrameDetector(ROOT / "demo")
    classes = {c["id"]: c["name"] for c in data["categories"]}
    counts = {c: Counter(tp=0, fp=0, fn=0, ignored_crowd=0) for c in CLASSES}
    for img in data["images"]:
        frame = cv2.imread(str(ROOT / "docker/rockchip/COCO/subset" / img["file_name"]))
        boxes, _ = model.detect(frame)
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
                best = max(unused, key=lambda k: overlap(box, unused[k]), default=None)
                if best is not None and overlap(box, unused[best]) >= 0.5:
                    row["tp"] += 1
                    del unused[best]
                elif any(overlap(box, b, True) >= 0.5 for b in crowds):
                    row["ignored_crowd"] += 1
                else:
                    row["fp"] += 1
            row["fn"] += len(unused)
    totals = sum(counts.values(), Counter())
    tp, fp, fn = (totals[key] for key in ("tp", "fp", "fn"))
    return {
        "images": len(data["images"]),
        "model_sha256": MODEL_SHA,
        "provider": model.session.get_providers()[0],
        "onnxruntime": onnxruntime.__version__,
        "score_cutoff": 0.4,
        "match_iou": 0.5,
        "per_class": counts,
        "total": dict(totals),
        "micro_precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "micro_recall": round(tp / (tp + fn), 4) if tp + fn else None,
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
