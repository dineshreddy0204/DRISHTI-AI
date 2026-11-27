"""Reproduce Hyderabad frame predictions for YOLOX-Nano (416) vs YOLOX-S (640) and save comparison images."""

import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frigate.drishti.detector import UnifiedDetector


def main() -> None:
    cap = cv2.VideoCapture("/opt/drishti/demo/hyderabad.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 90)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Cannot read frame 90 of hyderabad.mp4")

    summary = {}
    for mid, outname in (
        ("yolox-nano-416", "/opt/drishti/validation/hyderabad_yolox_nano.jpg"),
        ("yolox-s-640", "/opt/drishti/validation/hyderabad_yolox_s_v2.jpg"),
    ):
        det = UnifiedDetector(Path("/opt/drishti/demo"), model_id=mid, score_threshold=0.35)
        items, ms = det.detect(frame)
        img = frame.copy()
        for it in items:
            x1, y1, x2, y2 = [int(v) for v in it["xyxy"]]
            col = (0, 180, 255) if it["label"] == "unknown_vehicle" else (49, 199, 201)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            cv2.putText(
                img,
                f"{it['label']} {it['score']:.2f}",
                (x1, max(14, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )
        cv2.imwrite(outname, img)
        summary[mid] = {
            "latency_ms": round(ms, 2),
            "detections": len(items),
            "labels": sorted({it["label"] for it in items}),
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

