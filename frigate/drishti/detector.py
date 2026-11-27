"""Unified architecture-aware ONNX object detector for DRISHTI AI V2."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from frigate.drishti.indian_traffic import map_detection_label
from frigate.drishti.model_registry import ModelSpec, get_model_spec
from frigate.util.model import (
    post_process_yolo26,
    post_process_yolo_v8_11,
    post_process_yolox,
)


TRAFFIC_ALLOWED_LABELS: frozenset[str] = frozenset(
    {
        "person",
        "bicycle",
        "motorcycle",
        "scooter",
        "auto_rickshaw",
        "e_rickshaw",
        "car",
        "suv",
        "van",
        "bus",
        "mini_bus",
        "truck",
        "tractor",
        "ambulance",
        "traffic_light",
        "road_obstacle",
        "backpack",
        "handbag",
        "suitcase",
        "unknown_vehicle",
    }
)


def _load_labels(path: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            labels[int(parts[0])] = parts[1]
        else:
            labels[len(labels)] = line
    return labels


class UnifiedDetector:
    """Architecture-aware ONNX Runtime CPU detector with registry verification."""

    def __init__(
        self,
        assets_dir: Path,
        model_id: str = "yolox-nano-416",
        input_size: int | None = None,
        score_threshold: float = 0.35,
        nms_iou: float = 0.45,
        intra_op_threads: int = 2,
        max_detections: int = 80,
        uncertain_floor: float = 0.25,
        allowed_labels: frozenset[str] | None = TRAFFIC_ALLOWED_LABELS,
    ) -> None:
        self.allowed_labels = allowed_labels
        self.filtered_non_traffic_counts: dict[str, int] = {}
        self.spec: ModelSpec = get_model_spec(model_id)
        avail = self.spec.availability(assets_dir)
        if not avail["available"]:
            raise RuntimeError(
                f"Model '{model_id}' is unavailable ({avail['status']} at {avail['path']})"
            )

        self.model_sha256: str = avail["actual_sha256"]
        self.score_threshold = score_threshold
        self.nms_iou = nms_iou
        self.max_detections = max_detections
        self.uncertain_floor = uncertain_floor
        self.size = input_size or self.spec.input_width
        if self.size not in self.spec.supported_sizes:
            raise ValueError(
                f"Input size {self.size} unsupported for {model_id}; "
                f"allowed: {self.spec.supported_sizes}"
            )

        self.labels = _load_labels(assets_dir / self.spec.labelmap_file)
        self.supported_canonical = set(self.spec.supported_labels)

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, intra_op_threads)
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            avail["path"],
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

        self.grid: np.ndarray | None = None
        self.strides: np.ndarray | None = None
        if self.spec.architecture == "yolox":
            grids, strides = [], []
            for stride in (8, 16, 32):
                n = self.size // stride
                grids.append(
                    np.stack(np.meshgrid(np.arange(n), np.arange(n)), -1).reshape(
                        1, -1, 2
                    )
                )
                strides.append(np.full((1, n * n, 1), stride))
            self.grid = np.concatenate(grids, 1)
            self.strides = np.concatenate(strides, 1)

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, int, int]:
        h, w = frame.shape[:2]
        r = self.size / max(h, w)
        rw, rh = max(1, int(w * r)), max(1, int(h * r))
        canvas = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        resized = cv2.resize(frame, (rw, rh))
        if self.spec.pixel_format == "rgb":
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        canvas[:rh, :rw] = resized
        tensor = canvas.transpose(2, 0, 1)[None].astype(np.float32)
        if self.spec.normalization == "float_0_1":
            tensor /= 255.0
        return np.ascontiguousarray(tensor), rw, rh

    def detect(self, frame: np.ndarray) -> tuple[list[dict[str, Any]], float]:
        """Run detection and return canonical Indian traffic detections + latency (ms)."""
        h, w = frame.shape[:2]
        tensor, rw, rh = self._preprocess(frame)
        t0 = time.perf_counter()
        raw = self.session.run(None, {self.input_name: tensor})[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        eval_floor = min(self.score_threshold, self.uncertain_floor)
        if self.spec.architecture == "yolox":
            rows = post_process_yolox(
                raw,
                self.size,
                self.size,
                self.grid,
                self.strides,
                score_threshold=eval_floor,
                nms_threshold=self.nms_iou,
                max_detections=self.max_detections,
            )
        elif self.spec.architecture in ("yolov8", "yolo11"):
            rows = post_process_yolo_v8_11(
                raw,
                self.size,
                self.size,
                score_threshold=eval_floor,
                nms_threshold=self.nms_iou,
                max_detections=self.max_detections,
            )
        elif self.spec.architecture == "yolo26":
            rows = post_process_yolo26(
                raw,
                self.size,
                self.size,
                score_threshold=eval_floor,
                max_detections=self.max_detections,
            )
        else:
            raise ValueError(f"Unsupported architecture: {self.spec.architecture}")

        items: list[dict[str, Any]] = []
        for cls_id, score, y1, x1, y2, x2 in rows:
            if score <= 0:
                continue
            raw_label = self.labels.get(int(cls_id), "unknown_vehicle")
            mapped = map_detection_label(
                raw_label=raw_label,
                score=float(score),
                supported_labels=self.supported_canonical,
                score_threshold=self.score_threshold,
                uncertain_floor=self.uncertain_floor,
            )
            if mapped.is_uncertain and mapped.label != "unknown_vehicle":
                continue
            if self.allowed_labels is not None and mapped.label not in self.allowed_labels:
                self.filtered_non_traffic_counts[mapped.label] = (
                    self.filtered_non_traffic_counts.get(mapped.label, 0) + 1
                )
                continue

            px1, px2 = np.clip(np.array([x1, x2]) * self.size * w / rw, 0, w)
            py1, py2 = np.clip(np.array([y1, y2]) * self.size * h / rh, 0, h)
            if px2 > px1 and py2 > py1:
                items.append(
                    {
                        "label": mapped.label,
                        "raw_label": mapped.raw_label,
                        "is_uncertain": mapped.is_uncertain,
                        "uncertainty_reason": mapped.uncertainty_reason,
                        "score": round(float(score), 6),
                        "xyxy": [round(float(v), 2) for v in (px1, py1, px2, py2)],
                    }
                )
        return items, elapsed_ms

