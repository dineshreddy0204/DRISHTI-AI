"""Model registry for DRISHTI AI V2 object detection models."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import onnxruntime as ort

from frigate.drishti.indian_traffic import (
    CANONICAL_INDIAN_CLASSES,
    COCO80_TO_INDIAN,
    INDIAN_ONLY_CLASSES,
)


@dataclass
class ModelSpec:
    """Metadata, preprocessing contract, and availability of a detection model."""

    id: str
    name: str
    version: str
    architecture: str
    source_repository: str
    license: str
    filename: str
    sha256: str | None
    input_width: int
    input_height: int
    supported_sizes: list[int]
    tensor_format: str
    pixel_format: str
    normalization: str
    labelmap_file: str
    supported_labels: list[str]
    unavailable_labels: list[str]
    runtime_provider: str = "CPUExecutionProvider"
    notes: str = ""
    evaluation_results: dict[str, Any] | None = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def resolve_path(self, assets_dir: Path) -> Path:
        return assets_dir / self.filename

    def availability(self, assets_dir: Path) -> dict[str, Any]:
        model_path = self.resolve_path(assets_dir)
        if not model_path.is_file():
            return {
                "status": "unavailable_weights_missing",
                "available": False,
                "checksum_verified": False,
                "path": str(model_path),
            }
        with model_path.open("rb") as fh:
            actual_sha = hashlib.file_digest(fh, "sha256").hexdigest()
        checksum_ok = self.sha256 is None or actual_sha == self.sha256
        return {
            "status": "available" if checksum_ok else "checksum_mismatch",
            "available": checksum_ok,
            "checksum_verified": checksum_ok,
            "actual_sha256": actual_sha,
            "path": str(model_path),
        }

    def to_dict(self, assets_dir: Path | None = None) -> dict[str, Any]:
        data = asdict(self)
        if assets_dir is not None:
            data["availability"] = self.availability(assets_dir)
        return data


COCO_SUPPORTED_CANONICAL = sorted(set(COCO80_TO_INDIAN.values()))
COCO_UNAVAILABLE_INDIAN = list(INDIAN_ONLY_CLASSES)


def build_default_registry() -> dict[str, ModelSpec]:
    """Return the verified model registry for DRISHTI AI V2."""
    return {
        "yolox-nano-416": ModelSpec(
            id="yolox-nano-416",
            name="YOLOX-Nano",
            version="0.1.1rc0",
            architecture="yolox",
            source_repository="https://github.com/Megvii-BaseDetection/YOLOX",
            license="Apache-2.0",
            filename="yolox_nano.onnx",
            sha256="c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d",
            input_width=416,
            input_height=416,
            supported_sizes=[416],
            tensor_format="nchw",
            pixel_format="bgr",
            normalization="float_denorm",
            labelmap_file="coco-80.txt",
            supported_labels=COCO_SUPPORTED_CANONICAL,
            unavailable_labels=COCO_UNAVAILABLE_INDIAN,
            runtime_provider=(
                "CPUExecutionProvider"
                if "CPUExecutionProvider" in ort.get_available_providers()
                else "unavailable"
            ),
            notes=(
                "Bundled baseline CPU model. Trained on COCO-80; does not include "
                "dedicated auto-rickshaw, e-rickshaw, scooter, or helmet classes."
            ),
        ),
        "yolox-tiny-416": ModelSpec(
            id="yolox-tiny-416",
            name="YOLOX-Tiny",
            version="0.1.1rc0",
            architecture="yolox",
            source_repository="https://github.com/Megvii-BaseDetection/YOLOX",
            license="Apache-2.0",
            filename="yolox_tiny.onnx",
            sha256="427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7",
            input_width=416,
            input_height=416,
            supported_sizes=[416],
            tensor_format="nchw",
            pixel_format="bgr",
            normalization="float_denorm",
            labelmap_file="coco-80.txt",
            supported_labels=COCO_SUPPORTED_CANONICAL,
            unavailable_labels=COCO_UNAVAILABLE_INDIAN,
            runtime_provider="CPUExecutionProvider",
            notes=(
                "Balanced 416x416 YOLOX-Tiny CPU model (5.06M params): higher recall "
                "than YOLOX-Nano while staying fast on Snapdragon X ARM64 CPU."
            ),
        ),
        "yolox-s-640": ModelSpec(
            id="yolox-s-640",
            name="YOLOX-S",
            version="0.1.1rc0",
            architecture="yolox",
            source_repository="https://github.com/Megvii-BaseDetection/YOLOX",
            license="Apache-2.0",
            filename="yolox_s.onnx",
            sha256="c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
            input_width=640,
            input_height=640,
            supported_sizes=[640],
            tensor_format="nchw",
            pixel_format="bgr",
            normalization="float_denorm",
            labelmap_file="coco-80.txt",
            supported_labels=COCO_SUPPORTED_CANONICAL,
            unavailable_labels=COCO_UNAVAILABLE_INDIAN,
            runtime_provider="CPUExecutionProvider",
            notes=(
                "Higher-resolution 640x640 YOLOX-S model for improved small-object "
                "recall on CPU at higher latency than YOLOX-Nano."
            ),
        ),
        "yolov8n-coco": ModelSpec(
            id="yolov8n-coco",
            name="YOLOv8n",
            version="8.x",
            architecture="yolov8",
            source_repository="https://github.com/ultralytics/ultralytics",
            license="AGPL-3.0",
            filename="yolov8n.onnx",
            sha256=None,
            input_width=640,
            input_height=640,
            supported_sizes=[416, 512, 640],
            tensor_format="nchw",
            pixel_format="rgb",
            normalization="float_0_1",
            labelmap_file="coco-80.txt",
            supported_labels=COCO_SUPPORTED_CANONICAL,
            unavailable_labels=COCO_UNAVAILABLE_INDIAN,
            runtime_provider="CPUExecutionProvider",
            notes="Supported by decoder; requires placing exported yolov8n.onnx in demo/.",
        ),
        "yolo11n-coco": ModelSpec(
            id="yolo11n-coco",
            name="YOLO11n",
            version="11.x",
            architecture="yolo11",
            source_repository="https://github.com/ultralytics/ultralytics",
            license="AGPL-3.0",
            filename="yolo11n.onnx",
            sha256=None,
            input_width=640,
            input_height=640,
            supported_sizes=[416, 512, 640],
            tensor_format="nchw",
            pixel_format="rgb",
            normalization="float_0_1",
            labelmap_file="coco-80.txt",
            supported_labels=COCO_SUPPORTED_CANONICAL,
            unavailable_labels=COCO_UNAVAILABLE_INDIAN,
            runtime_provider="CPUExecutionProvider",
            notes="Supported by decoder; requires placing exported yolo11n.onnx in demo/.",
        ),
        "yolo26n-coco": ModelSpec(
            id="yolo26n-coco",
            name="YOLO26n (NMS-free)",
            version="26.x",
            architecture="yolo26",
            source_repository="https://github.com/ultralytics/ultralytics",
            license="AGPL-3.0",
            filename="yolo26n.onnx",
            sha256=None,
            input_width=640,
            input_height=640,
            supported_sizes=[416, 512, 640],
            tensor_format="nchw",
            pixel_format="rgb",
            normalization="float_0_1",
            labelmap_file="coco-80.txt",
            supported_labels=COCO_SUPPORTED_CANONICAL,
            unavailable_labels=COCO_UNAVAILABLE_INDIAN,
            runtime_provider="CPUExecutionProvider",
            notes="End-to-end NMS-free decoder ready; requires placing yolo26n.onnx in demo/.",
        ),
        "drishti-idd-v1": ModelSpec(
            id="drishti-idd-v1",
            name="DRISHTI Indian Traffic Detector (IDD)",
            version="1.0.0-untrained",
            architecture="yolo11",
            source_repository="https://idd.insaan.iiit.ac.in/",
            license="Research / Dataset-Specific",
            filename="drishti_idd_yolo11n.onnx",
            sha256=None,
            input_width=640,
            input_height=640,
            supported_sizes=[416, 512, 640],
            tensor_format="nchw",
            pixel_format="rgb",
            normalization="float_0_1",
            labelmap_file="indian-17.txt",
            supported_labels=list(CANONICAL_INDIAN_CLASSES),
            unavailable_labels=[],
            runtime_provider="CPUExecutionProvider",
            notes=(
                "Target 17-class Indian road model (including auto_rickshaw, e_rickshaw, "
                "scooter, tractor, ambulance). Weights unavailable until trained with "
                "scripts/train_indian_traffic.py on licensed IDD data."
            ),
        ),
        "movenet-lightning-192": ModelSpec(
            id="movenet-lightning-192",
            name="MoveNet SinglePose Lightning",
            version="v4-onnx",
            architecture="movenet",
            source_repository="https://huggingface.co/Xenova/movenet-singlepose-lightning",
            license="Apache-2.0",
            filename="movenet_lightning.onnx",
            sha256="1ad4f8d6c2f776a9967db3993c9ca740bc350104f9d37c151dc183fc29a464ad",
            input_width=192,
            input_height=192,
            supported_sizes=[192],
            tensor_format="nhwc",
            pixel_format="rgb",
            normalization="int32_0_255",
            labelmap_file="coco-80.txt",
            supported_labels=["person_17_keypoints"],
            unavailable_labels=[],
            runtime_provider="CPUExecutionProvider",
            notes="Verified 17-keypoint human pose model for live webcam and upload fall analysis.",
        ),
        "fire-smoke-yolo11n": ModelSpec(
            id="fire-smoke-yolo11n",
            name="Fire & Smoke Detector",
            version="untrained",
            architecture="yolo11",
            source_repository="https://github.com/ultralytics/ultralytics",
            license="AGPL-3.0",
            filename="fire_smoke_yolo11n.onnx",
            sha256=None,
            input_width=640,
            input_height=640,
            supported_sizes=[640],
            tensor_format="nchw",
            pixel_format="rgb",
            normalization="float_0_1",
            labelmap_file="coco-80.txt",
            supported_labels=[],
            unavailable_labels=["fire", "smoke"],
            runtime_provider="CPUExecutionProvider",
            notes="Feature disabled until trained fire/smoke ONNX weights are placed in demo/.",
        ),
        "altercation-action-v1": ModelSpec(
            id="altercation-action-v1",
            name="Physical Altercation Temporal Classifier",
            version="untrained",
            architecture="temporal_action",
            source_repository="https://github.com/onnx/models",
            license="Apache-2.0",
            filename="altercation_action.onnx",
            sha256=None,
            input_width=224,
            input_height=224,
            supported_sizes=[224],
            tensor_format="nchw",
            pixel_format="rgb",
            normalization="float_0_1",
            labelmap_file="coco-80.txt",
            supported_labels=[],
            unavailable_labels=["physical_altercation"],
            runtime_provider="CPUExecutionProvider",
            notes="Feature disabled until validated temporal action weights are placed in demo/.",
        ),
    }


MODEL_REGISTRY = build_default_registry()


def list_models(assets_dir: Path) -> list[dict[str, Any]]:
    """Return all registered models with live disk & checksum verification."""
    return [spec.to_dict(assets_dir) for spec in MODEL_REGISTRY.values()]


def get_model_spec(model_id: str) -> ModelSpec:
    """Lookup a registered model specification by ID."""
    if model_id not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model_id: {model_id}")
    return MODEL_REGISTRY[model_id]
