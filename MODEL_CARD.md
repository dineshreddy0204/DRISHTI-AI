# MODEL_CARD.md — DRISHTI AI V2 Object Detection Models

**Maintainer:** PALLEM DINESH KUMAR REDDY  
**Runtime:** ONNX Runtime (`CPUExecutionProvider`) on ARM64 (`aarch64`) & AMD64

## 1. Verified Bundled Models

| Model ID | Architecture | Input Size | License | Supported Canonical Classes | Unavailable Indian Classes |
|---|---|---|---|---|---|
| `yolox-nano-416` | YOLOX-Nano (`0.1.1rc0`) | 416 x 416 BGR (`float_denorm`) | Apache-2.0 | `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`, `traffic_light` (+ `unknown_vehicle` uncertain state) | `scooter`, `auto_rickshaw`, `e_rickshaw`, `suv`, `van`, `mini_bus`, `tractor`, `ambulance`, `road_obstacle` |
| `yolox-s-640` | YOLOX-S (`0.1.1rc0`) | 640 x 640 BGR (`float_denorm`) | Apache-2.0 | Same COCO-80 road-user subset | Same as above until IDD weights are trained |

### Checksums (SHA-256)
- `demo/yolox_nano.onnx`: `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d`
- `demo/yolox_s.onnx`: `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`

## 2. Registered Architectures Awaiting Trained Weights

| Model ID | Architecture | Status | How to Enable |
|---|---|---|---|
| `yolov8n-coco` | YOLOv8n (`(1, 84, N)` RGB 0..1) | `unavailable_weights_missing` | Place `demo/yolov8n.onnx` |
| `yolo11n-coco` | YOLO11n (`(1, 84, N)` RGB 0..1) | `unavailable_weights_missing` | Place `demo/yolo11n.onnx` |
| `yolo26n-coco` | YOLO26n NMS-free (`(1, N, 6)` RGB 0..1) | `unavailable_weights_missing` | Place `demo/yolo26n.onnx` |
| `drishti-idd-v1` | YOLO11n 17-class Indian Traffic | `unavailable_weights_missing` | Train on licensed IDD data via `scripts/prepare_idd_dataset.py` + `scripts/train_and_export_indian_traffic.py` |

## 3. Ethical & Accuracy Boundaries
- COCO-80 models do **not** have trained weights for `auto_rickshaw`, `e_rickshaw`, `scooter`, `ambulance`, or helmets.
- DRISHTI AI V2 **never** relabels COCO `car` detections as `auto_rickshaw`.
- Low-confidence or ambiguous vehicle detections in the uncertain band are surfaced as `unknown_vehicle` rather than guessed.
- No facial recognition or personal identity tracking is performed.

