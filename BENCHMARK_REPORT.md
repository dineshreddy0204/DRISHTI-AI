# BENCHMARK_REPORT.md — Measured ARM64 CPU Performance

**Maintainer:** PALLEM DINESH KUMAR REDDY  
**Hardware / Runtime:** `aarch64` (Snapdragon X Docker WSL2, 8 vCPUs, 8.11 GB RAM), ONNX Runtime `1.31.1` (`CPUExecutionProvider`), OpenCV `4.11.0`, FFmpeg `5.0`.  
**Source Data:** [validation/benchmark-results.json](validation/benchmark-results.json)

## 1. COCO-20 Validation Subset (20 Images, Match IoU 0.5, Score 0.4)

| Model ID | Input Size | TP | FP | FN | Ignored Crowd | Precision | Recall | Median Latency | p95 Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `yolox-nano-416` | 416 x 416 | 75 | 12 | 77 | 11 | 86.21% | 49.34% | 98.89 ms | 169.31 ms |
| `yolox-s-640` | 640 x 640 | 113 | 9 | 39 | 26 | 92.62% | 74.34% | 1660.07 ms | 2394.91 ms |

## 2. Hyderabad 20s Traffic Clip (`demo/hyderabad.mp4`, 5 FPS -> 100 Frames, 640x360)

| Model ID | Per-Frame Boxes | Unique Confirmed Tracks | Median Inference | p95 Inference | End-to-End Pipeline FPS | Wall Time | False Accident Alerts |
|---|---:|---:|---:|---:|---:|---:|---:|
| `yolox-nano-416` | 734 | 47 (`29 person, 9 motorcycle, 8 unknown_vehicle, 1 car`) | 114.81 ms | 190.13 ms | 3.03 FPS | 32.95 s | 0 |
| `yolox-s-640` | 1315 | 75 (`37 person, 10 motorcycle, 15 unknown_vehicle, 8 backpack, 4 car, 1 banana`) | 2115.22 ms | 2951.53 ms | 0.40 FPS | 248.03 s | 0 |

## 3. Key Findings
- Upgrading from `yolox-nano-416` to `yolox-s-640` improves COCO-20 recall from **49.34% to 74.34%** (+25.0 pp) and precision from **86.21% to 92.62%**, at the cost of ~16.8x higher CPU inference latency on ARM64.
- `yolox-nano-416` remains the recommended real-time CPU default; `yolox-s-640` is selectable for higher-accuracy offline upload analysis.
- Target of 10–15 end-to-end processed FPS is **not** achieved on full-pipeline CPU inference with `yolox-nano-416` (measured 3.03 FPS end-to-end at 5 FPS sampling including FFmpeg decode/encode, tracking, accident/risk/safety fusion).

