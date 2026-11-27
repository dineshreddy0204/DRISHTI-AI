# VALIDATION_REPORT.md — DRISHTI AI V2 Verification Evidence

**Maintainer:** PALLEM DINESH KUMAR REDDY

## 1. Automated Test Results
- `python3 -m unittest frigate.test.test_drishti_video frigate.test.test_drishti_v2`: **11/11 tests PASSED** (`0.026s`) inside the ARM64 Docker container.
  - Verified invalid media rejection and below-threshold YOLOX empty-NMS handling.
  - Verified Indian taxonomy honest mapping (never relabeling `car` to `auto_rickshaw`, emitting `unknown_vehicle` for uncertain detections).
  - Verified `YOLOv8/11` and `YOLO26` NMS-free decoders.
  - Verified model registry availability & missing-weights status.
  - Verified `TrackAnalyticsEngine` duplicate suppression, multi-frame confirmation, and replay-loop isolation.
  - Verified `TemporalAccidentEngine` suppresses bounding-box overlap alone & normal traffic, while detecting multi-frame convergence + contact + sudden deceleration.
  - Verified `CollisionRiskEngine` uncalibrated vs calibrated homography modes.
  - Verified `SafetyAnalyticsEngine` explicit disabled flags for missing helmet/lane config.
  - Verified `classify_camera_source` (`VIDEO REPLAY`) and `redact_stream_url` credential masking.

## 2. Normal-Traffic False-Positive Verification
- Tested `demo/hyderabad.mp4` (100 sampled frames at 5 FPS) with both `yolox-nano-416` and `yolox-s-640`:
  - `incidents_detected = 0` for both models (PASS — normal traffic does not trigger false accident alerts).

## 3. Frontend & Container Verification
- `tsc && vite build --base=/BASE_PATH/` compiled cleanly inside Docker (`dist/` generated).
- Container `drishti-ai-nvr-1` (`drishti-ai:ce5b26d-1`) running on `127.0.0.1:8971` (`aarch64`).

## 4. Incomplete Validation Scope
- Real-world annotated Indian accident video dataset precision/recall and IDD multi-class mAP50/mAP50-95 remain **incomplete** until trained weights and annotated crash footage are supplied.

