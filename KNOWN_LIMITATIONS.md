# KNOWN_LIMITATIONS.md — Honest Scope & Remaining Blockers

**Maintainer:** PALLEM DINESH KUMAR REDDY

1. **Indian-specific object classes (`auto_rickshaw`, `e_rickshaw`, `scooter`, `ambulance`, etc.):**
   - Unavailable in bundled COCO-80 weights (`yolox_nano.onnx`, `yolox_s.onnx`).
   - Marked explicitly as `unavailable_labels` in `frigate/drishti/model_registry.py` until trained on licensed IDD data using `scripts/prepare_idd_dataset.py` and `scripts/train_and_export_indian_traffic.py`.
2. **Helmet detection:**
   - Disabled in `SafetyAnalyticsEngine` because no trained helmet-classification ONNX model is bundled.
3. **Real-world accident dataset validation:**
   - No annotated real-world Indian crash video dataset is bundled in the repository.
   - `TemporalAccidentEngine` is verified against synthetic kinematic collision/overlap/normal-braking tests and verified to produce zero false accident alerts on normal Hyderabad traffic (`demo/hyderabad.mp4`). Real crash recall/precision remains unmeasured until tested on annotated crash footage.
4. **Uncalibrated cameras:**
   - Without a 4-point ground-plane homography in `/drishti/v2/calibration/{camera_id}`, `CollisionRiskEngine` operates in `image_pixels_uncalibrated` space and does not report physical `km/h`, meters, or overspeeding.
5. **Wrong-way & red-light analytics:**
   - Explicitly disabled unless `lane_direction_deg`, `signal_state`, and `stop_line_y` are configured for the camera.
6. **Hardware acceleration:**
   - Runs on ONNX Runtime `CPUExecutionProvider`; no NVIDIA CUDA or Qualcomm NPU execution provider is active.
7. **Legal & emergency notice:**
   - All alerts are decision-support only and require human review; the system never contacts emergency services automatically.

