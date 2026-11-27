# AUDIT_REPORT.md — DRISHTI AI V1 -> V2 Engineering Audit

**Maintainer:** PALLEM DINESH KUMAR REDDY

| Original Problem | Root Cause | Modified / Added Files | Implemented Fix | Status |
|---|---|---|---|---|
| Only `yolox_nano.onnx` (416x416) bundled; hardcoded path/SHA/size in `frigate/drishti_video.py` | Single static `FrameDetector` class without model registry or multi-architecture decoders | `frigate/drishti/model_registry.py`, `frigate/drishti/detector.py`, `frigate/util/model.py`, `frigate/drishti_video.py` | Added verified model registry, `yolox_s.onnx` (640x640), `YOLOv8`/`YOLO11`/`YOLO26` decoders, configurable `model_id`, `input_size`, `score_threshold` | PASS |
| No Indian auto-rickshaw / e-rickshaw / scooter classes in `coco-80.txt` | COCO-80 weights lack Indian vehicle classes | `frigate/drishti/indian_traffic.py`, `demo/indian-17.txt`, `scripts/prepare_idd_dataset.py`, `scripts/train_and_export_indian_traffic.py` | Canonical 17-class taxonomy, `unknown_vehicle` uncertain state, explicit `unavailable_labels`, reproducible IDD training & ONNX export pipeline | PARTIAL (COCO-80 active; IDD weights require training on licensed dataset) |
| Per-frame box counts instead of unique tracks; class flickering & duplicate thumbnails | Separate per-label Norfair trackers and no deduplication in `drishti_video.py` | `frigate/drishti/tracker_analytics.py`, `frigate/drishti_video.py` | Unified `TrackAnalyticsEngine` with temporal class smoothing, multi-frame confirmation, IoU duplicate suppression, separated live vs unique counts, replay loop isolation | PASS |
| No temporal accident or collision-risk engine | Single-frame detection only | `frigate/drishti/accident_engine.py`, `frigate/drishti/collision_risk.py` | Multi-frame kinematic evidence fusion (`Normal` -> `Probable accident`), false-positive suppression (0 false alerts on `hyderabad.mp4`), separate homography-aware `CollisionRiskEngine` | PARTIAL (0 false alarms on normal traffic; real crash recall unmeasured without annotated crash dataset) |
| No Indian safety analytics or CCTV health classification | Missing safety modules and source classification | `frigate/drishti/safety_analytics.py`, `frigate/drishti/camera_health.py` | Modular triple-riding, stopped-vehicle, congestion, wrong-way (gated on lane config), explicit disabled flags for helmet/red-light/overspeeding; honest `LIVE CCTV` vs `VIDEO REPLAY` vs `CAMERA OFFLINE` | PASS |
| Upload workflow hardcoded 2 FPS, no progress/cancel/CSV | Synchronous/minimal `drishti_video.py` API | `frigate/drishti_video.py`, `frigate/api/drishti_video.py`, `frigate/api/drishti_intelligence.py`, `frigate/api/fastapi_app.py` | Configurable 1–15 FPS, progress & cancellation, `report.csv` + `report.json`, `/drishti/v2/*` intelligence, calibration, review & audit APIs | PASS |
| No V2 frontend intelligence, calibration, or diagnostics UI | Only `VideoAnalysis.tsx` existed | `web/src/pages/TrafficIntelligence.tsx`, `web/src/pages/VideoAnalysis.tsx`, `web/src/App.tsx`, `web/src/hooks/use-navigation.ts` | Added `/intelligence` console (6 tabs + direct routes) and upgraded `VideoAnalysis.tsx` with model/FPS selectors, cancel, unique tracks & CSV export | PASS |
| No pre-migration backup in `start.py` | `start.py` only checked config hash | `deploy/drishti/start.py` | Added timestamped backups of `config.yml` and `frigate.db` to `/config/backups/` before startup | PASS |

## ARM64 Regression Repair - 2026-11-27

| Finding | Root Cause | Fix | Validation |
|---|---|---|---|
| Moving-camera looming produced false accident evidence | Area expansion was calculated with an extreme first-box minimum and did not account for sustained growth or camera motion | Median the early area, require sustained recent area growth, and suppress high-occupancy looming when sustained camera motion is detected | New synthetic moving-camera test passes; normal track behavior remains unchanged |
| Convergence distance used incorrect coordinate pairing | Previous pair distance mixed `pb0.cy` with `pa0.cy` | Corrected the Euclidean distance to pair both previous centers consistently | Existing convergence and sudden-stop regression passes |
| Class labels flickered between vehicle classes | Weighted smoothing still allowed a single strong two-frame vote to replace a stable track label | Require at least three winning observations or retain the stable current label when only one or two frames disagree | New class-hysteresis regression passes |
| Native Windows tests could not import Frigate test package | Native Windows lacks the Unix `resource` module, while Docker/Linux test setup provides it | Direct module loading was used for focused native ARM64 verification | 13 relevant DRISHTI tests pass (11 V2 and 2 video) |

### Crash1 ARM64 Measurements

| Model | Median inference | End-to-end FPS | Incident records |
|---|---:|---:|---:|
| YOLOX-Nano 416 | 18.98 ms | 4.91 | 0 after moving-camera false-positive suppression |
| YOLOX-Tiny 416 | 62.28 ms | 3.17 | 1 in the prior baseline result |
| YOLOX-S 640 | 233.14 ms | 2.04 | 1 in the prior baseline result |

The crash1 measurements are decision-support evidence, not an annotated accuracy benchmark. The model weights are COCO-80-trained and do not provide dedicated Indian auto-rickshaw, e-rickshaw, helmet, or trained accident classes. External emergency dispatch remains blocked for uploaded clips.
