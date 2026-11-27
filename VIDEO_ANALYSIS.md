# VIDEO_ANALYSIS.md — Unified Video Upload Intelligence

**Maintainer:** PALLEM DINESH KUMAR REDDY  
**Modules:** `frigate/drishti_video.py`, `frigate/api/drishti_video.py`, `web/src/pages/VideoAnalysis.tsx`

- **Unified Pipeline:** Uses the exact same `UnifiedDetector`, `TrackAnalyticsEngine`, `TemporalAccidentEngine`, `CollisionRiskEngine`, `SafetyAnalyticsEngine`, `MoveNetPoseEstimator`, `TemporalFallEngine`, and `PublicSafetyEngine` as live cameras.
- **Configurable Limits & Sampling:** Supports `DRISHTI_MAX_UPLOAD_BYTES` (default `32 MiB`), `DRISHTI_MAX_UPLOAD_SECONDS` (default `120s`), `DRISHTI_MAX_RETAINED_ANALYSES` (default `12`), selectable `model_id`, and `sample_fps` (`1–15 FPS`).
- **Outputs:** Annotated `review.mp4` (with track IDs, trajectories, and pose skeletons), structured `report.json`, and tabular `report.csv`, plus live progress and job cancellation (`POST /api/drishti/analysis/cancel`).
- **Historical Isolation:** All uploaded video reports are flagged `is_historical: true` and `sos_dispatch_blocked: true`.

