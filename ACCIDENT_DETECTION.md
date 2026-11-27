# ACCIDENT_DETECTION.md — Temporal Incident & Collision-Risk Architecture

**Maintainer:** PALLEM DINESH KUMAR REDDY  
**Modules:** `frigate/drishti/accident_engine.py`, `frigate/drishti/collision_risk.py`

## 1. Design Principles
- **No single-frame bounding-box accident claims:** Bounding-box overlap (`IoU`) alone never triggers an accident alert.
- **Multi-frame kinematic fusion (`TemporalAccidentEngine`):** Requires at least `min_history_frames = 7` trajectory points per confirmed track and `confirm_frames = 3` consecutive frames of corroborating evidence.
- **Evidence signals:**
  - Trajectory convergence (`closing_speed >= 30 px/s`)
  - Proximity / contact (`IoU >= 0.15` or `edge_gap <= 8 px`)
  - Sustained pre-impact motion (`38 <= pre_speed <= 115 px/s`) followed by abrupt deceleration (`>= 72%` drop to `< 12 px/s`)
  - Unexpected post-contact stop (`< 8 px/s`)
  - Single-vehicle rollover / rider fall (`aspect_ratio` shift `>= 1.75x` paired with sudden deceleration)

## 2. False-Positive Suppression
- **Tracker jump guard:** Speeds above `115 px/s` are rejected as track re-association jumps rather than physical motion.
- **Crowded intersection / jam suppression:** Scenes with `>= 6` slow/stationary vehicles (`< 14 px/s`) suppress normal stop-and-go braking alerts.
- **Stop-zone suppression:** Deceleration inside configured `stop_zones` is suppressed.
- **Cooldown & hysteresis:** `15.0s` cooldown per track pair prevents duplicate alerts.

## 3. Incident States & Evidence Score
- States: `Normal` -> `Potential risk` -> `Suspected collision` -> `Probable accident` -> `Human-confirmed accident` / `Dismissed incident`.
- The `evidence_score` (`0.0` to `0.95`) is a deterministic weighted sum of kinematic cues, **not** a calibrated accident probability.

## 4. Predictive Collision Risk (`CollisionRiskEngine`)
- Estimates Time-to-Collision (`TTC`), Post-Encroachment Time (`PET`), and near-miss events from converging velocity vectors.
- When camera homography (`4+` image-to-ground-plane points) is configured, metrics are reported in `ground_meters`.
- Without homography calibration, metrics are explicitly labeled `image_pixels_uncalibrated`; physical `km/h` and meters are never fabricated.

