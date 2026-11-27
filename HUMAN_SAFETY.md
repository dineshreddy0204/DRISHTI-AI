# HUMAN_SAFETY.md — Pose Estimation & Temporal Fall Detection

**Maintainer:** PALLEM DINESH KUMAR REDDY  
**Modules:** `frigate/drishti/pose_estimator.py`, `frigate/drishti/fall_detection.py`, `web/src/pages/HumanSafety.tsx`

## 1. Pose Estimation (`MoveNetPoseEstimator`)
- **Model:** MoveNet SinglePose Lightning (`demo/movenet_lightning.onnx`, Apache-2.0, `192x192` `int32` RGB).
- **SHA-256:** `1ad4f8d6c2f776a9967db3993c9ca740bc350104f9d37c151dc183fc29a464ad`
- **Conditional Execution:** Runs only when a `person` track is detected by `UnifiedDetector` (`yolox-nano-416`).
- **Extracted Features:** 17 COCO body keypoints, torso angle relative to vertical, hip vertical position, posture (`standing`, `sitting`, `bending`, `lying_horizontal`, `insufficient_evidence`), and keypoint reliability (`good`, `reduced`, `insufficient_evidence`).

## 2. Temporal Fall Detection (`TemporalFallEngine`)
- **States:** `NORMAL`, `POTENTIAL_FALL`, `SUSPECTED_FALL`, `PROLONGED_INACTIVITY`, `SOS_PENDING`, `SOS_ACTIVATED`, `RECOVERED`, `HUMAN_CONFIRMED_EMERGENCY`, `DISMISSED`.
- **False-Positive Suppression:** Horizontal posture alone never confirms a fall; sitting, bending, and slow lying down are suppressed. Requires rapid downward hip displacement (`>= 0.22/s`) + upright-to-horizontal transition (`<= 2.5s`) across `confirm_frames >= 2`.
- **Non-Diagnostic Notice:** DRISHTI AI does not diagnose medical conditions (e.g., unconsciousness, cardiac events) from camera video.

