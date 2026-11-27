# SOS_SYSTEM.md — Emergency SOS Workflow & Mandatory Test Mode

**Maintainer:** PALLEM DINESH KUMAR REDDY  
**Modules:** `frigate/drishti/sos_engine.py`, `web/src/pages/CommandCenter.tsx`, `web/src/pages/HumanSafety.tsx`

## 1. Safe Test Mode (Enabled by Default)
- Controlled by `DRISHTI_TEST_MODE` (defaults to `"true"` in backend).
- In TEST MODE:
  - Uses real camera frames and genuine ONNX AI inference.
  - Plays local audible alarm (Web Audio API) and runs configurable SOS countdown (`15s`).
  - Persists clearly tagged `TEST MODE` incidents and snapshots to `/config/drishti_v2.db`.
  - **Hard-blocks** all external SMS/call/WhatsApp/email dispatch (`notification_status = "Test only"`).
- Uploaded historical videos (`is_historical = True`) never trigger external emergency dispatch.

## 2. Operator Controls & Notification States
- Controls: `I Am Okay`, `Cancel Alert`, `Acknowledge`, `Confirm Emergency`, `Manual SOS`, `View Live Camera`, `View Evidence`.
- Notification states: `Not configured`, `Pending`, `Sending`, `Sent`, `Delivered`, `Failed`, `Test only`.
- Risk levels: `NORMAL`, `LOW`, `HIGH`, `CRITICAL` (describing the observed event, never profiling people).

