# WEBCAM_SETUP.md — Windows 11 ARM64 Laptop Webcam Setup (CAM-001)

**Maintainer:** PALLEM DINESH KUMAR REDDY

## Option A: Direct Browser Webcam (Recommended — Zero Host Dependencies)
1. Ensure DRISHTI AI V2 is running (`http://127.0.0.1:8971`).
2. Sign in and open **Human Safety Webcam** (`/human-safety`) from the sidebar.
3. Click **Start Camera** and grant Chrome/Edge permission to use your laptop's built-in webcam (`CAM-001`).
4. Live frames are captured from the webcam and analyzed in real time by the backend ONNX Runtime (`yolox-nano-416` + `movenet_lightning.onnx`).
5. Click **Stop Camera** to release the webcam hardware immediately.

## Option B: Native Windows Webcam Bridge (`scripts/webcam_bridge.py`)
For headless MJPEG/JPEG streaming (`http://127.0.0.1:8765`):
```powershell
cd DRISHTI-AI
.\scripts\start_webcam_bridge.bat
```
Endpoints: `/health`, `/devices`, `/snapshot.jpg`, `/stream.mjpg`, `POST /start`, `POST /stop`.

