@echo off
echo Starting DRISHTI AI V2 Windows Webcam Bridge (CAM-001)...
cd /d "%~dp0.."
python -c "import cv2" 2>nul
if errorlevel 1 (
    echo Installing opencv-python-headless for native Windows webcam capture...
    python -m pip install opencv-python-headless
)
python scripts\webcam_bridge.py --host 127.0.0.1 --port 8765 --device 0
pause

