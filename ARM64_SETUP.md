# ARM64_SETUP.md — Snapdragon X / Windows 11 ARM64 & Linux Deployment

**Maintainer:** PALLEM DINESH KUMAR REDDY

## 1. Verified Target Environment
- **Hardware:** HP OmniBook 3, Qualcomm Snapdragon X ARM64 (`aarch64`), 16 GB RAM (8.11 GB allocated to Docker/WSL2).
- **OS & Container Runtime:** Windows 11 ARM64 + Docker Desktop WSL 2 Linux engine.
- **Container Stack:** Python `3.11.2` (`aarch64`), ONNX Runtime `1.31.1` (`CPUExecutionProvider`), OpenCV `4.11.0`, Norfair `2.3.0`, FFmpeg `5.0`, FastAPI, React/Vite.

## 2. Start & Upgrade Commands
```powershell
cd DRISHTI-AI
docker compose -f compose.drishti.yml up --build -d
docker compose -f compose.drishti.yml ps
docker compose -f compose.drishti.yml logs nvr
```
- Access UI at `http://127.0.0.1:8971` (loopback-only by default).
- `deploy/drishti/start.py` automatically creates timestamped backups of `config.yml` and `frigate.db` in `/config/backups/` before startup and preserves custom configurations.

## 3. Performance Tuning on Snapdragon X CPU
- `OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=1` in `compose.drishti.yml`.
- `yolox-nano-416` is the recommended real-time CPU profile; `yolox-s-640` increases recall (`49.34% -> 74.34%` on COCO-20) at higher CPU latency.
- Qualcomm NPU (`QNNExecutionProvider`) is **not** assumed or enabled without verified runtime support; execution uses `CPUExecutionProvider`.

