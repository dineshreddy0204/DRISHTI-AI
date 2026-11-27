# Release validation

## Source and deployment audit

The complete upstream archive contains 3,336 files and two symbolic links among them. Every upstream source path was accounted for in the India Edition. Archive metadata and content comparisons identified 103 changed files, including backend, auth, detector and OS-specific changes. The India Edition also contained Windows packages, compiled caches, old builds, databases, a JWT signing secret, machine-specific configuration and duplicate media. The release starts with the upstream tree and selectively brings in the DRISHTI assets and regional text. Root license and trademark text remain byte-identical to the upstream ZIP. The moved manifest has a single source in `web/public/`.

The India Edition had hardcoded Windows paths, lost Unix executable bits on service scripts, an unsafe filename-based upload handler, a 15-byte file named yolov8n.onnx, and a build that could serve a UI without a reverse proxy. The exact process previously listening on the user's port 5000 cannot be identified from a ZIP. The release gives nginx, HTTP APIs, WebSockets and the UI one authenticated origin. The same-commit base image supplies the native software, and the release copies its own application source, built frontend and configuration into the runtime.

## Executed checks

| Check | Evidence |
|---|---|
| Production frontend | Clean locked npm install, TypeScript compile, Vite production bundle, Monaco workers and icons produced. The compiler reported its upstream large-chunk advisory. |
| Deployment configuration | Docker Compose v2.40.3 accepted the file with `config -q`. The default port maps host loopback to authenticated container port 8971. Both Linux target digests were resolved in the registry during the audit. |
| Running application | Fresh x86_64 Linux guest booted the source and frontend over the native libraries from the pinned amd64 image. Upstream s6, nginx, go2rtc, FastAPI, camera subprocesses and CPU detector started. Docker Engine was unavailable on the build host, so this run uses the unpacked OCI filesystem inside a Linux VM. |
| Original browser-console failures | Before login, config returned 401. After login, config, profile, profiles, notices, stats and notices/muted_checks returned JSON and HTTP 200. The WebSocket handshake returned 101 with a valid accept header. All four manifest resources returned PNG bytes and the three language resources loaded. |
| Upstream regression tests | 128 tests passed across FFmpeg presets, output/WebSocket authorization, Norfair tracker modes, record retention, notices and HTTP authentication. A follow-up run of nine upstream suites including WebSocket origins, the notice facade and proxy authentication passed 152 tests. These runs overlap and their counts are reported separately. |
| Fork-specific tests | Two regressions passed for rejecting invalid media and suppressing below-threshold YOLOX predictions when OpenCV returns an empty NMS tuple. An additional real-route integration test checked admin and viewer access, malformed input, filename handling, actual model inference, saved history, JSON and playable MP4 bytes. |
| API specification | Generated from the runtime routers, including admin-only upload routes. The upstream generator's `--check` passed. |
| Reference video | Included Hyderabad clip, 20 seconds. The packaged release-runtime reference run sampled 40 frames, emitted 247 boxes and encoded a 20-second 640 × 360 H.264 review. Per-label counts and file digests are in the JSON. |
| Native CPU timing | Linux x86_64, ONNX Runtime 1.31.1, CPUExecutionProvider, two inference threads. Median 8.64 ms and p95 19.31 ms for model calls in the packaged run, excluding decode, preprocessing, NMS, drawing and encoding. Timing varies with load. |

A box records a candidate object in a single sampled frame. The same object can produce boxes in multiple frames. The final NVR upload and an independent native Linux run with the pinned release image agreed on all per-class counts and the review file size. Both used the same input SHA-256. The separately installed workstation stack (ONNX Runtime 1.32.0 and host FFmpeg) returned 254 boxes on that input. The runtime stacks differ. These tests did not isolate which dependency caused the difference.


## Replay and upload validation

Frigate's default motion threshold of 30 did not finish calibration on this crowded moving scene during the two-loop component check. An unmasked threshold of 120 produced camera recordings, but the observed NVR event and review lists stayed empty. The packaged replay uses a central roadway band from 28–64% of the width and 44–54% of the height and a pixel threshold of 45. This controls where motion triggers inference. The object detector can search a larger region around that motion.

Over 60 frames, Frigate's motion detector returned calibrated motion on 60 frames. The reproducible fixed-region check observed 8 car, 76 motorcycle, 162 person per-frame boxes and 1 car, 2 motorcycle, 11 person tracker IDs. Tracks may split or merge. See `validation/replay-detector.json`. The running NVR independently selected its own regions, created events, and stored reviews and recordings.

The full service ran from an empty config and media store in a fresh x86_64 Linux VM. The test image included diagnostic logging of motion state, region counts and proposal counts. Application logic and media came from the release. The live camera check observed 3 event rows, 1 review rows, downloaded an actual person snapshot (67,628 JPEG bytes) and a recording segment clip (722,811 MP4 bytes). Counts represent a point in the loop, not a fixed performance target. The startup configuration installer was also tested with an empty directory, both pristine prior defaults, a custom config, the alternate config.yaml filename and a restricted file mode. The packaged command independently checked a stored event flagged as having a snapshot. See `validation/nvr-events.json`, `validation/nvr-http.json` and `validation/nvr-event-snapshot.jpg`.

On this software-emulated CPU the camera captured 3.1 fps, processed 0.1 fps and skipped 3.0 fps during this sample. The recording maintainer also warned of a cache backlog during concurrent analysis. Some recording segments can be dropped when the VM cannot keep up. These timings are specific to the VM. The separate native CPU inference benchmark excludes the rest of the pipeline.

The final authenticated full-file upload processed the 20-second Hyderabad input and returned 40 sampled frames, 247 boxes and a 2,173,783-byte MP4 download. The downloaded report matched the input SHA-256. See `validation/nvr-upload.json`.


The earlier Linux NVR run produced 50 recording segments and returned a valid 942,898-byte MP4 clip from the recordings endpoint. Chromium verified login, navigation from Exports and rendering of the upload page in all three languages. Its simultaneous second upload was correctly rejected with HTTP 409. The original synchronous full upload exceeded the six-minute client timeout in CPU emulation.

The updated UI submits a background job, receives HTTP 202, and polls its persisted status. Processing continues when the user navigates away. Work runs in a separate CPU process. On native Linux x86_64, the FastAPI integration test used both modes with the full real Hyderabad clip, downloaded the artifacts, and checked role restrictions, malformed media, conflict handling and interrupted-job detection. The combined integration and detector regression run passed all three tests. Docker’s Moby build-context matcher confirmed the replay clip, model, regional checker, frontend entry points and required runtime paths are available to the build. The production React build and generated API specification were checked again. 

A separate native Linux test started Uvicorn on a loopback socket, uploaded the entire 20-second clip, disconnected, and retrieved its completed 40-frame report and MP4 through a new client. This regression runs as `python -m unittest validation.test_upload_http` inside the release runtime with the source tree available.

In a fresh x86_64 Linux VM running the updated services, Chromium signed in, followed the Exports link, rendered the workspace in Telugu, Hindi and English, switched to Telugu using Settings → Languages, confirmed the selection survived a reload, and uploaded a one-second excerpt of the included Hyderabad video. The endpoint returned HTTP 202. The browser navigated away, reopened the page, and showed the finished downloads. The JSON matched the uploaded file's SHA-256. It contained two sampled frames and 14 boxes (six person, five motorcycle, three car). The downloaded review contained 97,851 bytes. All six original authenticated HTTP checks passed again, along with the WebSocket and manifest checks. See `validation/browser-results.json` and `validation/nvr-http.json`.

## Labelled evaluation

For each of the 20 COCO validation images already present in the upstream archive, the evaluator runs YOLOX-Nano, letterboxes the image to 416 × 416 BGR, decodes the raw head with Frigate, and matches each score-sorted prediction to at most one ground-truth box of the same class. Minimum confidence is 0.4, NMS IoU is 0.4 and match IoU is 0.5. Unmatched predictions overlapping a same-class crowd region by at least half their own area are ignored. The decoder retains up to 20 predictions per image across all classes. Only the six road-user classes contribute to these totals.

| Metric | Measured |
|---|---:|
| True positives | 75 |
| False positives | 12 |
| Missed labelled objects | 77 |
| Predictions ignored in crowd regions | 11 |
| Micro precision | 86.21% |
| Micro recall | 49.34% |

This is a fixed-threshold check on a small calibration subset. It gives no population-level estimate, AP, tracking accuracy, crash risk or false-alarm rate in an Indian deployment. Class-level results, including sparse classes, are in `validation/coco20-results.json`. Run `python validation/evaluate_coco.py` to reproduce them.

## Scope still open

Snapdragon X latency, Windows ARM64 Docker Desktop startup, hardware CCTV, ONVIF controls, long-duration retention, concurrent-user capacity and production reliability need target-device validation. Linux camera integrations remain in the inherited source. Browser push alerts need HTTPS or a secure loopback context, client permission and deployment-specific configuration. The release does not add specialised safety detectors or an enterprise incident-resolution/audit workflow. Hindi and Telugu cover the upload workspace and selected existing screens; untranslated upstream strings fall back to English.
