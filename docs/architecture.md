# Jetson Nano WebApp Architecture

## Overview
Jetson Nano acts as the edge server that: 
- Captures the camera feed and runs YOLOv8 + MediaPipe-based face recognition/embedding.
- Writes labeled face crops under `data/faces_enroll/<label>/` and unknown faces under `data/faces_unknown/`.
- Exposes a REST + WebSocket API via FastAPI for the web client.
- Serves an HTTP Live Streaming (HLS) playlist produced by ffmpeg/GStreamer.
- Sends notifications (future) when detections match configured rules.

A React single-page app (SPA) connects over HTTPS to watch the live stream, review detections, and label unknown faces.

## Backend Components

| Component | Purpose |
| --- | --- |
| `backend/app.py` | FastAPI application wiring routers, storage, and startup tasks. |
| `backend/config.py` | Central configuration (paths, HLS settings, device names). |
| `backend/database.py` | SQLModel session factory on SQLite (`app.db`). |
| `backend/models.py` | SQLModel tables for `Person`, `UnknownFace`, `DetectionEvent`. |
| `backend/schemas.py` | Pydantic response models consumed by the frontend. |
| `backend/storage.py` | Helpers for saving/moving face crops and exposing media URLs. |
| `backend/events.py` | Async pub/sub (queues) for pushing detection events to WebSocket clients. |
| `backend/worker.py` | Background detection loop: captures frames, runs YOLO + embeddings, writes to storage/db, notifies event bus. |
| `backend/routers/*.py` | REST/WebSocket endpoints (`status`, `faces`, `media`, `detections`). |

### Detection Flow
1. `DetectionWorker` grabs frames with `jetson_utils.videoSource`.
2. YOLO detects persons; MediaPipe Face Mesh + Face Embedder compute embeddings.
3. Matches update `Person` last_seen + emit events. Unrecognized faces saved to `data/faces_unknown/` and inserted into DB.
4. Successful label actions move the image to `faces_enroll/<label>/`, regenerate embeddings, update DB, and remove from unknown queue.

### Streaming Flow
- Separate `scripts/run_hls_stream.sh` invokes `gst-launch-1.0` or `ffmpeg` to publish HLS segments under `hls/`.
- FastAPI serves `/stream.m3u8` by exposing that directory with `StaticFiles`.
- React client uses `hls.js` to play it.

## Frontend (React + Vite)

Pages:
1. **Live Monitor** – HLS player + real-time detection list via WebSocket (`/api/v1/detections/ws`).
2. **Unknown Faces** – Grid/table of pending unlabeled faces, each with thumbnail, timestamp, and "Assign label" dialog calling `POST /api/v1/unknown-faces/{id}/label`.
3. **People Directory** – Table showing labeled individuals, total captures, and last seen timestamp; supports removing/renaming (future).
4. **Settings** – Placeholder for stream URL, notification toggles, and (later) authentication controls.

Data fetching uses Axios; application state managed with React Query for caching/invalidation. Styling via Tailwind CSS for quick iteration.

## Deployment Notes
- `uvicorn backend.app:app` launched via systemd on Jetson, optionally behind nginx.
- nginx proxies `/api` and `/ws` to FastAPI, `/hls` to static playlist directory, and serves the built React app from `web/dist`.
- HTTPS termination handled by nginx + Let's Encrypt (if exposed outside LAN).
- Authentication layer (future) will add JWT-protected routes and hashed admin credentials stored in SQLite.
