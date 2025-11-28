# Jetson Nano WebApp

This repository now hosts a full-stack monitoring/web application for the Jetson Nano. The Nano handles camera capture, YOLO/MediaPipe inference, face recognition, image storage, and exposes REST/WebSocket endpoints that a React web client consumes. The UI offers live video via HLS, recent detections, an unknown-face labeling queue, and a people directory.

## Stack Overview

- **Backend:** FastAPI + SQLModel + SQLite, runs on the Jetson. A background worker (YOLOv8 + MediaPipe) writes detections to disk/DB and pushes events over a WebSocket bus. Static endpoints expose HLS playlists and stored media.
- **Frontend:** React (Vite + TypeScript). Consumes `/api/v1` endpoints, plays `stream.m3u8` through `hls.js`, and manages labeling workflows.
- **Data/Storage:** Face crops live under `data/faces_enroll/` (labeled) and `data/faces_unknown/` (queue). The worker moves files between these directories when unknown faces are labeled.

## Getting Started

1. **Install Python dependencies** (ideally inside your Jetson virtualenv):
	```bash
	pip install -r requirements.txt
	```
2. **Start the backend** (serves API + websockets + static files):
	```bash
	uvicorn backend.app:app --host 0.0.0.0 --port 8000
	```
	The detection worker launches automatically on startup. Ensure your camera is available at `v4l2:///dev/video0` or override `JETSON_CAMERA_URI`.
3. **Produce the HLS playlist:** run your preferred ffmpeg/GStreamer pipeline so that `hls/stream.m3u8` and segment files are continuously updated. (See `docs/architecture.md` for ideas.)
4. **Run the web client** during development:
	```bash
	cd web
	npm install
	npm run dev
	```
	The Vite dev server proxies `/api`, `/media`, and `/hls` to the backend on port 8000.
5. **Build frontend for production:**
	```bash
	npm run build
	```
	Serve the generated `web/dist` directory via nginx or FastAPI’s `StaticFiles` (see architecture notes).

## Admin Workflow

1. Browse to the Live Monitor page to view the HLS feed and incoming detection events.
2. As the worker finds new faces it cannot match, thumbnails appear in the **Unknown Faces** grid.
3. Click **Label** on a tile, enter a name, and the backend moves the file into `data/faces_enroll/<label>/`, updates the embedding library, and removes the snapshot from the queue.
4. The person immediately shows up in the **People Directory** and subsequent detections are tagged with their name in real time.

See `docs/architecture.md` for a detailed breakdown of services, data flow, and deployment considerations.
