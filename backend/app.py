from __future__ import annotations

import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import config
from .database import init_db
from .events import event_bus
from .routers import status, faces, detections, stream
from .services.detection_worker import detection_worker

app = FastAPI(title="Jetson Nano WebApp", version="0.1.0")

# Allow LAN + dev origins (adjust later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"]
    ,
    allow_headers=["*"],
)

app.include_router(status.router)
app.include_router(faces.router)
app.include_router(detections.router)
app.include_router(stream.router)

app.mount("/media", StaticFiles(directory=str(config.root_dir)), name="media")
app.mount(config.hls_mount, StaticFiles(directory=str(config.hls_dir)), name="hls")


@app.on_event("startup")
async def on_startup() -> None:
    init_db()
    event_bus.attach_loop(asyncio.get_running_loop())
    detection_worker.start()


@app.on_event("shutdown")
async def on_shutdown() -> None:
    detection_worker.stop()
