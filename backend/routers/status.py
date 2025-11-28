from __future__ import annotations

from fastapi import APIRouter

from ..config import config
from ..schemas import StatusResponse
from ..storage import ensure_hls_playlist
from ..services.detection_worker import detection_worker

router = APIRouter(prefix=f"{config.api_prefix}/status", tags=["status"])


@router.get("", response_model=StatusResponse)
def get_status() -> StatusResponse:
    playlist = ensure_hls_playlist()
    playlist_url = f"{config.hls_mount}/{config.hls_playlist}" if playlist else ""
    return StatusResponse(
        worker_running=detection_worker.running,
        camera_uri=config.camera_uri,
        hls_playlist=playlist_url,
    )
