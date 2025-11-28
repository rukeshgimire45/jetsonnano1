from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse

from ..config import config
from ..stream import stream_buffer

router = APIRouter(prefix=f"{config.api_prefix}/stream", tags=["stream"])


@router.get("/frame.jpg")
def latest_frame() -> Response:
    frame = stream_buffer.get_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="Stream not ready")
    return Response(content=frame, media_type="image/jpeg")


@router.get("/mjpeg")
async def mjpeg_stream() -> StreamingResponse:
    boundary = "frame"

    async def frame_generator():
        while True:
            frame = stream_buffer.get_frame()
            if frame is None:
                await asyncio.sleep(0.1)
                continue
            yield (
                b"--" + boundary.encode() + b"\r\n"
                + b"Content-Type: image/jpeg\r\n"
                + b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
                + frame + b"\r\n"
            )
            await asyncio.sleep(0.05)

    return StreamingResponse(
        frame_generator(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
    )
