from __future__ import annotations

from typing import List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlmodel import select

from ..config import config
from ..database import session_scope
from ..events import event_bus
from ..models import DetectionEvent, PersonVisit
from ..schemas import DetectionEventResponse, PersonVisitResponse
from ..storage import encode_image_url

router = APIRouter(prefix=f"{config.api_prefix}/detections", tags=["detections"])


@router.get("", response_model=List[DetectionEventResponse])
def recent_detections(limit: int = 50) -> List[DetectionEventResponse]:
    with session_scope() as session:
        stmt = select(DetectionEvent).order_by(DetectionEvent.created_at.desc()).limit(limit)
        events = session.exec(stmt).all()
        return [
            DetectionEventResponse(
                id=e.id,
                label=e.label,
                confidence=e.confidence,
                is_known=e.is_known,
                created_at=e.created_at,
                image_url=encode_image_url(e.image_path) if e.image_path else None,
            )
            for e in events
        ]


@router.websocket("/ws")
async def detection_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        async for payload in event_bus.subscribe():
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        return


@router.get("/visits", response_model=List[PersonVisitResponse])
def list_visits(limit: int = 100) -> List[PersonVisitResponse]:
    with session_scope() as session:
        stmt = select(PersonVisit).order_by(PersonVisit.detection_in.desc()).limit(limit)
        visits = session.exec(stmt).all()
        responses: List[PersonVisitResponse] = []
        for visit in visits:
            end_time = visit.detection_out or visit.last_seen
            duration = (end_time - visit.detection_in).total_seconds() if end_time else None
            responses.append(
                PersonVisitResponse(
                    id=visit.id,
                    label=visit.label,
                    detection_in=visit.detection_in,
                    last_seen=visit.last_seen,
                    detection_out=visit.detection_out,
                    duration_seconds=duration,
                )
            )
        return responses
