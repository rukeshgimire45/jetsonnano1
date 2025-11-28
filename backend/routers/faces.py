from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException
from sqlmodel import select

from ..config import config
from ..database import session_scope
from ..events import event_bus
from ..models import DetectionEvent, Person, UnknownFace
from ..schemas import LabelRequest, PersonResponse, UnknownFaceResponse
from ..storage import delete_file, encode_image_url, move_to_enroll
from ..services.face_registry import face_registry

router = APIRouter(prefix=f"{config.api_prefix}/faces", tags=["faces"])


@router.get("/unknown", response_model=List[UnknownFaceResponse])
def list_unknown_faces(limit: int = 50) -> List[UnknownFaceResponse]:
    with session_scope() as session:
        stmt = select(UnknownFace).order_by(UnknownFace.created_at.desc()).limit(limit)
        entries = session.exec(stmt).all()
        return [
            UnknownFaceResponse(
                id=entry.id,
                image_url=encode_image_url(Path(entry.image_path)),
                created_at=entry.created_at,
                confidence=entry.confidence,
            )
            for entry in entries
        ]


@router.post("/unknown/{face_id}/label", response_model=PersonResponse)
def label_unknown_face(face_id: int, payload: LabelRequest) -> PersonResponse:
    event_payloads: list[dict] = []
    with session_scope() as session:
        entry = session.get(UnknownFace, face_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Unknown face not found")
        image_path = Path(entry.image_path)
        if not image_path.exists():
            session.delete(entry)
            raise HTTPException(status_code=404, detail="Image file missing")

        dest_path = move_to_enroll(image_path, payload.label)
        face_registry.add_from_path(payload.label, dest_path)

        session.delete(entry)

        events = session.exec(select(DetectionEvent).where(DetectionEvent.image_path == str(image_path))).all()
        for event in events:
            event.label = payload.label
            event.is_known = True
            event.image_path = str(dest_path)
            if event.confidence == 0.0:
                event.confidence = 0.99
            session.add(event)
        event_payloads = [
            {
                "id": event.id,
                "label": event.label,
                "confidence": event.confidence,
                "is_known": event.is_known,
                "created_at": event.created_at.isoformat(),
                "image_url": encode_image_url(dest_path),
            }
            for event in events
        ]

        person = session.exec(select(Person).where(Person.label == payload.label)).first()
        if person is None:
            person = Person(label=payload.label)
        person.total_detections += 1
        person.last_seen = person.last_seen or entry.created_at
        person.updated_at = entry.created_at
        session.add(person)
        session.flush()

        response = PersonResponse(
            id=person.id,
            label=person.label,
            created_at=person.created_at,
            last_seen=person.last_seen,
            total_detections=person.total_detections,
        )

    for payload_dict in event_payloads:
        event_bus.publish_from_thread(payload_dict)

    return response


@router.delete("/unknown/{face_id}")
def delete_unknown_face(face_id: int) -> dict:
    with session_scope() as session:
        entry = session.get(UnknownFace, face_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Unknown face not found")
        delete_file(Path(entry.image_path))
        session.delete(entry)
        return {"status": "deleted"}


@router.get("/people", response_model=List[PersonResponse])
def list_people() -> List[PersonResponse]:
    with session_scope() as session:
        stmt = select(Person).order_by(Person.label.asc())
        people = session.exec(stmt).all()
        return [
            PersonResponse(
                id=p.id,
                label=p.label,
                created_at=p.created_at,
                last_seen=p.last_seen,
                total_detections=p.total_detections,
            )
            for p in people
        ]
