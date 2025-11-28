from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class StatusResponse(BaseModel):
    worker_running: bool
    camera_uri: str
    hls_playlist: str


class PersonResponse(BaseModel):
    id: int
    label: str
    created_at: datetime
    last_seen: Optional[datetime]
    total_detections: int


class UnknownFaceResponse(BaseModel):
    id: int
    image_url: str
    created_at: datetime
    confidence: float


class LabelRequest(BaseModel):
    label: str


class DetectionEventResponse(BaseModel):
    id: int
    label: str
    confidence: float
    is_known: bool
    created_at: datetime
    image_url: Optional[str]


class PersonVisitResponse(BaseModel):
    id: int
    label: str
    detection_in: datetime
    last_seen: datetime
    detection_out: Optional[datetime]
    duration_seconds: Optional[float]
