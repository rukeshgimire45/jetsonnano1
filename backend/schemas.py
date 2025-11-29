from __future__ import annotations

from datetime import datetime
from typing import Optional, List
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


class PersonActivityResponse(BaseModel):
    id: int
    label: str
    visit_id: Optional[int]
    activity: Optional[str]
    confidence: float
    needs_review: bool
    clip_url: Optional[str]
    window_start: datetime
    window_end: datetime
    created_at: datetime
    updated_at: datetime
    actions: List["ActivityActionResponse"]


class ActivityUpdateRequest(BaseModel):
    activity: Optional[str] = None
    confidence: Optional[float] = None
    needs_review: Optional[bool] = None


class ActivityActionResponse(BaseModel):
    id: int
    activity_id: int
    action_type: str
    confidence: float
    start_offset: Optional[float]
    end_offset: Optional[float]
    created_at: datetime
    updated_at: datetime


class ActivityActionCreateRequest(BaseModel):
    activity_id: int
    action_type: str
    confidence: Optional[float] = None
    start_offset: Optional[float] = None
    end_offset: Optional[float] = None


class ActivityActionUpdateRequest(BaseModel):
    action_type: Optional[str] = None
    confidence: Optional[float] = None
    start_offset: Optional[float] = None
    end_offset: Optional[float] = None
