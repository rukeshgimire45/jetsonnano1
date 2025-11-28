from __future__ import annotations

from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field


class Person(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    label: str = Field(index=True, unique=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_seen: Optional[datetime] = None
    total_detections: int = Field(default=0)


class UnknownFace(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    image_path: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    confidence: float = Field(default=0.0)


class DetectionEvent(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    label: str
    is_known: bool = Field(default=False)
    confidence: float = Field(default=0.0)
    image_path: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PersonVisit(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    label: str = Field(index=True)
    detection_in: datetime = Field(default_factory=datetime.utcnow)
    last_seen: datetime = Field(default_factory=datetime.utcnow)
    detection_out: Optional[datetime] = None
