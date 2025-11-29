from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from sqlmodel import select

from ..config import config
from ..database import session_scope
from ..models import PersonActivity, ActivityAction
from ..schemas import (
    PersonActivityResponse,
    ActivityUpdateRequest,
    ActivityActionResponse,
)
from ..storage import encode_image_url
from ..services.activity_recognizer import activity_recognizer

router = APIRouter(prefix=f"{config.api_prefix}/activities", tags=["activities"])


def _serialize_actions(actions: List[ActivityAction]) -> List[ActivityActionResponse]:
    return [
        ActivityActionResponse(
            id=action.id,
            activity_id=action.activity_id,
            action_type=action.action_type,
            confidence=action.confidence,
            start_offset=action.start_offset,
            end_offset=action.end_offset,
            created_at=action.created_at,
            updated_at=action.updated_at,
        )
        for action in actions
    ]


def _to_response(activity: PersonActivity, actions: Optional[List[ActivityAction]] = None) -> PersonActivityResponse:
    return PersonActivityResponse(
        id=activity.id,
        label=activity.label,
        visit_id=activity.visit_id,
        activity=activity.activity,
        confidence=activity.confidence,
        needs_review=activity.needs_review,
        clip_url=encode_image_url(activity.clip_path) if activity.clip_path else None,
        window_start=activity.window_start,
        window_end=activity.window_end,
        created_at=activity.created_at,
        updated_at=activity.updated_at,
        actions=_serialize_actions(actions or []),
    )


@router.get("", response_model=List[PersonActivityResponse])
def list_activities(
    label: Optional[str] = None,
    needs_review: Optional[bool] = None,
    limit: int = 50,
) -> List[PersonActivityResponse]:
    with session_scope() as session:
        stmt = select(PersonActivity)
        if label:
            stmt = stmt.where(PersonActivity.label == label)
        if needs_review is not None:
            stmt = stmt.where(PersonActivity.needs_review.is_(needs_review))
        stmt = stmt.order_by(PersonActivity.created_at.desc()).limit(limit)
        activities = session.exec(stmt).all()
        action_map = _load_actions_for(session, [a.id for a in activities if a.id is not None])
        return [_to_response(a, action_map.get(a.id, [])) for a in activities]


@router.get("/unknown", response_model=List[PersonActivityResponse])
def list_unknown_activities(limit: int = 25) -> List[PersonActivityResponse]:
    with session_scope() as session:
        stmt = (
            select(PersonActivity)
            .where(PersonActivity.needs_review.is_(True))
            .order_by(PersonActivity.created_at.desc())
            .limit(limit)
        )
        activities = session.exec(stmt).all()
        action_map = _load_actions_for(session, [a.id for a in activities if a.id is not None])
        return [_to_response(a, action_map.get(a.id, [])) for a in activities]


@router.get("/{activity_id}", response_model=PersonActivityResponse)
def get_activity(activity_id: int) -> PersonActivityResponse:
    with session_scope() as session:
        activity = session.get(PersonActivity, activity_id)
        if activity is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Activity not found")
        actions = _load_actions_for(session, [activity_id]).get(activity_id, [])
        return _to_response(activity, actions)


@router.patch("/{activity_id}", response_model=PersonActivityResponse)
def update_activity(activity_id: int, payload: ActivityUpdateRequest) -> PersonActivityResponse:
    with session_scope() as session:
        activity = session.get(PersonActivity, activity_id)
        if activity is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Activity not found")
        if payload.activity is not None:
            activity.activity = payload.activity
        if payload.confidence is not None:
            activity.confidence = payload.confidence
        if payload.needs_review is not None:
            activity.needs_review = payload.needs_review
        activity.updated_at = datetime.utcnow()
        session.add(activity)
        session.flush()
        if payload.activity and activity.clip_path:
            activity_recognizer.register_clip(payload.activity, Path(activity.clip_path))
        actions = _load_actions_for(session, [activity_id]).get(activity_id, [])
        return _to_response(activity, actions)


def _load_actions_for(session, activity_ids: List[int]) -> Dict[int, List[ActivityAction]]:
    if not activity_ids:
        return {}
    stmt = (
        select(ActivityAction)
        .where(ActivityAction.activity_id.in_(activity_ids))
        .order_by(ActivityAction.created_at.asc())
    )
    rows = session.exec(stmt).all()
    grouped: Dict[int, List[ActivityAction]] = defaultdict(list)
    for action in rows:
        grouped[action.activity_id].append(action)
    return grouped
