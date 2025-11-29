from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, status
from sqlmodel import select

from ..config import config
from ..database import session_scope
from ..models import ActivityAction, PersonActivity
from ..schemas import (
    ActivityActionCreateRequest,
    ActivityActionResponse,
    ActivityActionUpdateRequest,
)

router = APIRouter(prefix=f"{config.api_prefix}/actions", tags=["actions"])


def _to_response(action: ActivityAction) -> ActivityActionResponse:
    return ActivityActionResponse(
        id=action.id,
        activity_id=action.activity_id,
        action_type=action.action_type,
        confidence=action.confidence,
        start_offset=action.start_offset,
        end_offset=action.end_offset,
        created_at=action.created_at,
        updated_at=action.updated_at,
    )


@router.get("", response_model=List[ActivityActionResponse])
def list_actions(activity_id: Optional[int] = None) -> List[ActivityActionResponse]:
    with session_scope() as session:
        stmt = select(ActivityAction)
        if activity_id is not None:
            stmt = stmt.where(ActivityAction.activity_id == activity_id)
        stmt = stmt.order_by(ActivityAction.created_at.desc())
        actions = session.exec(stmt).all()
        return [_to_response(action) for action in actions]


@router.post("", response_model=ActivityActionResponse, status_code=status.HTTP_201_CREATED)
def create_action(payload: ActivityActionCreateRequest) -> ActivityActionResponse:
    with session_scope() as session:
        activity = session.get(PersonActivity, payload.activity_id)
        if activity is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Activity not found")
        action = ActivityAction(
            activity_id=payload.activity_id,
            action_type=payload.action_type,
            confidence=payload.confidence if payload.confidence is not None else 0.0,
            start_offset=payload.start_offset,
            end_offset=payload.end_offset,
        )
        session.add(action)
        session.flush()
        session.refresh(action)
        return _to_response(action)


@router.patch("/{action_id}", response_model=ActivityActionResponse)
def update_action(action_id: int, payload: ActivityActionUpdateRequest) -> ActivityActionResponse:
    with session_scope() as session:
        action = session.get(ActivityAction, action_id)
        if action is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Action not found")
        if payload.action_type is not None:
            action.action_type = payload.action_type
        if payload.confidence is not None:
            action.confidence = payload.confidence
        if payload.start_offset is not None:
            action.start_offset = payload.start_offset
        if payload.end_offset is not None:
            action.end_offset = payload.end_offset
        action.updated_at = datetime.utcnow()
        session.add(action)
        session.flush()
        session.refresh(action)
        return _to_response(action)


@router.delete("/{action_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_action(action_id: int) -> None:
    with session_scope() as session:
        action = session.get(ActivityAction, action_id)
        if action is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Action not found")
        session.delete(action)
