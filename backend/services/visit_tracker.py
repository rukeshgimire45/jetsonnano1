from __future__ import annotations

import threading
from datetime import datetime, timedelta

from sqlmodel import select

from ..config import config
from ..database import session_scope
from ..models import PersonVisit


class VisitTracker:
    def __init__(self, gap_seconds: int = config.visit_gap_seconds) -> None:
        self.gap = timedelta(seconds=gap_seconds)
        self._lock = threading.Lock()

    def record_detection(self, label: str, timestamp: datetime) -> None:
        if not label:
            return
        with self._lock:
            with session_scope() as session:
                active = session.exec(
                    select(PersonVisit)
                    .where(PersonVisit.label == label, PersonVisit.detection_out.is_(None))
                    .order_by(PersonVisit.detection_in.desc())
                    .limit(1)
                ).first()
                if active is not None:
                    if timestamp - active.last_seen <= self.gap:
                        active.last_seen = timestamp
                        session.add(active)
                        return
                    active.detection_out = active.last_seen
                    session.add(active)
                visit = PersonVisit(label=label, detection_in=timestamp, last_seen=timestamp)
                session.add(visit)

    def expire_inactive(self, timestamp: datetime) -> None:
        with self._lock:
            with session_scope() as session:
                active_visits = session.exec(
                    select(PersonVisit).where(PersonVisit.detection_out.is_(None))
                ).all()
                changed = False
                for visit in active_visits:
                    if timestamp - visit.last_seen > self.gap:
                        visit.detection_out = visit.last_seen
                        session.add(visit)
                        changed = True
                if changed:
                    session.flush()


visit_tracker = VisitTracker()
