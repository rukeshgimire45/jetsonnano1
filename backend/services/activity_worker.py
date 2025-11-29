from __future__ import annotations

import queue
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from sqlmodel import select

from ..config import config
from ..database import session_scope
from ..events import event_bus
from ..models import PersonActivity, ActivityAction
from ..storage import encode_image_url
from .activity_buffer import ActivityWindow, ActivitySample
from .activity_recognizer import activity_recognizer


class ActivityWorker:
    def __init__(self) -> None:
        self._queue: "queue.Queue[Optional[ActivityWindow]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._fps = max(1, config.activity_sample_rate)
        # Prefer browser-friendly codecs, but keep a fallback if unavailable at runtime.
        self._clip_codecs: list[tuple[str, str]] = [
            (".webm", "VP90"),  # VP9 inside WebM works in modern browsers
            (".mp4", "mp4v"),   # fallback for environments without VP9 encoder
        ]

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ActivityWorker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._queue.put(None)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def submit(self, window: ActivityWindow) -> None:
        if not config.activity_enabled:
            return
        self._queue.put(window)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                window = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if window is None:
                continue
            try:
                self._process_window(window)
            except Exception as exc:  # pragma: no cover - safety net
                print(f"ActivityWorker error: {exc}")
            finally:
                self._queue.task_done()

    def _process_window(self, window: ActivityWindow) -> None:
        frames = self._decode_samples(window.samples)
        if not frames:
            return
        clip_path = self._write_clip(window, frames)
        activity_id = self._create_activity_record(window, clip_path)
        label, confidence = activity_recognizer.predict([frame for _, frame in frames])
        activity_label = label
        needs_review = True
        if label and confidence >= config.activity_min_confidence:
            needs_review = False
        self._update_activity_record(
            activity_id=activity_id,
            activity=activity_label,
            confidence=confidence,
            needs_review=needs_review,
        )
        self._ensure_default_action(activity_id, window, activity_label, confidence)
        self._publish_event(
            activity_id=activity_id,
            label=window.label,
            visit_id=window.visit_id,
            activity=activity_label,
            confidence=confidence,
            needs_review=needs_review,
            clip_path=clip_path,
            window_start=window.window_start,
            window_end=window.window_end,
        )

    def _decode_samples(self, samples: List[ActivitySample]) -> List[tuple[datetime, np.ndarray]]:
        decoded: List[tuple[datetime, np.ndarray]] = []
        for sample in samples:
            arr = np.frombuffer(sample.jpeg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            decoded.append((sample.timestamp, frame))
        return decoded

    def _write_clip(self, window: ActivityWindow, frames: List[tuple[datetime, np.ndarray]]) -> Path:
        label_dir = config.activity_clip_dir / window.label
        label_dir.mkdir(parents=True, exist_ok=True)
        base_name = (
            f"activity_{window.label}_{window.visit_id or 'novisit'}_"
            f"{window.window_start.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        )
        height, width = frames[0][1].shape[:2]
        last_error: Optional[str] = None
        for extension, codec in self._clip_codecs:
            clip_path = label_dir / f"{base_name}{extension}"
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(
                str(clip_path),
                fourcc,
                self._fps,
                (width, height),
            )
            if not writer.isOpened():
                last_error = f"codec {codec}"
                writer.release()
                if clip_path.exists():
                    clip_path.unlink(missing_ok=True)
                continue
            for _, frame in frames:
                writer.write(frame)
            writer.release()
            return clip_path
        raise RuntimeError(f"Failed to initialize video writer (last error: {last_error})")

    def _create_activity_record(self, window: ActivityWindow, clip_path: Path) -> int:
        with session_scope() as session:
            entry = PersonActivity(
                label=window.label,
                visit_id=window.visit_id,
                activity=None,
                confidence=0.0,
                needs_review=True,
                clip_path=str(clip_path),
                window_start=window.window_start,
                window_end=window.window_end,
            )
            session.add(entry)
            session.flush()
            if entry.id is None:
                raise RuntimeError("Failed to persist PersonActivity record")
            return entry.id

    def _update_activity_record(
        self,
        activity_id: int,
        activity: Optional[str],
        confidence: float,
        needs_review: bool,
    ) -> None:
        with session_scope() as session:
            entry = session.get(PersonActivity, activity_id)
            if entry is None:
                return
            entry.activity = activity
            entry.confidence = confidence
            entry.needs_review = needs_review
            entry.updated_at = datetime.utcnow()
            session.add(entry)

    def _publish_event(
        self,
        activity_id: int,
        label: str,
        visit_id: Optional[int],
        activity: Optional[str],
        confidence: float,
        needs_review: bool,
        clip_path: Path,
        window_start: datetime,
        window_end: datetime,
    ) -> None:
        actions_payload = self._serialize_actions(activity_id)
        payload = {
            "type": "activity",
            "id": activity_id,
            "label": label,
            "visit_id": visit_id,
            "activity": activity,
            "confidence": confidence,
            "needs_review": needs_review,
            "clip_url": encode_image_url(clip_path),
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "actions": actions_payload,
        }
        event_bus.publish_from_thread(payload)

    def _ensure_default_action(
        self,
        activity_id: int,
        window: ActivityWindow,
        action_label: Optional[str],
        confidence: float,
    ) -> None:
        if not action_label:
            return
        duration = (window.window_end - window.window_start).total_seconds()
        with session_scope() as session:
            existing = session.exec(
                select(ActivityAction).where(ActivityAction.activity_id == activity_id)
            ).first()
            if existing is not None:
                return
            action = ActivityAction(
                activity_id=activity_id,
                action_type=action_label,
                confidence=confidence,
                start_offset=0.0,
                end_offset=max(duration, 0.0),
            )
            session.add(action)

    def _serialize_actions(self, activity_id: int) -> List[dict]:
        with session_scope() as session:
            stmt = (
                select(ActivityAction)
                .where(ActivityAction.activity_id == activity_id)
                .order_by(ActivityAction.created_at.asc())
            )
            actions = session.exec(stmt).all()
        return [
            {
                "id": action.id,
                "activity_id": action.activity_id,
                "action_type": action.action_type,
                "confidence": action.confidence,
                "start_offset": action.start_offset,
                "end_offset": action.end_offset,
                "created_at": action.created_at.isoformat(),
                "updated_at": action.updated_at.isoformat(),
            }
            for action in actions
        ]


activity_worker = ActivityWorker()
