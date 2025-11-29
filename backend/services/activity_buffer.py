from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..config import config


@dataclass
class ActivitySample:
    timestamp: datetime
    jpeg: bytes


@dataclass
class ActivityWindow:
    label: str
    visit_id: Optional[int]
    samples: List[ActivitySample]
    sample_rate: int

    @property
    def window_start(self) -> datetime:
        return self.samples[0].timestamp

    @property
    def window_end(self) -> datetime:
        return self.samples[-1].timestamp


class _BufferState:
    def __init__(self, maxlen: int) -> None:
        self.samples: Deque[ActivitySample] = deque(maxlen=maxlen)
        self.last_sample_ts: Optional[datetime] = None
        self.last_emit_ts: Optional[datetime] = None


class ActivityBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: Dict[Tuple[str, Optional[int]], _BufferState] = {}
        self._sample_interval = 1.0 / max(1, config.activity_sample_rate)
        self._window = timedelta(seconds=config.activity_window_seconds)
        self._maxlen = config.activity_buffer_seconds * config.activity_sample_rate
        self._min_samples = max(1, config.activity_window_seconds * config.activity_sample_rate)
        self._frame_size = (
            max(1, config.activity_frame_width),
            max(1, config.activity_frame_height),
        )
        self._quality = 70

    def push_frame(
        self,
        label: str,
        visit_id: Optional[int],
        frame_bgr: np.ndarray,
        timestamp: datetime,
    ) -> Optional[ActivityWindow]:
        if not config.activity_enabled or visit_id is None:
            return None
        key = (label, visit_id)
        with self._lock:
            state = self._states.get(key)
            if state is None:
                state = _BufferState(self._maxlen)
                self._states[key] = state
            if not self._should_sample(state, timestamp):
                return None
            sample = self._encode_sample(frame_bgr, timestamp)
            if sample is None:
                return None
            state.samples.append(sample)
            state.last_sample_ts = timestamp
            if not state.samples:
                return None
            if state.last_emit_ts is None or timestamp - state.last_emit_ts >= self._window:
                if len(state.samples) >= self._min_samples:
                    state.last_emit_ts = timestamp
                    return ActivityWindow(
                        label=label,
                        visit_id=visit_id,
                        samples=list(state.samples),
                        sample_rate=config.activity_sample_rate,
                    )
        return None

    def finalize_visit(self, label: str, visit_id: Optional[int]) -> Optional[ActivityWindow]:
        if visit_id is None:
            return None
        key = (label, visit_id)
        with self._lock:
            state = self._states.pop(key, None)
        if state and state.samples:
            return ActivityWindow(
                label=label,
                visit_id=visit_id,
                samples=list(state.samples),
                sample_rate=config.activity_sample_rate,
            )
        return None

    def _should_sample(self, state: _BufferState, timestamp: datetime) -> bool:
        if state.last_sample_ts is None:
            return True
        delta = (timestamp - state.last_sample_ts).total_seconds()
        return delta >= self._sample_interval

    def _encode_sample(self, frame_bgr: np.ndarray, timestamp: datetime) -> Optional[ActivitySample]:
        try:
            resized = cv2.resize(frame_bgr, self._frame_size)
        except cv2.error:
            return None
        success, encoded = cv2.imencode(
            ".jpg",
            resized,
            [int(cv2.IMWRITE_JPEG_QUALITY), self._quality],
        )
        if not success:
            return None
        return ActivitySample(timestamp=timestamp, jpeg=encoded.tobytes())


activity_buffer = ActivityBuffer()
