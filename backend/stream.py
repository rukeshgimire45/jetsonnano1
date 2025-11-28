from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np


class LiveStreamBuffer:
    """Stores the latest JPEG frame for lightweight streaming."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Optional[bytes] = None
        self._timestamp: float = 0.0
        self._min_interval = 0.05  # throttle JPEG encoding to ~20 FPS

    def push_frame(self, frame_bgr: np.ndarray) -> None:
        now = time.time()
        if now - self._timestamp < self._min_interval:
            return
        success, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not success:
            return
        data = encoded.tobytes()
        with self._lock:
            self._frame = data
            self._timestamp = now

    def get_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._frame


stream_buffer = LiveStreamBuffer()
