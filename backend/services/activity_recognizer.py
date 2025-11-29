from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..config import config


class ActivityRecognizer:
    """Lightweight nearest-neighbor recognizer using averaged frame features."""

    def __init__(self) -> None:
        self._memory_path = config.activity_dir / "activity_memory.json"
        self._lock = threading.Lock()
        self._examples: list[dict] = self._load_memory()

    def _load_memory(self) -> list[dict]:
        if not self._memory_path.exists():
            return []
        try:
            with self._memory_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        return []

    def _save_memory(self) -> None:
        self._memory_path.parent.mkdir(parents=True, exist_ok=True)
        with self._memory_path.open("w", encoding="utf-8") as fh:
            json.dump(self._examples, fh)

    def _feature_from_frames(self, frames: List[np.ndarray]) -> Optional[np.ndarray]:
        if not frames:
            return None
        feats = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (32, 32))
            feats.append(resized.flatten().astype("float32") / 255.0)
        if not feats:
            return None
        vector = np.mean(feats, axis=0)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return None
        return vector / norm

    def predict(self, frames: List[np.ndarray]) -> Tuple[Optional[str], float]:
        feature = self._feature_from_frames(frames)
        if feature is None:
            return None, 0.0
        with self._lock:
            if not self._examples:
                return None, 0.0
            best_label: Optional[str] = None
            best_score = 0.0
            for entry in self._examples:
                exemplar = np.array(entry.get("feature", []), dtype="float32")
                if exemplar.size != feature.size:
                    continue
                score = float(np.dot(feature, exemplar))
                if score > best_score:
                    best_label = entry.get("label")
                    best_score = score
            return best_label, best_score

    def register_clip(self, label: str, clip_path: Path) -> None:
        frames = self._read_clip_frames(clip_path)
        if not frames:
            return
        feature = self._feature_from_frames(frames)
        if feature is None:
            return
        with self._lock:
            self._examples = [ex for ex in self._examples if ex.get("clip_path") != str(clip_path)]
            self._examples.append({
                "label": label,
                "clip_path": str(clip_path),
                "feature": feature.tolist(),
            })
            self._save_memory()

    def _read_clip_frames(self, clip_path: Path, max_frames: int = 16) -> List[np.ndarray]:
        if not clip_path.exists():
            return []
        cap = cv2.VideoCapture(str(clip_path))
        frames: List[np.ndarray] = []
        try:
            while len(frames) < max_frames:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frames.append(frame)
        finally:
            cap.release()
        return frames


activity_recognizer = ActivityRecognizer()
