from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from ..config import config


EMBED_SIZE = (64, 64)


class FaceRegistry:
    def __init__(
        self,
        distance_threshold: float = config.face_distance_threshold,
        confidence_threshold: float = config.face_match_confidence,
    ):
        self.distance_threshold = distance_threshold
        self.confidence_threshold = confidence_threshold
        self.embedder = None  # kept for backward compatibility
        self.db: Dict[str, List[np.ndarray]] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        if not config.enroll_dir.exists():
            return
        for person_dir in config.enroll_dir.iterdir():
            if not person_dir.is_dir():
                continue
            label = person_dir.name
            for img_path in person_dir.glob("*.*"):
                embedding = self._embedding_from_file(img_path)
                if embedding is None:
                    continue
                self.db.setdefault(label, []).append(embedding)

    def _embedding_from_file(self, img_path: Path):
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        return self.embed(img)

    def embed(self, face_bgr: np.ndarray):
        if face_bgr.size == 0:
            return None
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, EMBED_SIZE, interpolation=cv2.INTER_AREA)
        embedding = resized.flatten().astype(np.float32) / 255.0
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return None
        return embedding / norm

    def match(self, embedding: np.ndarray):
        best_label = None
        best_distance = 1.0
        for label, vectors in self.db.items():
            for ref in vectors:
                distance = 1.0 - float(np.dot(ref, embedding))
                if distance < best_distance:
                    best_distance = distance
                    best_label = label
        if best_label is not None:
            confidence = max(0.0, 1.0 - best_distance)
            if confidence >= self.confidence_threshold:
                return best_label, confidence
        return None, None

    def add(self, label: str, face_bgr: np.ndarray):
        label = label.strip()
        if not label:
            return None
        dest_dir = config.enroll_dir / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        img_path = dest_dir / f"face_{timestamp}.png"
        cv2.imwrite(str(img_path), face_bgr)
        embedding = self.embed(face_bgr)
        if embedding is None:
            return None
        self.db.setdefault(label, []).append(embedding)
        return img_path

    def add_from_path(self, label: str, path: Path):
        img = cv2.imread(str(path))
        if img is None:
            return
        embedding = self.embed(img)
        if embedding is None:
            return
        self.db.setdefault(label, []).append(embedding)


class UnknownFaceMemory:
    def __init__(self, maxlen: int = config.unknown_memory):
        from collections import deque
        self.store = deque(maxlen=maxlen)

    def should_prompt(self, embedding: np.ndarray, threshold: float = 0.1) -> bool:
        for ref in self.store:
            distance = 1.0 - float(np.dot(ref, embedding))
            if distance < threshold:
                return False
        return True

    def remember(self, embedding: np.ndarray) -> None:
        self.store.append(embedding)


face_registry = FaceRegistry()
unknown_memory = UnknownFaceMemory()
