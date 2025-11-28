from __future__ import annotations

from pathlib import Path
from typing import Optional
from datetime import datetime
import shutil
import uuid

import cv2
import numpy as np

from .config import config


def save_unknown_face(image_bgr: np.ndarray, confidence: float) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"unknown_{timestamp}_{uuid.uuid4().hex[:6]}.png"
    dest = config.unknown_dir / filename
    cv2.imwrite(str(dest), image_bgr)
    return dest


def move_to_enroll(image_path: Path, label: str) -> Path:
    label_dir = config.enroll_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    dest = label_dir / image_path.name
    shutil.move(str(image_path), dest)
    return dest


def encode_image_url(path: Path | str) -> str:
    # Allow both Path objects and raw strings from persisted DB rows
    path = Path(path)
    rel = path.relative_to(config.root_dir)
    return f"/media/{rel.as_posix()}"


def delete_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def ensure_hls_playlist() -> Optional[Path]:
    playlist = config.hls_dir / config.hls_playlist
    return playlist if playlist.exists() else None
