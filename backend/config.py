from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(slots=True)
class AppConfig:
    root_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = root_dir / "data"
    enroll_dir: Path = data_dir / "faces_enroll"
    unknown_dir: Path = data_dir / "faces_unknown"
    db_path: Path = root_dir / "data" / "app.db"
    hls_dir: Path = root_dir / "hls"
    hls_playlist: str = "stream.m3u8"
    hls_mount: str = "/hls"
    api_prefix: str = "/api/v1"
    camera_uri: str = os.environ.get("JETSON_CAMERA_URI", "v4l2:///dev/video0")
    laptop_stream_ip: str = os.environ.get("JETSON_LAPTOP_IP", "10.0.4.28")
    max_face_tracks: int = 5
    face_distance_threshold: float = 0.24
    face_match_confidence: float = 0.9
    unknown_memory: int = 20
    visit_gap_seconds: int = 10

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.enroll_dir.mkdir(parents=True, exist_ok=True)
        self.unknown_dir.mkdir(parents=True, exist_ok=True)
        self.hls_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


config = AppConfig()
config.ensure_dirs()
