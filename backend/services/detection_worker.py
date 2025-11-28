from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import List

import numpy as np
import cv2

try:  # pragma: no cover - hardware-specific imports
    from ultralytics import YOLO
    import jetson_utils as ju
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover
    YOLO = None
    ju = None
    mp = None
    print(f"DetectionWorker warning: dependency missing -> {exc}")

from sqlmodel import select

from ..config import config
from ..database import session_scope
from ..models import DetectionEvent, Person, UnknownFace
from ..storage import save_unknown_face, encode_image_url
from ..events import event_bus
from .face_registry import face_registry, unknown_memory
from .visit_tracker import visit_tracker
from ..stream import stream_buffer


class DetectionWorker:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.running = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="DetectionWorker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self.running = False

    def _run(self) -> None:
        if YOLO is None or ju is None:
            print("DetectionWorker unavailable: YOLO/jetson_utils missing")
            return

        model = YOLO("yolov8n.pt")
        model.overrides["device"] = "cpu"
        model.overrides["compile"] = False
        model.to("cpu")

        camera = ju.videoSource(config.camera_uri)

        face_mesh_detector = None
        if mp is not None:
            face_mesh_detector = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=config.max_face_tracks,
                refine_landmarks=True,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4,
            )

        timeout_count = 0
        max_timeouts = 30
        self.running = True
        print("DetectionWorker started")

        try:
            while not self._stop.is_set():
                img_cuda = camera.Capture()
                if img_cuda is None:
                    timeout_count += 1
                    if timeout_count >= max_timeouts:
                        print("DetectionWorker: camera timeout, exiting loop")
                        break
                    time.sleep(0.1)
                    continue
                timeout_count = 0

                frame_rgba = ju.cudaToNumpy(img_cuda)
                frame_u8 = frame_rgba.astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame_u8, cv2.COLOR_RGBA2BGR)
                frame_h, frame_w = frame_bgr.shape[:2]

                stream_buffer.push_frame(frame_bgr)

                visit_tracker.expire_inactive(datetime.utcnow())

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if face_mesh_detector else None
                face_infos = self._detect_faces(frame_rgb, frame_w, frame_h, face_mesh_detector)
                self._process_faces(face_infos, frame_bgr)

                results = model(frame_bgr, verbose=False)
                _ = results[0].boxes  # placeholder for future object actions

        finally:
            self.running = False
            if face_mesh_detector is not None:
                face_mesh_detector.close()
            camera.Close()
            print("DetectionWorker stopped")

    def _detect_faces(self, frame_rgb, frame_w, frame_h, face_mesh_detector):
        if face_mesh_detector is None or frame_rgb is None:
            return []
        results = face_mesh_detector.process(frame_rgb)
        if not getattr(results, "multi_face_landmarks", None):
            return []
        faces = []
        for face_landmarks in results.multi_face_landmarks:
            xs = [lm.x for lm in face_landmarks.landmark]
            ys = [lm.y for lm in face_landmarks.landmark]
            face_box = (
                max(0, int(min(xs) * frame_w)),
                max(0, int(min(ys) * frame_h)),
                min(frame_w, int(max(xs) * frame_w)),
                min(frame_h, int(max(ys) * frame_h)),
            )
            faces.append({"box": face_box})
        return faces

    def _crop_with_margin(self, frame_bgr, box, margin: float = 0.1):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            return None
        dw = int(w * margin)
        dh = int(h * margin)
        x1c = max(0, x1 - dw)
        y1c = max(0, y1 - dh)
        x2c = min(frame_bgr.shape[1], x2 + dw)
        y2c = min(frame_bgr.shape[0], y2 + dh)
        crop = frame_bgr[y1c:y2c, x1c:x2c]
        return crop.copy() if crop.size else None

    def _process_faces(self, face_entries: List[dict], frame_bgr):
        if not face_entries:
            return
        for face in face_entries:
            crop = self._crop_with_margin(frame_bgr, face["box"])
            if crop is None:
                continue
            embedding = face_registry.embed(crop)
            if embedding is None:
                continue
            match_label, confidence = face_registry.match(embedding)
            if match_label:
                self._record_known_face(match_label, confidence or 0.9)
                continue
            if not unknown_memory.should_prompt(embedding):
                continue
            face_path = save_unknown_face(crop, confidence=0.0)
            self._record_unknown_face(face_path)
            unknown_memory.remember(embedding)

    def _record_known_face(self, label: str, confidence: float):
        with session_scope() as session:
            person = session.exec(select(Person).where(Person.label == label)).first()
            if person is None:
                person = Person(label=label)
            person.total_detections += 1
            person.last_seen = datetime.utcnow()
            person.updated_at = datetime.utcnow()
            session.add(person)
            event = DetectionEvent(label=label, confidence=confidence, is_known=True)
            session.add(event)
            session.flush()
            payload = {
                "id": event.id,
                "label": label,
                "confidence": confidence,
                "is_known": True,
                "created_at": event.created_at.isoformat(),
                "image_url": None,
            }
        event_bus.publish_from_thread(payload)
        visit_tracker.record_detection(label, datetime.utcnow())

    def _record_unknown_face(self, image_path):
        path = image_path
        with session_scope() as session:
            entry = UnknownFace(image_path=str(path))
            session.add(entry)
            session.flush()
            event = DetectionEvent(label="unknown", confidence=0.0, is_known=False, image_path=str(path))
            session.add(event)
            session.flush()
            payload = {
                "id": event.id,
                "label": "unknown",
                "confidence": 0.0,
                "is_known": False,
                "created_at": event.created_at.isoformat(),
                "image_url": encode_image_url(path),
            }
        event_bus.publish_from_thread(payload)


detection_worker = DetectionWorker()
