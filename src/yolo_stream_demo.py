import time
import hashlib
from datetime import datetime
from pathlib import Path
from collections import deque
import urllib.request
import numpy as np
import cv2

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    _mp_face_mesh = mp.solutions.face_mesh
except ImportError:  # mediapipe may not be installed on Jetson devices
    mp = None
    mp_tasks = None
    mp_vision = None
    _mp_face_mesh = None

from ultralytics import YOLO
import jetson_utils as ju


FACE_DB_DIR = Path("data/faces_enroll")
FACE_DB_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FACE_EMBEDDER_MODEL = MODEL_DIR / "face_embedder.tflite"
FACE_EMBEDDER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_embedder/face_embedder/float16/1/float16.tflite"
)

FACE_DISTANCE_THRESHOLD = 0.24  # tweak if recognition is too strict/loose
UNKNOWN_MEMORY = 20
MAX_FACE_TRACKS = 5


def _download_if_missing(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return

    print(f"Downloading {url} -> {dest} ...")
    urllib.request.urlretrieve(url, dest)


def _create_face_embedder():
    if mp is None or mp_tasks is None or mp_vision is None:
        return None

    try:
        _download_if_missing(FACE_EMBEDDER_URL, FACE_EMBEDDER_MODEL)
        base_options = mp_tasks.BaseOptions(model_asset_path=str(FACE_EMBEDDER_MODEL))
        options = mp_vision.FaceEmbedderOptions(base_options=base_options)
        return mp_vision.FaceEmbedder.create_from_options(options)
    except Exception as exc:  # pragma: no cover - defensive for missing dependencies
        print(f"Face embedder unavailable: {exc}")
        return None


class FaceRegistry:
    def __init__(self, embedder, root_dir: Path = FACE_DB_DIR, distance_threshold: float = FACE_DISTANCE_THRESHOLD):
        self.embedder = embedder
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.distance_threshold = distance_threshold
        self.db = {}
        self._load_existing()

    def _load_existing(self):
        for person_dir in self.root_dir.iterdir():
            if not person_dir.is_dir():
                continue
            label = person_dir.name
            for img_path in person_dir.glob("*.*"):
                embedding = self._embedding_from_file(img_path)
                if embedding is None:
                    continue
                self.db.setdefault(label, []).append(embedding)

        if self.db:
            print(f"Loaded embeddings for {len(self.db)} person(s).")

    def _embedding_from_file(self, img_path: Path):
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        return self.embed(img)

    def embed(self, face_bgr: np.ndarray):
        if self.embedder is None or face_bgr.size == 0:
            return None

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_rgb)
        result = self.embedder.embed(mp_image)
        if not result.embeddings:
            return None
        embedding = np.array(result.embeddings[0].embedding, dtype=np.float32)
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

        if best_label is not None and best_distance <= self.distance_threshold:
            confidence = max(0.0, 1.0 - best_distance)
            return best_label, confidence
        return None, None

    def add(self, label: str, face_bgr: np.ndarray):
        label = label.strip()
        if not label:
            return

        dest_dir = self.root_dir / label
        dest_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        img_path = dest_dir / f"face_{timestamp}.png"
        cv2.imwrite(str(img_path), face_bgr)

        embedding = self.embed(face_bgr)
        if embedding is None:
            print("Failed to compute embedding for new face; saved image for later.")
            return

        self.db.setdefault(label, []).append(embedding)
        print(f"Registered new face '{label}' with snapshot {img_path}.")


class UnknownFaceMemory:
    def __init__(self, maxlen=UNKNOWN_MEMORY):
        self.store = deque(maxlen=maxlen)

    def should_prompt(self, embedding: np.ndarray, threshold: float = 0.1):
        for ref in self.store:
            distance = 1.0 - float(np.dot(ref, embedding))
            if distance < threshold:
                return False
        return True

    def remember(self, embedding: np.ndarray):
        self.store.append(embedding)


def _class_color(label: str) -> tuple:
    """Deterministically map a class label to a bright RGBA color."""
    digest = hashlib.md5(label.encode("utf-8")).digest()
    r = 128 + digest[0] // 2
    g = 128 + digest[1] // 2
    b = 128 + digest[2] // 2
    return (int(r), int(g), int(b), 255)


def _draw_rect_outline(img_cuda, box, color, thickness=4):
    """Draw rectangle borders only (no fill) using four skinny rects."""
    x1, y1, x2, y2 = box
    t = max(1, thickness)

    # top border
    ju.cudaDrawRect(img_cuda, (x1, y1, x2, min(y2, y1 + t)), color)
    # bottom border
    ju.cudaDrawRect(img_cuda, (x1, max(y1, y2 - t), x2, y2), color)
    # left border
    ju.cudaDrawRect(img_cuda, (x1, y1, min(x2, x1 + t), y2), color)
    # right border
    ju.cudaDrawRect(img_cuda, (max(x1, x2 - t), y1, x2, y2), color)


def _detect_faces(frame_rgb, frame_w, frame_h, face_mesh_detector):
    """Return bounding boxes for faces detected via MediaPipe Face Mesh."""
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
            min(frame_h, int(max(ys) * frame_h))
        )
        faces.append({"box": face_box, "landmarks": face_landmarks})

    return faces


def _crop_with_margin(frame_bgr, box, margin: float = 0.1):
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


def _recognize_faces(face_entries, frame_bgr, registry, unknown_memory):
    if registry is None or not face_entries:
        return []

    overlays = []
    for face in face_entries:
        x1, y1, x2, y2 = face["box"]
        crop = _crop_with_margin(frame_bgr, (x1, y1, x2, y2))
        if crop is None:
            continue

        embedding = registry.embed(crop)
        if embedding is None:
            continue

        match_label, confidence = registry.match(embedding)
        if match_label:
            overlays.append({
                "box": (x1, y1, x2, y2),
                "label": match_label,
                "confidence": confidence or 0.9,
                "color": _class_color(match_label)
            })
            continue

        if not unknown_memory.should_prompt(embedding):
            continue

        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        print(f"\n[FaceID] New face detected at {timestamp}." )
        label = input("Enter label (leave blank to skip): ").strip()
        if label:
            registry.add(label, crop)
            overlays.append({
                "box": (x1, y1, x2, y2),
                "label": label,
                "confidence": 0.99,
                "color": _class_color(label)
            })
        else:
            unknown_memory.remember(embedding)
            overlays.append({
                "box": (x1, y1, x2, y2),
                "label": "unknown",
                "confidence": 0.0,
                "color": (200, 200, 200, 255)
            })

    return overlays


def main():
    # --------------------
    # 1. YOLO setup (CPU-safe)
    # --------------------
    model = YOLO("yolov8n.pt")  # will auto-download if not present

    # force CPU and disable fancy backends
    model.overrides["device"] = "cpu"
    model.overrides["compile"] = False
    model.to("cpu")

    # --------------------
    # 2. Camera & encoder
    # --------------------
    camera = ju.videoSource("v4l2:///dev/video0")

    # ⚠️ change this if your laptop IP changes
    LAPTOP_IP = "10.0.4.28"
    encoder = ju.videoOutput(f"rtp://{LAPTOP_IP}:5000")

    print(f"Streaming with YOLO overlay to rtp://{LAPTOP_IP}:5000 ...  Ctrl+C to stop")

    # font for drawing text overlays
    font = ju.cudaFont()
    face_mesh_detector = None
    if _mp_face_mesh is not None:
        face_mesh_detector = _mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=MAX_FACE_TRACKS,
            refine_landmarks=True,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4
        )

    face_embedder = _create_face_embedder()
    face_registry = FaceRegistry(face_embedder) if face_embedder else None
    unknown_memory = UnknownFaceMemory()

    timeout_count = 0
    max_timeouts = 30  # allow temporary camera hiccups

    try:
        while True:
            # Capture a frame from the camera (CUDA buffer, typically RGBA)
            img_cuda = camera.Capture()
            if img_cuda is None:
                timeout_count += 1
                if timeout_count >= max_timeouts:
                    print("No frame captured for a while, exiting loop...")
                    break

                time.sleep(0.1)  # brief pause before retrying
                continue

            timeout_count = 0

            # Convert CUDA image -> NumPy for YOLO
            frame_rgba = ju.cudaToNumpy(img_cuda)        # float32 [0–255], shape (H, W, 4)
            frame_u8 = frame_rgba.astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_u8, cv2.COLOR_RGBA2BGR)
            frame_h, frame_w = frame_bgr.shape[:2]

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if face_mesh_detector is not None else None
            face_infos = _detect_faces(frame_rgb, frame_w, frame_h, face_mesh_detector)
            face_id_overlays = _recognize_faces(face_infos, frame_bgr, face_registry, unknown_memory)

            # --------------------
            # 3. Run YOLO (CPU)
            # --------------------
            results = model(frame_bgr, verbose=False)
            dets = results[0].boxes

            # --------------------
            # 4. Draw boxes back onto CUDA image
            # --------------------
            overlays = []
            overlays.extend(face_id_overlays)

            for item in overlays:
                x1_i, y1_i, x2_i, y2_i = item["box"]
                color = item["color"]

                # Draw bounding box outline only
                _draw_rect_outline(
                    img_cuda,
                    (x1_i, y1_i, x2_i, y2_i),
                    color
                )
                
                # Draw label with class name and confidence
                label = f"{item['label']} {item['confidence']*100:0.1f}%"
                label_height = 24
                label_y = max(0, y1_i - label_height)
                label_width = min(frame_w - x1_i, max(60, len(label) * 11))

                # semi-transparent background bar for readability
                ju.cudaDrawRect(
                    img_cuda,
                    (x1_i, label_y, x1_i + label_width, min(frame_h, label_y + label_height)),
                    (0, 0, 0, 160)
                )

                text_y = min(frame_h - 1, label_y + label_height - 4)
                font.OverlayText(
                    img_cuda,
                    x1_i + 4,
                    text_y,
                    label,
                    color=color
                )

            # --------------------
            # 5. Send frame to laptop via UDP/RTP
            # --------------------
            encoder.Render(img_cuda)

            if not camera.IsStreaming() or not encoder.IsStreaming():
                print("Stream stopped (camera or encoder not streaming).")
                break

    except KeyboardInterrupt:
        print("Interrupted by user (Ctrl+C)")

    finally:
        if face_mesh_detector is not None:
            face_mesh_detector.close()
        if face_registry is not None and face_registry.embedder is not None:
            try:
                face_registry.embedder.close()
            except AttributeError:
                pass
        camera.Close()
        encoder.Close()
        print("Clean shutdown complete.")


if __name__ == "__main__":
    main()
