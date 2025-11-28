import time
import hashlib
import numpy as np
import cv2

try:
    import mediapipe as mp
    _mp_hands = mp.solutions.hands
    _HAND_LANDMARK = _mp_hands.HandLandmark
except ImportError:  # mediapipe may not be installed on Jetson devices
    mp = None
    _mp_hands = None
    _HAND_LANDMARK = None

from ultralytics import YOLO
import jetson_utils as ju


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


_PALM_FINGER_PAIRS = []
if _HAND_LANDMARK is not None:
    _PALM_FINGER_PAIRS = [
        (_HAND_LANDMARK.INDEX_FINGER_TIP, _HAND_LANDMARK.INDEX_FINGER_PIP),
        (_HAND_LANDMARK.MIDDLE_FINGER_TIP, _HAND_LANDMARK.MIDDLE_FINGER_PIP),
        (_HAND_LANDMARK.RING_FINGER_TIP, _HAND_LANDMARK.RING_FINGER_PIP),
        (_HAND_LANDMARK.PINKY_TIP, _HAND_LANDMARK.PINKY_PIP)
    ]


def _is_palm_open(landmarks) -> bool:
    """Heuristic: at least three fingers extend beyond their PIP joints."""
    if not _PALM_FINGER_PAIRS:
        return False

    extended = 0
    for tip_idx, pip_idx in _PALM_FINGER_PAIRS:
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]
        if tip.y < pip.y:
            extended += 1

    return extended >= 3


def _detect_hand_candidates(frame_rgb, frame_w, frame_h, hands_detector):
    """Return overlay metadata for each detected hand/palm."""
    if hands_detector is None or frame_rgb is None:
        return []

    results = hands_detector.process(frame_rgb)
    if not results.multi_hand_landmarks:
        return []

    overlays = []
    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
        xs = [lm.x for lm in hand_landmarks.landmark]
        ys = [lm.y for lm in hand_landmarks.landmark]
        x1 = max(0, int(min(xs) * frame_w))
        y1 = max(0, int(min(ys) * frame_h))
        x2 = min(frame_w, int(max(xs) * frame_w))
        y2 = min(frame_h, int(max(ys) * frame_h))

        score = float(handedness.classification[0].score)
        label_base = "palm" if _is_palm_open(hand_landmarks.landmark) else "hand"
        overlays.append({
            "box": (x1, y1, x2, y2),
            "label": label_base,
            "confidence": score,
            "color": _class_color(label_base)
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
    hands_detector = None
    if _mp_hands is not None:
        hands_detector = _mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4
        )

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
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if hands_detector is not None else None
            hand_overlays = _detect_hand_candidates(frame_rgb, frame_w, frame_h, hands_detector)

            # --------------------
            # 3. Run YOLO (CPU)
            # --------------------
            results = model(frame_bgr, verbose=False)
            dets = results[0].boxes

            # --------------------
            # 4. Draw boxes back onto CUDA image
            # --------------------
            overlays = []
            for box in dets:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                if conf < 0.4:
                    continue

                x1_i, y1_i, x2_i, y2_i = map(int, (x1, y1, x2, y2))
                
                # Get class name and color
                class_name = model.names.get(cls_id, str(cls_id))
                color = _class_color(class_name)

                overlays.append({
                    "box": (x1_i, y1_i, x2_i, y2_i),
                    "label": class_name,
                    "confidence": conf,
                    "color": color
                })

            overlays.extend(hand_overlays)

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
        if hands_detector is not None:
            hands_detector.close()
        camera.Close()
        encoder.Close()
        print("Clean shutdown complete.")


if __name__ == "__main__":
    main()
