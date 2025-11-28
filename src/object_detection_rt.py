#!/usr/bin/env python3


import time
import jetson_inference as ji
import jetson_utils as ju



def main():
    # Load SSD-Mobilenet v2 detection model
    net = ji.detectNet("ssd-mobilenet-v2", threshold=0.5)

    # USB camera on /dev/video0 (same as detectnet)
    camera = ju.videoSource("v4l2:///dev/video0")

    print("Starting real-time object detection (headless). Ctrl+C to stop.")

    try:
        while True:
            img = camera.Capture()
            if img is None:
                continue  # no frame, skip

            detections = net.Detect(img)

            if detections:
                ts = time.strftime("%H:%M:%S")
                print(f"\n[{ts}] Detected {len(detections)} object(s):")
                for d in detections:
                    class_name = net.GetClassDesc(d.ClassID)
                    conf = d.Confidence
                    left, top, right, bottom = int(d.Left), int(d.Top), int(d.Right), int(d.Bottom)
                    print(
                        f"  - {class_name:15} "
                        f"conf={conf:5.2f} "
                        f"bbox=({left},{top})-({right},{bottom})"
                    )

            # stop if camera stops streaming
            if not camera.IsStreaming():
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user, exiting.")


if __name__ == "__main__":
    main()
