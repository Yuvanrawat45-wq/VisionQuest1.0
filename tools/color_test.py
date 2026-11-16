r"""Standalone camera-based colour recognition verification tool.

Runs the camera and detector, and renders the recognized colour and shade
for each detected object directly on the bounding box.

Run with:
    e:\VisionProject\venv\Scripts\python.exe tools\color_test.py
"""

import sys
import time
sys.path.insert(0, ".")

import cv2
from src.config import Config
from src.camera import make_camera
from src.detector import Detector
from src.pipeline import Pipeline
from src.color import extract_object_color, get_scene_dominant_color


def main():
    cfg = Config()
    cam = make_camera("0", (cfg.width, cfg.height))
    det = Detector(cfg.model_path, cfg.imgsz, cfg.conf)
    pipe = Pipeline(cfg)

    print("=" * 60)
    print("VISIONQUEST COLOUR RECOGNITION TEST")
    print("Point the camera at colored objects (bottles, shirts, cups, phone).")
    print("Press Q to quit.")
    print("=" * 60)

    last_t = time.time()

    try:
        while True:
            frame, t = cam.read()
            if frame is None:
                break

            obs = det.track(frame)
            infos, _ = pipe.step(frame, t, obs)

            now = time.time()
            fps = 1.0 / max(now - last_t, 1e-3)
            last_t = now

            # 1. Scene Dominant Colour
            scene_col = get_scene_dominant_color(frame)

            # 2. Draw each object with its recognized colour
            for item in infos:
                x1, y1, x2, y2 = map(int, item.box)
                is_person = (item.name == "person")

                # Extract colour using inner-box / torso sampling
                color_res = extract_object_color(frame, item.box, is_person=is_person)

                # Color tag
                tag = f"#{item.id} {item.name}: {color_res.display_name}"
                if not color_res.is_reliable:
                    tag += " (?)"

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.putText(frame, tag, (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            # HUD
            hud_text = f"{fps:.1f} FPS | Scene Dominant: {scene_col.display_name} | Press Q to quit"
            cv2.putText(frame, hud_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            cv2.imshow("VisionQuest Colour Test", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
