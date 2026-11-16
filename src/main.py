"""Entry point.

Run from the repo root (visionquest/ directory) with the venv interpreter:

    python run.py --source 0 --show

Controls (window must have focus):
    V  — activate voice question (5 s recording window)
    Q  — quit
    ESC — quit
"""

import argparse
import json
import threading
import time
from typing import List, Optional

import cv2

from .config import Config
from .camera import make_camera
from .detector import Detector
from .pipeline import Pipeline
from .speech import make_speaker
from .beeper import ProximityBeeper
from .listener import Listener
from .intent import parse as parse_intent
from .qa import answer as build_answer, is_silenced


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw(frame, infos, cfg, fps, qa_active: bool = False, beep_status = (None, None)):
    for i in infos:
        x1, y1, x2, y2 = map(int, i.box)
        col = (0, 0, 255) if i.moving else (0, 200, 0) if i.moving is False else (160, 160, 160)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        tag = f"#{i.id} {i.name} ~{i.dist_m:.1f}m"
        if i.ttc_s is not None:
            tag += f" TTC:{i.ttc_s:.1f}s"
        cv2.putText(frame, tag, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

    for f in (cfg.corridor_l, cfg.corridor_r):
        cv2.line(frame,
                 (int(f * cfg.width), 0),
                 (int(f * cfg.width), cfg.height),
                 (0, 255, 255), 1)

    status = "LISTENING... (speak your question)" if qa_active else f"{fps:.1f} FPS | V=Ask Question | Q=Quit"
    col_st = (0, 128, 255) if qa_active else (255, 255, 255)
    cv2.putText(frame, status, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, col_st, 1)

    zone_idx, ema_dist = beep_status
    if zone_idx is not None and ema_dist is not None:
        zone_labels = ["VERY CLOSE (rapid)", "CLOSE (fast)", "MEDIUM", "FAR (slow)"]
        zone_colors = [(0, 0, 255), (0, 140, 255), (0, 255, 255), (0, 255, 0)]
        z_text = f"BEEP: {zone_labels[zone_idx]} ~{ema_dist:.2f}m"
        z_col = zone_colors[zone_idx]
        cv2.putText(frame, z_text, (8, cfg.height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, z_col, 2)
    else:
        cv2.putText(frame, "BEEP: Silent (>4.5m or no obstacle)", (8, cfg.height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    return frame


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="'pi' | webcam index | video file")
    ap.add_argument("--model",  default=None)
    ap.add_argument("--mode",   choices=["obstacle", "describe"], default="describe")
    ap.add_argument("--imgsz",  type=int,   default=None)
    ap.add_argument("--hfov",   type=float, default=None)
    ap.add_argument("--focal",  type=float, default=None)
    ap.add_argument("--show",   action="store_true")
    ap.add_argument("--mute",   action="store_true")
    ap.add_argument("--log",        default=None)
    ap.add_argument("--debug-beep", action="store_true",
                    help="Print beeper distance/zone info every frame")
    a = ap.parse_args()

    cfg = Config()
    if a.model:   cfg.model_path = a.model
    cfg.mode = a.mode
    if a.imgsz:  cfg.imgsz = a.imgsz
    if a.hfov:   cfg.hfov_deg = a.hfov
    if a.focal:  cfg.focal_px_override = a.focal

    cam     = make_camera(a.source, (cfg.width, cfg.height))
    det     = Detector(cfg.model_path, cfg.imgsz, cfg.conf)
    pipe    = Pipeline(cfg)
    speaker = make_speaker(a.mute)
    beeper  = ProximityBeeper(muted=a.mute, debug=getattr(a, "debug_beep", False))
    listener = Listener()

    # YOLO class names for intent parsing and Q&A
    yolo_names: List[str] = list(det.model.names.values())

    # Shared state for voice Q&A (written from worker thread, read on main)
    _qa_lock   = threading.Lock()
    _qa_active = False          # True while recording/processing
    _latest_infos: List = []   # snapshot for the Q&A worker
    _latest_frame = None       # snapshot of latest camera frame for color analysis

    def _on_heard(text: Optional[str]):
        """Called from the listener thread when STT completes."""
        nonlocal _qa_active
        try:
            if text is None:
                speaker.say("I could not understand. Please try again.")
                return

            intent = parse_intent(text, yolo_names)
            print(f"[qa] intent={intent}")

            with _qa_lock:
                infos_snapshot = list(_latest_infos)
                frame_snapshot = _latest_frame.copy() if _latest_frame is not None else None

            response = build_answer(intent, infos_snapshot, yolo_names, frame_bgr=frame_snapshot)
            print(f"[qa] answer: {response}")
            speaker.say(response, urgent=False)
        finally:
            if not is_silenced():
                beeper.resume()
            else:
                beeper.pause()
            with _qa_lock:
                _qa_active = False

    log  = open(a.log, "w") if a.log else None
    last = time.time()

    speaker.say("System ready")

    try:
        while True:
            frame, t = cam.read()
            if frame is None:
                break

            obs           = det.track(frame)
            infos, alert  = pipe.step(frame, t, obs)

            # --- Update shared infos and frame snapshot for Q&A ---
            with _qa_lock:
                _latest_infos = infos
                _latest_frame = frame.copy()

            # --- Proximity beeper (non-blocking) ---
            beeper.update(infos)

            # --- Legacy alert TTS (obstacle mode only, not in beep/qa mode) ---
            # In "describe" mode the TTS announcements are suppressed;
            # in "obstacle" mode we keep the existing urgent-alert behaviour.
            if alert and cfg.mode == "obstacle":
                result = speaker.say(alert.text, alert.urgent)
                if result:
                    pipe.mark_spoken(alert.key, t)

            if infos:
                for i in infos:
                    print(
                        f"ID={i.id} age={i.age} moving={i.moving} "
                        f"distance={i.dist_m:.2f}m TTC={i.ttc_s} "
                        f"corridor={i.in_corridor} truncated={i.truncated}"
                    )

            if log:
                log.write(json.dumps({
                    "t": t,
                    "alert": alert.text if alert else None,
                    "tracks": [
                        {"id": i.id, "n": i.name, "mv": i.moving,
                         "d": round(i.dist_m, 2), "ttc": i.ttc_s, "dir": i.direction}
                        for i in infos
                    ]
                }) + "\n")

            # --- Display ---
            if a.show:
                now = time.time()
                fps = 1 / max(now - last, 1e-3)
                last = now

                with _qa_lock:
                    qa_live = _qa_active

                beep_st = beeper.current_status
                cv2.imshow("VisionQuest", draw(frame, infos, cfg, fps, qa_active=qa_live, beep_status=beep_st))

                key = cv2.waitKey(1) & 0xFF

                if key in (27, ord("q")):
                    break

                # V key → start voice Q&A
                if key == ord("v"):
                    with _qa_lock:
                        already_active = _qa_active
                    if not already_active and not listener.busy:
                        with _qa_lock:
                            _qa_active = True
                        beeper.pause()
                        listener.listen_async(_on_heard)

    finally:
        beeper.close()
        cam.close()
        speaker.close()
        if log:
            log.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
