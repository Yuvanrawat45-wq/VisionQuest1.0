"""Record test clips for deterministic replay.  Usage:
  python -m tools.record --source pi --out clips/static_chair_01.mp4 --seconds 20
Replay:  python run.py --source clips/static_chair_01.mp4 --mute --show"""
import argparse
import os
import time
import cv2
from src.camera import make_camera

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="0")
ap.add_argument("--out", required=True)
ap.add_argument("--seconds", type=float, default=20)
ap.add_argument("--fps", type=float, default=20)
a = ap.parse_args()

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
cam = make_camera(a.source)
w = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (640, 480))
t0 = time.time(); nxt = t0
while time.time() - t0 < a.seconds:
    f, _ = cam.read()
    if f is None: break
    if time.time() >= nxt:           # fixed fps => virtual time on replay matches reality
        w.write(f); nxt += 1 / a.fps
w.release(); cam.close()
print("saved", a.out)
