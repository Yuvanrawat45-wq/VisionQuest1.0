# VisionProject (repo: visionquest): Merged Plan, Obstacle Awareness

**Date:** 14 Nov 2026 | **Team:** 2 | **Core mode (decided):** obstacle awareness. Scene description is a secondary mode.

## Decisions
- One codebase, hardware-agnostic core (`motion`, `distance`, `policy`, `pipeline`), thin adapters (`camera`, `detector`, `speech`).
- Detector: YOLO11n + ByteTrack (persistent IDs). `.pt` on laptop, NCNN export on Pi 5.
- Moving vs static: subtract local background optical flow (ego-motion compensation).
- TTC (bbox growth) is the trusted danger signal; monocular distance is a rough prior.
- Speech: single prioritized alert, per-track cooldown, urgent preempts. espeak-ng on Pi, pyttsx3 worker thread on Windows.
- Regression tests via recorded clips replayed deterministically.
- Hardware target: Raspberry Pi 5 (supersedes Pi 4 / ESP32-S3 options).

## Interface contract (do not change without telling the other person)
`src/types.py`: `Obs` (detector out), `TrackInfo` (analysis out), `Alert` (policy out).

## Repo reconciliation (what changed vs the original 5 files)
- Original files untouched: `webcam.py`, `camera_test.py`, `distance_test.py`, `tts_test.py`, `yolo11n.pt`. `webcam.py` stays as the known-good fallback.
- Added `src/`, `tools/`, `tests/`, `run.py`, `requirements.txt`, `.gitignore`.
- Kept from `webcam.py`: YOLO11n, `yolo11n.pt` at repo root, spoken phrasing ("on your left" / "in front of you" / "on your right"), 3-zone split, `q` to quit.
- Replaced: speech (thread-per-utterance + `speaking` flag -> single worker/queue), `model(frame)` -> `model.track()` (IDs), per-class cooldown dict -> per-track cooldown.
- `webcam.py` behaviour now lives on as `--mode describe` (secondary mode), but only for stable tracks.
- Dropped: `distance_test.py` constants (`REFERENCE_HEIGHT=300` was an uncalibrated placeholder at unknown resolution). Use the A4 calibration below.
- `src/types.py` was renamed `src/contracts.py`: a file named `types.py` shadows the stdlib `types` when a script inside `src/` is run directly (e.g. VS Code Code Runner) and crashes imports.
- Repo hygiene: `yolo11n.pt` (5.6 MB) is tracked. Harmless, but `git rm --cached yolo11n.pt` + `.gitignore` it later; Ultralytics re-downloads it.

## Run
```
# laptop (Windows, venv interpreter; run from repo root)
.\venv\Scripts\python.exe run.py --source 0 --show --mute --imgsz 640
.\venv\Scripts\python.exe run.py --mode describe --source 0 --show
# replay a recorded clip
.\venv\Scripts\python.exe run.py --source clips\walk_01.mp4 --show --mute --log logs\walk_01.jsonl
# Pi 5
yolo export model=yolo11n.pt format=ncnn imgsz=320
python run.py --source pi --model yolo11n_ncnn_model --show
# tests
python -m unittest discover -s tests -t . -v
```

## Calibration (do first)
1. Print/measure an A4 sheet (29.7 cm tall). Hold it upright at exactly 1.0 m from the camera.
2. Read its pixel height `h_px` from a still frame.
3. `f_px = h_px * 1.0 / 0.297` (`src/distance.calibrate_focal`). Pass with `--focal`.

## Known limits
- COCO has no walls, poles, glass, stairs, drops, curbs. Not a cane replacement.
- Ego-motion test covers pan/translation only. Forward-walk expansion needs real clips.
- Monocular distance is about +/-30-50%. ToF/ultrasonic fusion is phase 2.
- Truncated boxes (touching the frame edge) have their distance ignored.

## Next steps
1. Record 10-20 clips: static obstacle, person crossing, person approaching, wearer panning, wearer walking.
2. Replay and tune `move_thresh_frac_s`, `min_track_age`, thresholds against labelled expectations.
3. Calibrate focal length on the real camera.
4. Pi 5 NCNN benchmark (FPS, CPU temp).
5. Add ToF sensor and fuse range into policy.
6. Haptic output (L/C/R vibration).
7. Custom classes (doors, steps) only after the above is solid.

## Open
- Camera module, mounting position (chest vs head), audio output device.
- Who owns which module (suggested: motion/TTC/policy/Pi port vs speech/calibration/repo/tests).
- Git remote and branch strategy.
