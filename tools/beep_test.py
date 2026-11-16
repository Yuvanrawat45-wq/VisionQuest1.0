r"""Standalone beep progression test — no camera needed.
Directly tests the 4 audio distance zones.

Run with:
    e:\VisionProject\venv\Scripts\python.exe tools\beep_test.py
"""
import sys
import time

sys.path.insert(0, ".")

from src.beeper import ProximityBeeper
from src.contracts import TrackInfo

ZONE_TESTS = [
    ("Zone 0 (Very Close, <=1.0m) -> Rapid, loud consecutive beeps", 0.5, 2.5),
    ("Zone 1 (Close, <=1.8m)      -> Faster, medium-high beeps",      1.4, 2.5),
    ("Zone 2 (Medium, <=3.2m)     -> Moderate interval beeps",        2.4, 2.5),
    ("Zone 3 (Far, <=5.0m)        -> Slow, quiet beeps",              4.0, 3.0),
]


def make_track(dist_m):
    return TrackInfo(
        id=1,
        name="person",
        box=(200, 100, 440, 400),
        age=15,
        moving=False,
        dist_m=dist_m,
        ttc_s=None,
        direction="ahead",
        in_corridor=True,
        truncated=False,
    )


def main():
    print("=" * 60)
    print("VISIONQUEST PROXIMITY BEEP TEST")
    print("Testing 4 proximity audio zones:")
    print("  Very Close (rapid/loud) -> Close -> Medium -> Far (slow/quiet)")
    print("=" * 60)

    beeper = ProximityBeeper(muted=False, debug=True)
    time.sleep(0.1)

    for description, dist, duration in ZONE_TESTS:
        print(f"\n>>> PLAYING: {description} (dist={dist}m)")
        start = time.time()
        while time.time() - start < duration:
            beeper.update([make_track(dist)])
            time.sleep(0.04)

    beeper.close()
    print("\n" + "=" * 60)
    print("Test completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
