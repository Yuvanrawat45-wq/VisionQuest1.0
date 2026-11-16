"""Natural Language Distance & Direction Classification Engine for VisionQuest.

Classifies detected objects into four human-friendly distance categories:
1. VERY CLOSE (<= 1.0m or filling >= 65% frame height)
2. CLOSE      (1.0m < d <= 2.0m or filling >= 40% frame height)
3. MEDIUM     (2.0m < d <= 3.5m)
4. FAR        (> 3.5m)
5. UNCERTAIN  (Truncated or uncalibrated distance)

Provides consistent direction mapping:
- Left third   -> "on your left"
- Middle third -> "in the centre"
- Right third  -> "on your right"

Provides time-to-collision / approach analysis for moving objects.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from .contracts import TrackInfo


CATEGORY_PHRASES = {
    "VERY_CLOSE": "very close to you",
    "CLOSE":      "at close distance",
    "MEDIUM":     "at medium distance",
    "FAR":        "at far distance",
    "UNCERTAIN":  "at an uncertain distance",
}

DIRECTION_PHRASES = {
    "left":  "on your left",
    "ahead": "in the centre",
    "center": "in the centre",
    "centre": "in the centre",
    "right": "on your right",
}


@dataclass
class DistanceInfo:
    category: str                       # 'VERY_CLOSE', 'CLOSE', 'MEDIUM', 'FAR', 'UNCERTAIN'
    phrase: str                         # e.g., 'very close to you', 'at medium distance'
    direction_phrase: str               # e.g., 'on your left', 'in the centre', 'on your right'
    is_reliable: bool                   # False if distance is uncertain
    is_approaching: bool                # True if TTC indicates object is closing in
    is_moving: bool                     # True if moving
    smoothed_dist_m: float              # Smoothed metric distance estimate


class DistanceClassifier:
    """Classifies metric distance into natural language categories with per-track smoothing."""

    def __init__(self, history_size: int = 5, frame_height: float = 480.0):
        self._history: Dict[int, deque] = {}
        self._history_size = history_size
        self._frame_height = frame_height

    def classify(self, track: TrackInfo) -> DistanceInfo:
        """Classify a single TrackInfo into distance and directional categories."""
        # Update distance smoothing history
        hist = self._history.setdefault(track.id, deque(maxlen=self._history_size))
        hist.append(track.dist_m)
        smoothed_dist = float(np.median(hist))

        # Check visual scale (occupancy in frame)
        _, y1, _, y2 = track.box
        bh = max(y2 - y1, 1.0)
        h_frac = bh / self._frame_height

        # Determine approach status
        is_approaching = (track.ttc_s is not None and 0.0 < track.ttc_s <= 4.0)
        is_moving = bool(track.moving is True)

        # Direction phrase
        dir_key = track.direction.lower()
        direction_phrase = DIRECTION_PHRASES.get(dir_key, "in front of you")

        # Visual occupancy override for close/very close objects
        # Large objects in frame are physically close even if monocular prior has slight error
        if h_frac >= 0.65:
            cat = "VERY_CLOSE"
            is_reliable = True
        elif h_frac >= 0.40:
            cat = "CLOSE"
            is_reliable = True
        elif track.truncated and smoothed_dist > 2.5:
            # If truncated on screen borders and far away, height measurement is cut
            cat = "UNCERTAIN"
            is_reliable = False
        else:
            # Metric categorization
            if smoothed_dist <= 1.0:
                cat = "VERY_CLOSE"
                is_reliable = True
            elif smoothed_dist <= 2.0:
                cat = "CLOSE"
                is_reliable = True
            elif smoothed_dist <= 3.5:
                cat = "MEDIUM"
                is_reliable = True
            else:
                cat = "FAR"
                is_reliable = True

        return DistanceInfo(
            category=cat,
            phrase=CATEGORY_PHRASES[cat],
            direction_phrase=direction_phrase,
            is_reliable=is_reliable,
            is_approaching=is_approaching,
            is_moving=is_moving,
            smoothed_dist_m=round(smoothed_dist, 2)
        )

    def reset(self):
        """Clear all historical track distance caches."""
        self._history.clear()

    def prune(self, active_track_ids: List[int]):
        """Clean up state for tracks that are no longer in view."""
        active_set = set(active_track_ids)
        for tid in list(self._history.keys()):
            if tid not in active_set:
                del self._history[tid]


# Global classifier instance
_CLASSIFIER = DistanceClassifier()


def clear_distance_history():
    """Clear historical distance smoothing cache."""
    _CLASSIFIER.reset()


def get_distance_info(track: TrackInfo) -> DistanceInfo:
    """Convenience functional interface for distance classification."""
    return _CLASSIFIER.classify(track)
