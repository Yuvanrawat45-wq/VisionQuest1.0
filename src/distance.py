
"""Monocular range + robust time-to-collision estimation."""

from collections import deque
from typing import Dict, Optional
import numpy as np


def calibrate_focal(h_px: float, dist_m: float, real_h_m: float) -> float:
    return h_px * dist_m / real_h_m


def estimate_dist(
    name: str,
    h_px: float,
    f_px: float,
    heights: dict,
    default_h: float
) -> float:
    return f_px * heights.get(name, default_h) / max(h_px, 1.0)


class TTCEstimator:
    """
    Approximate TTC using relative bounding-box height growth.

    Uses median smoothing and a robust median of pairwise growth rates.
    This reduces jitter but does not guarantee collision prediction.
    """

    def __init__(
        self,
        window_s: float = 1.2,
        min_samples: int = 5,
        min_duration_s: float = 0.30,
        min_growth_rate: float = 0.02
    ):
        self.window = window_s
        self.min_n = min_samples
        self.min_duration = min_duration_s
        self.min_growth_rate = min_growth_rate

        self.history: Dict[int, deque] = {}
        self.raw: Dict[int, deque] = {}

    def update(self, tid: int, t: float, h_px: float) -> Optional[float]:

        if h_px <= 1 or not np.isfinite(h_px) or not np.isfinite(t):
            return None

        samples = self.history.setdefault(tid, deque())
        raw = self.raw.setdefault(tid, deque(maxlen=3))

        # Reset if timestamps go backwards.
        if samples and t <= samples[-1][0]:
            samples.clear()
            raw.clear()

        raw.append(float(h_px))

        # Median smoothing over the latest three detections.
        smooth_h = float(np.median(raw))
        samples.append((float(t), smooth_h))

        while samples and t - samples[0][0] > self.window:
            samples.popleft()

        if len(samples) < self.min_n:
            return None

        if samples[-1][0] - samples[0][0] < self.min_duration:
            return None

        ts = np.array([item[0] for item in samples])
        hs = np.array([item[1] for item in samples])

        # Log-height slope approximates relative growth rate.
        log_h = np.log(hs)

        rates = []

        for a in range(len(ts)):
            for b in range(a + 1, len(ts)):
                dt = ts[b] - ts[a]

                if dt >= 0.10:
                    rates.append((log_h[b] - log_h[a]) / dt)

        if not rates:
            return None

        growth_rate = float(np.median(rates))

        if not np.isfinite(growth_rate):
            return None

        if growth_rate <= self.min_growth_rate:
            return None

        ttc = 1.0 / growth_rate

        # Reject extremely uncertain estimates.
        if not np.isfinite(ttc) or ttc > 30:
            return None

        return float(ttc)

    def prune(self, alive_ids):
        for k in list(self.history):
            if k not in alive_ids:
                del self.history[k]
                self.raw.pop(k, None)
