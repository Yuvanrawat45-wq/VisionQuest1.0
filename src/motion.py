"""Moving vs static, compensated for the wearer's own motion."""
from typing import Dict
import numpy as np

try:
    import cv2
except ImportError:  # keeps classifier importable without OpenCV
    cv2 = None


def background_flow(prev_gray, gray, boxes, max_pts: int = 150):
    """LK optical flow on feature points OUTSIDE all boxes (= static world).
    Returns (pts Nx2, flows Nx2) or (None, None)."""
    mask = np.full(gray.shape, 255, np.uint8)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 0, -1)
    p0 = cv2.goodFeaturesToTrack(prev_gray, max_pts, 0.01, 10, mask=mask)
    if p0 is None or len(p0) < 8:
        return None, None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, winSize=(21, 21), maxLevel=2)
    ok = st.ravel() == 1
    if ok.sum() < 8:
        return None, None
    p0 = p0.reshape(-1, 2)[ok]
    p1 = p1.reshape(-1, 2)[ok]
    return p0, p1 - p0


def local_flow(pts, flows, box, min_pts: int = 4) -> np.ndarray:
    """Median background flow near a box (falls back to global median)."""
    if pts is None or len(pts) == 0:
        return np.zeros(2)
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    m = ((pts[:, 0] > x1 - bw) & (pts[:, 0] < x2 + bw) &
         (pts[:, 1] > y1 - bh) & (pts[:, 1] < y2 + bh))
    f = flows[m] if m.sum() >= min_pts else flows
    return np.median(f, axis=0)


class MotionClassifier:
    """Per-track residual-motion EMA. Pure numpy: unit-testable without hardware."""

    def __init__(self, thresh_px_s: float, min_age: int, alpha: float = 0.4):
        self.thresh, self.min_age, self.alpha = thresh_px_s, min_age, alpha
        self.s: Dict[int, dict] = {}

    def update(self, tid: int, center, lf, dt: float):
        """Returns (age, moving|None)."""
        st = self.s.get(tid)
        if st is None:
            self.s[tid] = {"c": np.asarray(center, float), "ema": 0.0, "age": 0}
            return 0, None
        c = np.asarray(center, float)
        resid = float(np.linalg.norm((c - st["c"]) - np.asarray(lf))) / max(dt, 1e-3)
        st["ema"] = (1 - self.alpha) * st["ema"] + self.alpha * resid
        st["c"] = c
        st["age"] += 1
        moving = (st["ema"] > self.thresh) if st["age"] >= self.min_age else None
        return st["age"], moving

    def prune(self, alive_ids):
        for k in list(self.s):
            if k not in alive_ids:
                del self.s[k]
