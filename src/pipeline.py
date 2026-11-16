"""Glue: frame + detections -> TrackInfo list -> Alert. No camera/speaker/model imports."""
from typing import List, Optional, Tuple
import cv2
from .config import Config
from .contracts import Alert, Obs, TrackInfo
from .motion import MotionClassifier, background_flow, local_flow
from .distance import TTCEstimator, estimate_dist
from . import policy


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.motion = MotionClassifier(cfg.move_thresh_frac_s * cfg.width,
                                       cfg.min_track_age, cfg.motion_ema_alpha)
        self.ttc = TTCEstimator()
        self.prev_gray = None
        self.prev_t: Optional[float] = None
        self.last_spoken = {}

    def mark_spoken(self, key, t: float):
        self.last_spoken[key] = t

    def step(self, frame, t: float, obs: List[Obs]) -> Tuple[List[TrackInfo], Optional[Alert]]:
        c = self.cfg
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dt = (t - self.prev_t) if self.prev_t is not None else 1 / 30
        pts = flows = None
        if self.prev_gray is not None:
            pts, flows = background_flow(self.prev_gray, gray, [o.box for o in obs])

        infos = []
        for o in obs:
            x1, y1, x2, y2 = o.box
            cx, bh = (x1 + x2) / 2, max(y2 - y1, 1.0)
            lf = local_flow(pts, flows, o.box)
            age, moving = self.motion.update(o.id, ((x1 + x2) / 2, (y1 + y2) / 2), lf, dt)
            ttc = self.ttc.update(o.id, t, bh)
            infos.append(TrackInfo(
                id=o.id, name=o.name, box=o.box, age=age, moving=moving,
                dist_m=estimate_dist(o.name, bh, c.focal_px, c.real_h, c.default_h),
                ttc_s=ttc,
                direction="left" if cx < c.width / 3 else "right" if cx > 2 * c.width / 3 else "ahead",
                in_corridor=x2 > c.corridor_l * c.width and x1 < c.corridor_r * c.width,
                truncated=(y2 > c.height - 4 or y1 < 4 or x1 < 4 or x2 > c.width - 4)))

        alive = {o.id for o in obs}
        self.motion.prune(alive)
        self.ttc.prune(alive)
        self.prev_gray, self.prev_t = gray, t
        fn = policy.decide if c.mode == "obstacle" else policy.decide_describe
        return infos, fn(infos, c, t, self.last_spoken)
