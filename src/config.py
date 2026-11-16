import math
from dataclasses import dataclass, field


@dataclass
class Config:
    # --- camera ---
    width: int = 640
    height: int = 480
    hfov_deg: float = 66.0          # Pi Cam v3 standard=66, v3 wide=102, laptop webcam ~60-78
    focal_px_override: float = 616.16 # set from calibration to beat the HFOV guess

    # --- detector ---
    model_path: str = "yolo11n.pt"  # on Pi: "yolo11n_ncnn_model"
    imgsz: int = 320
    conf: float = 0.40

    # --- behaviour: "obstacle" (core) or "describe" (their original what/where announcer) ---
    mode: str = "describe"

    # --- corridor (walking path) as fractions of frame width ---
    corridor_l: float = 0.30
    corridor_r: float = 0.70

    # --- motion classification ---
    move_thresh_frac_s: float = 0.08   # residual px/s as fraction of frame width
    min_track_age: int = 6             # frames before classifying / alerting
    motion_ema_alpha: float = 0.4

    # --- alerting ---
    dist_static_m: float = 2.0
    dist_moving_m: float = 4.0
    ttc_alert_s: float = 3.0
    ttc_urgent_s: float = 1.5
    cooldown_s: float = 4.0
    urgent_min_gap_s: float = 1.0

    # --- real-world heights (m) for monocular distance (rough priors) ---
    real_h: dict = field(default_factory=lambda: {
        "person": 1.7, "bicycle": 1.0, "car": 1.5, "motorcycle": 1.1, "bus": 3.0,
        "truck": 3.0, "dog": 0.5, "cat": 0.3, "chair": 0.9, "bench": 0.9,
        "couch": 0.9, "potted plant": 0.5, "fire hydrant": 0.7, "stop sign": 2.0,
        "suitcase": 0.6, "backpack": 0.5, "dining table": 0.75, "tv": 0.5,
        "bottle": 0.25, "umbrella": 0.9})
    default_h: float = 1.0

    # --- proximity beep ---
    beep_max_dist_m: float = 5.0        # objects beyond this are silent
    beep_smooth_alpha: float = 0.30     # EMA weight for new distance readings
    beep_freq_hz: int = 880             # A5 tone — clear and easy to hear
    beep_max_vol_pct: int = 90          # safety ceiling (sounddevice path)

    # --- voice Q&A ---
    qa_record_secs: int = 5             # microphone window length
    qa_language: str = "en-IN"         # STT language code

    @property
    def focal_px(self) -> float:
        if self.focal_px_override > 0:
            return self.focal_px_override
        return (self.width / 2) / math.tan(math.radians(self.hfov_deg / 2))
