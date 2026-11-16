"""Shared data contracts. Both teammates code against these, nothing else."""
from dataclasses import dataclass
from typing import Optional, Tuple

Box = Tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


@dataclass
class Obs:
    """One tracked detection straight out of the detector."""
    id: int
    name: str
    box: Box
    conf: float


@dataclass
class TrackInfo:
    """Fully analysed track, input to the alert policy."""
    id: int
    name: str
    box: Box
    age: int
    moving: Optional[bool]       # None until age >= min_track_age
    dist_m: float
    ttc_s: Optional[float]       # None if not closing
    direction: str               # "left" | "ahead" | "right"
    in_corridor: bool
    truncated: bool              # box touches frame edge => height underestimates


@dataclass
class Alert:
    tid: int
    text: str
    urgent: bool
    key: object = None           # cooldown key; defaults to tid

    def __post_init__(self):
        if self.key is None:
            self.key = self.tid
