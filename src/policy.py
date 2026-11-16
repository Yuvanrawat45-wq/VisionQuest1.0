
"""Prioritized obstacle alerts using TTC and rough range estimates."""

from typing import Dict, List, Optional
from .config import Config
from .contracts import Alert, TrackInfo

PHRASE = {
    "left": "on your left",
    "ahead": "in front of you",
    "right": "on your right"
}


def decide(
    infos: List[TrackInfo],
    cfg: Config,
    now: float,
    last_spoken: Dict[object, float]
) -> Optional[Alert]:

    candidates = []

    for i in infos:

        if i.age < cfg.min_track_age:
            continue

        # TTC is useful for identifying approaching objects.
        closing = (
            i.ttc_s is not None
            and 0 < i.ttc_s <= cfg.ttc_alert_s
        )

        urgent = (
            closing
            and i.ttc_s <= cfg.ttc_urgent_s
            and i.in_corridor
        )

        # Monocular distance remains a rough fallback.
        limit = (
            cfg.dist_moving_m
            if i.moving is True
            else cfg.dist_static_m
        )

        range_warning = (
            not i.truncated
            and i.dist_m <= limit
        )

        # Prioritize objects in the walking corridor.
        corridor_danger = (
            i.in_corridor and (closing or range_warning)
        )

        # Also consider approaching moving objects outside
        # the corridor because their paths may cross.
        side_danger = i.moving is True and closing

        if not (corridor_danger or side_danger):
            continue

        gap = now - last_spoken.get(i.id, -1e9)

        required_gap = (
            cfg.urgent_min_gap_s if urgent
            else cfg.cooldown_s
        )

        if gap < required_gap:
            continue

        # Lower score = higher priority.
        ttc_priority = (
            i.ttc_s if closing else float("inf")
        )

        range_priority = (
            i.dist_m if range_warning else float("inf")
        )

        candidates.append((
            0 if urgent else 1,
            ttc_priority,
            range_priority,
            i
        ))

    if not candidates:
        return None

    _, _, _, i = min(
        candidates,
        key=lambda item: item[:3]
    )

    if i.ttc_s is not None and 0 < i.ttc_s <= cfg.ttc_alert_s:
        message = f"{i.name} approaching {PHRASE[i.direction]}"
    elif i.moving is True:
        message = f"{i.name} moving {PHRASE[i.direction]}"
    else:
        message=f"{i.name} {PHRASE[i.direction]}"

    if i.ttc_s is not None and i.ttc_s <= cfg.ttc_urgent_s and i.in_corridor:
        message="Stop. "+message

    return Alert(
        tid=i.id,
        text=message,
        urgent=(
            i.ttc_s is not None
            and 0 < i.ttc_s <= cfg.ttc_urgent_s
            and i.in_corridor
        )
    )


def decide_describe(
    infos: List[TrackInfo],
    cfg: Config,
    now: float,
    last_spoken: Dict[object, float]
) -> Optional[Alert]:

    best = None

    for i in infos:
        moving = i.moving is True
        key = (i.id, i.direction, moving)

        if now - last_spoken.get(key, -1e9) < cfg.cooldown_s:
            continue

        x1, y1, x2, y2 = i.box
        area = (x2 - x1) * (y2 - y1)

        if best is None or area > best[0]:
            best = (area, i, key)

    if best is None:
        return None

    _, i, key = best

    message = (
        f"{i.name} is moving {PHRASE[i.direction]}"
        if i.moving
        else f"{i.name} {PHRASE[i.direction]}"
    )

    return Alert(
        tid=i.id,
        text=message,
        urgent=False,
        key=key
    )
