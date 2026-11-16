"""OpenCV-based HSV colour recognition engine for VisionQuest.

Analyzes pixels within object bounding boxes with:
1. Inner-region sampling to avoid background wall/table bleed.
2. Robust HSV binning for red, blue, green, yellow, orange, purple, pink, brown, black, white, grey.
3. Shade classification (dark, light, bright).
4. Lighting/reflection/shadow handling with confidence guardrails.
5. Torso-targeted sampling for person/clothing questions.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np

from .contracts import TrackInfo


@dataclass
class ColorResult:
    name: str                           # e.g., 'blue', 'red', 'black', 'white'
    shade: Optional[str] = None         # 'dark', 'light', 'bright', or None
    confidence: float = 0.0             # 0.0 to 1.0 (proportion of dominant pixels)
    is_reliable: bool = True            # False if poor lighting or low confidence
    reason: Optional[str] = None        # Explanation if uncertain

    @property
    def display_name(self) -> str:
        """User-friendly name, e.g., 'dark blue', 'bright red', 'white'."""
        if not self.is_reliable:
            return "uncertain"
        if self.shade and self.name not in ("black", "white", "grey", "brown"):
            return f"{self.shade} {self.name}"
        return self.name


# ---------------------------------------------------------------------------
# HSV Classification Functions
# ---------------------------------------------------------------------------
def _classify_pixel_hsv(h: int, s: int, v: int) -> str:
    """Classify a single HSV pixel into one of 11 base color categories."""
    # 1. Extreme shadows / darkness
    if v < 42:
        return "black"

    # 2. Desaturated: white or grey
    if s < 38:
        if v >= 190:
            return "white"
        elif v >= 42:
            return "grey"

    # 3. Slightly desaturated mid-brightness
    if s < 50 and 42 <= v <= 180:
        return "grey"

    # 4. Brown (warm hues with low-to-medium brightness and moderate saturation)
    if 8 <= h <= 24 and 50 <= s <= 190 and 40 <= v <= 125:
        return "brown"

    # 5. Chromatic hues (OpenCV H range is 0 to 179)
    if h < 10 or h >= 170:
        return "red"
    elif 10 <= h <= 22:
        return "orange"
    elif 23 <= h <= 35:
        return "yellow"
    elif 36 <= h <= 85:
        return "green"
    elif 86 <= h <= 130:
        return "blue"
    elif 131 <= h <= 148:
        return "purple"
    elif 149 <= h <= 169:
        if v >= 110:
            return "pink"
        return "purple"

    return "unknown"


def _determine_shade(v_mean: float, s_mean: float) -> Optional[str]:
    """Determine shade modifier (dark, light, bright)."""
    if v_mean < 135:
        return "dark"
    if v_mean > 185 and s_mean < 130:
        return "light"
    if v_mean > 185 and s_mean > 175:
        return "bright"
    return None


# ---------------------------------------------------------------------------
# Core Analysis Function
# ---------------------------------------------------------------------------
def analyze_patch_color(patch_bgr: np.ndarray) -> ColorResult:
    """Analyze a BGR image patch and return its dominant colour and shade."""
    if patch_bgr is None or patch_bgr.size == 0 or patch_bgr.shape[0] < 4 or patch_bgr.shape[1] < 4:
        return ColorResult(name="unknown", is_reliable=False, reason="patch too small")

    h, w = patch_bgr.shape[:2]
    total_pixels = h * w

    # Convert to HSV
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = cv2.split(hsv)

    mean_v = float(np.mean(v_chan))
    mean_s = float(np.mean(s_chan))

    # Check poor lighting conditions
    dark_ratio = np.sum(v_chan < 35) / total_pixels
    if dark_ratio > 0.80 or mean_v < 30:
        return ColorResult(name="black", shade="dark", confidence=0.9, is_reliable=True)

    blown_out_ratio = np.sum((v_chan > 240) & (s_chan < 30)) / total_pixels
    if blown_out_ratio > 0.80:
        return ColorResult(name="white", confidence=0.85, is_reliable=True)

    # Classify each pixel
    # Vectorized fast classification for speed
    pixel_counts = {
        "black": int(np.sum(v_chan < 42)),
        "white": int(np.sum((v_chan >= 190) & (s_chan < 38))),
        "grey": int(np.sum((s_chan < 45) & (v_chan >= 42) & (v_chan < 190))),
        "brown": int(np.sum((h_chan >= 8) & (h_chan <= 24) & (s_chan >= 50) & (s_chan <= 190) & (v_chan >= 40) & (v_chan <= 125))),
        "red": int(np.sum(((h_chan < 10) | (h_chan >= 170)) & (s_chan >= 50) & (v_chan >= 42))),
        "orange": int(np.sum((h_chan >= 10) & (h_chan <= 22) & (s_chan >= 60) & (v_chan >= 70))),
        "yellow": int(np.sum((h_chan >= 23) & (h_chan <= 35) & (s_chan >= 60) & (v_chan >= 100))),
        "green": int(np.sum((h_chan >= 36) & (h_chan <= 85) & (s_chan >= 45) & (v_chan >= 42))),
        "blue": int(np.sum((h_chan >= 86) & (h_chan <= 130) & (s_chan >= 45) & (v_chan >= 42))),
        "purple": int(np.sum((h_chan >= 131) & (h_chan <= 148) & (s_chan >= 45) & (v_chan >= 42))),
        "pink": int(np.sum((h_chan >= 149) & (h_chan <= 169) & (s_chan >= 40) & (v_chan >= 110))),
    }

    # Find dominant color
    sorted_colors = sorted(pixel_counts.items(), key=lambda item: item[1], reverse=True)
    dominant_color, dominant_count = sorted_colors[0]
    confidence = dominant_count / float(max(total_pixels, 1))

    # Guardrails: Multi-colored or uncertain
    if confidence < 0.28:
        # Check if top 2 colors combined make sense
        return ColorResult(
            name=dominant_color,
            confidence=round(confidence, 2),
            is_reliable=False,
            reason="multi-coloured or low contrast"
        )

    # Determine shade for dominant color
    shade = _determine_shade(mean_v, mean_s)

    return ColorResult(
        name=dominant_color,
        shade=shade,
        confidence=round(confidence, 2),
        is_reliable=True
    )


# ---------------------------------------------------------------------------
# Public Object Sampling API
# ---------------------------------------------------------------------------
def extract_object_color(
    frame_bgr: np.ndarray,
    box: Tuple[float, float, float, float],
    is_person: bool = False,
    target_clothing: bool = False
) -> ColorResult:
    """Extract and analyze the dominant color from an object's bounding box.

    Avoids background bleed by sampling the inner center region.
    For persons/shirts, samples the torso/chest region.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return ColorResult(name="unknown", is_reliable=False, reason="empty frame")

    frame_h, frame_w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    # Clip coordinates to frame boundaries
    x1 = max(0, min(x1, frame_w - 1))
    x2 = max(0, min(x2, frame_w))
    y1 = max(0, min(y1, frame_h - 1))
    y2 = max(0, min(y2, frame_h))

    bw = x2 - x1
    bh = y2 - y1

    if bw < 8 or bh < 8:
        return ColorResult(name="unknown", is_reliable=False, reason="object too small")

    # Target region selection to avoid edge/background noise:
    if is_person or target_clothing:
        # Torso region: central 50% width, 20% to 65% down from top
        cx1 = int(x1 + 0.25 * bw)
        cx2 = int(x1 + 0.75 * bw)
        cy1 = int(y1 + 0.20 * bh)
        cy2 = int(y1 + 0.65 * bh)
    else:
        # Standard object: inner 60% box
        cx1 = int(x1 + 0.20 * bw)
        cx2 = int(x1 + 0.80 * bw)
        cy1 = int(y1 + 0.20 * bh)
        cy2 = int(y1 + 0.80 * bh)

    # Ensure valid patch
    cx1 = max(0, min(cx1, frame_w - 1))
    cx2 = max(cx1 + 2, min(cx2, frame_w))
    cy1 = max(0, min(cy1, frame_h - 1))
    cy2 = max(cy1 + 2, min(cy2, frame_h))

    patch = frame_bgr[cy1:cy2, cx1:cx2]
    return analyze_patch_color(patch)


def get_scene_dominant_color(frame_bgr: np.ndarray) -> ColorResult:
    """Sample the overall camera scene and return its dominant colour."""
    if frame_bgr is None or frame_bgr.size == 0:
        return ColorResult(name="unknown", is_reliable=False, reason="empty frame")

    # Downsample for fast execution
    small = cv2.resize(frame_bgr, (160, 120), interpolation=cv2.INTER_AREA)
    return analyze_patch_color(small)


def find_objects_with_color(
    frame_bgr: np.ndarray,
    infos: List[TrackInfo],
    target_color: str
) -> List[Tuple[TrackInfo, ColorResult]]:
    """Find all detected objects whose dominant colour matches target_color."""
    matches = []
    target_lower = target_color.lower().strip()

    for item in infos:
        color_res = extract_object_color(frame_bgr, item.box, is_person=(item.name == "person"))
        if color_res.is_reliable and color_res.name == target_lower:
            matches.append((item, color_res))

    return matches
