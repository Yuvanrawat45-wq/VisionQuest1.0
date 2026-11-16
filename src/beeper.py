"""Asynchronous proximity beep engine — Windows Multimedia WAV edition.

Uses winsound.PlaySound(wav_bytes, SND_MEMORY) on Windows:
- 44100 Hz 16-bit stereo PCM
- Zero subprocess overhead (0 ms latency)
- Clean, audible sine-wave tones with volume scaling per zone
- Runs safely inside a background thread without blocking the camera
- Fallback to sounddevice on Linux/Raspberry Pi
"""

import io
import math
import platform
import struct
import threading
import time
from typing import List, Optional

import numpy as np

from .contracts import TrackInfo


# ---------------------------------------------------------------------------
# Zone table — (max_dist_m, interval_s, amplitude 0.0 - 1.0)
# ---------------------------------------------------------------------------
# Zone 0: Very close (<= 1.0m) -> rapid loud consecutive beeps (100ms apart)
# Zone 1: Close      (<= 1.8m) -> fast beeps (250ms apart)
# Zone 2: Medium     (<= 3.2m) -> moderate beeps (600ms apart)
# Zone 3: Far        (<= 5.0m) -> slow quieter beeps (1200ms apart)
ZONES = [
    (1.0,  0.10, 1.00),   # very close (rapid consecutive)
    (1.8,  0.25, 0.80),   # close
    (3.2,  0.60, 0.55),   # medium
    (5.0,  1.20, 0.30),   # far
]
SILENT_DIST_M = 5.0

BEEP_FREQ_HZ = 880     # A5
BEEP_DUR_MS  = 70      # 70ms crisp tone
SAMPLE_RATE  = 44100   # Standard CD quality stereo
EMA_ALPHA    = 0.40    # Fast, responsive smoothing


# ---------------------------------------------------------------------------
# Pre-build 44100 Hz stereo WAVs for all zones at import time
# ---------------------------------------------------------------------------
def _build_pcm_wav(freq: int, dur_ms: int, amp: float, sr: int = SAMPLE_RATE) -> bytes:
    n = int(sr * dur_ms / 1000)
    t = np.linspace(0, dur_ms / 1000, n, endpoint=False)
    mono = (amp * 32767.0 * np.sin(2.0 * math.pi * freq * t)).astype(np.int16)

    # Smooth fade out on last 20% to prevent clicks
    fade_len = int(n * 0.20)
    if fade_len > 0:
        fade = np.linspace(1.0, 0.0, fade_len)
        mono[-fade_len:] = (mono[-fade_len:] * fade).astype(np.int16)

    # Interleave to stereo (L, R)
    stereo = np.column_stack((mono, mono)).ravel()
    pcm = stereo.tobytes()

    num_channels = 2
    bits_per_sample = 16
    byte_rate = sr * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8

    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(pcm)))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, num_channels, sr, byte_rate, block_align, bits_per_sample))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(pcm)))
    buf.write(pcm)
    return buf.getvalue()


_WAV_CACHE = {
    i: _build_pcm_wav(BEEP_FREQ_HZ, BEEP_DUR_MS, amp)
    for i, (_, __, amp) in enumerate(ZONES)
}

IS_WINDOWS = platform.system() == "Windows"

if IS_WINDOWS:
    import winsound

    def _play_tone(zone_idx: int):
        # SND_MEMORY plays the in-memory WAV buffer synchronously inside the worker thread
        winsound.PlaySound(_WAV_CACHE[zone_idx], winsound.SND_MEMORY)
else:
    import sounddevice as sd

    _FLOAT_WAVES = {}
    for i, (_, __, amp) in enumerate(ZONES):
        n = int(SAMPLE_RATE * BEEP_DUR_MS / 1000)
        t = np.linspace(0, BEEP_DUR_MS / 1000, n, endpoint=False)
        mono = (amp * np.sin(2.0 * math.pi * BEEP_FREQ_HZ * t)).astype(np.float32)
        _FLOAT_WAVES[i] = np.column_stack((mono, mono))

    def _play_tone(zone_idx: int):
        sd.play(_FLOAT_WAVES[zone_idx], samplerate=SAMPLE_RATE, blocking=True)


# ---------------------------------------------------------------------------
def _zone_index(dist_m: float) -> Optional[int]:
    for i, (max_d, _, __) in enumerate(ZONES):
        if dist_m <= max_d:
            return i
    return None


# ---------------------------------------------------------------------------
class ProximityBeeper:
    """Threaded audio proximity beeper."""

    def __init__(self, muted: bool = False, debug: bool = False):
        self._muted  = muted
        self._debug  = debug
        self._paused = False

        self._lock     = threading.Lock()
        self._ema_dist: Optional[float] = None
        self._zone_idx: Optional[int]   = None
        self._no_det   = 0

        self._wake = threading.Event()

        if not muted:
            self._thread = threading.Thread(
                target=self._worker, daemon=True, name="beeper-thread"
            )
            self._thread.start()
            if self._debug:
                print("[beeper] initialized and running")

    @property
    def current_status(self):
        """Return (zone_idx, ema_dist) for HUD/display."""
        with self._lock:
            return self._zone_idx, self._ema_dist

    # ------------------------------------------------------------------
    def update(self, infos: List[TrackInfo]):
        """Call once per frame with detected objects. Non-blocking."""
        if self._muted or self._paused:
            return

        dist = self._nearest_valid_dist(infos)

        if self._debug:
            if infos:
                for i in infos:
                    chosen = (dist is not None and abs(i.dist_m - dist) < 0.001)
                    print(f"[beeper-dbg] {i.name:10s} dist={i.dist_m:.2f}m "
                          f"dir={i.direction:6s} corridor={i.in_corridor} "
                          f"{'<-- NEAREST' if chosen else ''}")
            else:
                print("[beeper-dbg] no detections in frame")

        with self._lock:
            if dist is None:
                self._no_det += 1
                # Only silence after 10 consecutive frames (~0.3s) without any object
                if self._no_det > 10:
                    self._ema_dist = None
                    if self._zone_idx is not None:
                        self._zone_idx = None
                        self._wake.set()
                return

            self._no_det = 0

            # Smooth distance changes
            if self._ema_dist is None:
                self._ema_dist = dist
            else:
                self._ema_dist = EMA_ALPHA * dist + (1.0 - EMA_ALPHA) * self._ema_dist

            new_zone = _zone_index(self._ema_dist)
            old_zone = self._zone_idx
            self._zone_idx = new_zone

        # Wake worker on zone change or transition from silence
        if new_zone is not None:
            if old_zone is None or new_zone < old_zone:
                self._wake.set()

    def pause(self):
        """Mute during microphone / voice Q&A."""
        self._paused = True
        if IS_WINDOWS:
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass

    def resume(self):
        """Resume after voice Q&A."""
        self._paused = False
        self._wake.set()

    def close(self):
        with self._lock:
            self._zone_idx = None
        self._wake.set()
        if not self._muted and hasattr(self, "_thread"):
            self._thread.join(timeout=0.5)

    # ------------------------------------------------------------------
    @staticmethod
    def _nearest_valid_dist(infos: List[TrackInfo]) -> Optional[float]:
        """Find the effective proximity distance of the nearest obstacle in front.

        Accounts for:
        1. Monocular height-based distance.
        2. Cropped person priors (seated/webcam sees upper body ~0.8m, not 1.7m standing).
        3. Apparent visual scale (large bounding boxes touching/filling frame are close).
        """
        best = None
        for i in infos:
            # We care about objects ahead or in the corridor, or any close object
            if not i.in_corridor and i.direction != "ahead" and i.dist_m > 2.0:
                continue

            x1, y1, x2, y2 = i.box
            bh = max(y2 - y1, 1.0)
            bw = max(x2 - x1, 1.0)
            h_frac = bh / 480.0
            w_frac = bw / 640.0

            d = i.dist_m

            # If person is cropped at the bottom (seated at laptop / waist-up),
            # the real visible height is ~0.8m, not the 1.7m full-standing prior.
            if i.name == "person" and y2 >= 450:
                d = d * (0.85 / 1.70)

            # Visual occupancy override: if an object takes up a major portion
            # of the camera frame, it is physically right in front of the lens.
            if h_frac >= 0.65 or w_frac >= 0.55:
                d = min(d, 0.75)   # Zone 0: Very Close (< 1.0m)
            elif h_frac >= 0.45 or w_frac >= 0.38:
                d = min(d, 1.40)   # Zone 1: Close (< 1.8m)
            elif h_frac >= 0.28 or w_frac >= 0.22:
                d = min(d, 2.60)   # Zone 2: Medium (< 3.2m)

            if best is None or d < best:
                best = d

        return best

    def _worker(self):
        while True:
            with self._lock:
                zone_idx = self._zone_idx
                paused   = self._paused

            if zone_idx is None or paused or self._muted:
                self._wake.wait(timeout=0.25)
                self._wake.clear()
                continue

            _, interval_s, amp = ZONES[zone_idx]

            if self._debug:
                with self._lock:
                    curr_d = self._ema_dist or 0.0
                print(f"[beeper] BEEP! zone={zone_idx} dist={curr_d:.2f}m int={interval_s:.2f}s")

            try:
                _play_tone(zone_idx)
            except Exception as e:
                print(f"[beeper] audio play error: {e}")

            # Sleep for the rest of interval (tone is ~75ms)
            remaining = interval_s - (BEEP_DUR_MS / 1000.0)
            if remaining > 0.005:
                self._wake.wait(timeout=remaining)
                self._wake.clear()
