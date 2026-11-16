"""Microphone capture + Google Speech Recognition wrapper.

Records a fixed-duration audio clip in a background thread so the
camera loop is not blocked.  Returns the recognised text (or None).

Dependencies already in the venv:
  sounddevice  >= 0.4
  SpeechRecognition >= 3.8
"""

import threading
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import speech_recognition as sr


SAMPLE_RATE  = 16_000   # Hz — required by Google STT
RECORD_SECS  = 5        # maximum recording window
CHANNELS     = 1


class Listener:
    """Non-blocking microphone listener.

    Usage:
        listener = Listener()
        listener.listen_async(callback)   # fires callback(text_or_None)
    """

    def __init__(self):
        self._recognizer = sr.Recognizer()
        self._recognizer.pause_threshold = 1.0   # seconds of silence = end of phrase
        self._busy = False
        self._lock = threading.Lock()

    @property
    def busy(self) -> bool:
        return self._busy

    def listen_async(self, callback: Callable[[Optional[str]], None]):
        """Start recording in a daemon thread; call callback(text | None) when done."""
        with self._lock:
            if self._busy:
                return   # already listening
            self._busy = True

        t = threading.Thread(target=self._record_and_recognise, args=(callback,), daemon=True)
        t.start()

    # ------------------------------------------------------------------
    def _record_and_recognise(self, callback: Callable[[Optional[str]], None]):
        text = None
        try:
            print("[listener] Recording…  Speak now.")
            audio_np = sd.rec(
                int(RECORD_SECS * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16"
            )
            sd.wait()
            print("[listener] Processing speech…")

            audio_data = sr.AudioData(audio_np.tobytes(), SAMPLE_RATE, 2)
            text = self._recognizer.recognize_google(audio_data, language="en-IN")
            print(f"[listener] Heard: {text!r}")

        except sr.UnknownValueError:
            print("[listener] Could not understand speech.")
        except sr.RequestError as exc:
            print(f"[listener] STT service error: {exc}")
        except Exception as exc:
            print(f"[listener] Microphone error: {exc}")
        finally:
            with self._lock:
                self._busy = False
            callback(text)
