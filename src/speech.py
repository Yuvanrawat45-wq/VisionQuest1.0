"""Speech backends behind one interface: say(text, urgent) -> bool (accepted).
Never blocks the vision loop. Only the most recent/urgent message matters."""
import importlib.util
import queue
import shutil
import subprocess
import sys
import threading


class PrintSpeaker:
    """Dry-run / fallback: prints instead of speaking."""
    def say(self, text, urgent=False):
        print(("[URGENT] " if urgent else "[SAY] ") + text)
        return True

    def close(self):
        pass


class EspeakSpeaker:
    """Pi/Linux. Non-blocking subprocess; urgent kills current speech."""
    def __init__(self):
        self.proc = None

    def _busy(self):
        return self.proc is not None and self.proc.poll() is None

    def say(self, text, urgent=False):
        if self._busy():
            if not urgent:
                return False
            self.proc.terminate()
        self.proc = subprocess.Popen(["espeak-ng", "-s", "175", "-a", "200", text],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True

    def close(self):
        if self._busy():
            self.proc.terminate()


class Pyttsx3Speaker:
    """Laptop TTS. Each utterance gets a fresh engine process because repeated
    pyttsx3 runAndWait calls can stop producing audio on Windows."""
    _SCRIPT = (
        "import sys, pyttsx3; "
        "engine = pyttsx3.init(); "
        "engine.setProperty('rate', 190); "
        "engine.say(sys.argv[1]); "
        "engine.runAndWait()"
    )

    def __init__(self):
        if importlib.util.find_spec("pyttsx3") is None:
            raise ModuleNotFoundError("pyttsx3 is not installed")
        self.q = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self.proc = None
        self.speaking = False
        self.closed = False
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            text = self.q.get()
            if text is None:
                return
            try:
                with self._lock:
                    if self.closed:
                        return
                    self.proc = subprocess.Popen(
                        [sys.executable, "-c", self._SCRIPT, text],
                        stdin=subprocess.DEVNULL
                    )
                    proc = self.proc
                    self.speaking = True
            except OSError as e:
                print(f"[speech] failed to start TTS process: {e}")
                continue

            returncode = proc.wait()
            with self._lock:
                if self.proc is proc:
                    self.proc = None
                self.speaking = False
                closed = self.closed
            if returncode != 0:
                print(f"[speech] TTS process exited with status {returncode}")
            if closed:
                return

    def say(self, text, urgent=False):
        with self._lock:
            if self.closed:
                return False
            if urgent:
                try:
                    while True:
                        self.q.get_nowait()
                except queue.Empty:
                    pass
                if self.speaking and self.proc is not None and self.proc.poll() is None:
                    self.proc.terminate()
            elif not self.q.empty():
                return False
            try:
                self.q.put_nowait(text)
                return True
            except queue.Full:
                return False

    def close(self):
        with self._lock:
            self.closed = True
            if self.speaking and self.proc is not None and self.proc.poll() is None:
                self.proc.terminate()
            try:
                while True:
                    self.q.get_nowait()
            except queue.Empty:
                pass
            self.q.put_nowait(None)


def make_speaker(mute: bool = False):
    if mute:
        return PrintSpeaker()
    if shutil.which("espeak-ng"):
        return EspeakSpeaker()
    try:
        return Pyttsx3Speaker()
    except Exception as e:
        print(f"[speech] no TTS backend ({e}); falling back to print")
        return PrintSpeaker()
