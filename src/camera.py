"""Camera adapters. read() -> (frame_bgr, timestamp_s) or (None, 0).
Video files use virtual time (frame_idx / fps) so replays are deterministic."""
import time
import cv2


class OpenCVCamera:
    def __init__(self, source=0, size=(640, 480)):
        self.size = size
        self.is_file = isinstance(source, str)
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera/video source: {source!r}")
        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n = 0

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None, 0.0
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        t = self.n / self.fps if self.is_file else time.time()
        self.n += 1
        return frame, t

    def close(self):
        self.cap.release()


class PiCamera:
    def __init__(self, size=(640, 480)):
        from picamera2 import Picamera2  # only exists on the Pi
        self.cam = Picamera2()
        self.cam.configure(self.cam.create_video_configuration(
            main={"size": size, "format": "RGB888"}))   # RGB888 == BGR byte order for OpenCV
        self.cam.start()
        time.sleep(1.0)

    def read(self):
        return self.cam.capture_array(), time.time()

    def close(self):
        self.cam.stop()


def make_camera(source: str, size=(640, 480)):
    """source: 'pi' | '0' (webcam index) | path to a video file."""
    if source == "pi":
        return PiCamera(size)
    return OpenCVCamera(int(source) if source.isdigit() else source, size)
