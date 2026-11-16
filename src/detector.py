from typing import List
from .contracts import Obs


class Detector:
    """YOLO + ByteTrack. Persistent IDs come from model.track(persist=True)."""

    def __init__(self, model_path: str, imgsz: int, conf: float):
        from ultralytics import YOLO
        self.model = YOLO(model_path, task="detect")
        self.imgsz, self.conf = imgsz, conf

    def track(self, frame) -> List[Obs]:
        r = self.model.track(frame, persist=True, tracker="bytetrack.yaml",
                             imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        b = r.boxes
        if b is None or b.id is None:
            return []
        ids = b.id.int().cpu().tolist()
        xyxy = b.xyxy.cpu().numpy()
        cls = b.cls.int().cpu().tolist()
        cf = b.conf.cpu().tolist()
        return [Obs(i, self.model.names[c], tuple(map(float, bx)), p)
                for i, c, bx, p in zip(ids, cls, xyxy, cf)]
