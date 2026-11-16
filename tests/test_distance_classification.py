"""Unit tests for Natural Language Distance & Direction Classification."""

import unittest
import numpy as np

from src.contracts import TrackInfo
from src.intent import parse
from src.qa import answer
from src.distance_categories import get_distance_info, clear_distance_history


YOLO_NAMES = ["person", "bottle", "cup", "backpack", "car", "chair", "cell phone", "laptop"]


def make_track(
    name: str,
    direction: str = "ahead",
    dist_m: float = 1.2,
    box: tuple = (100, 100, 200, 200),
    truncated: bool = False,
    ttc_s: float = None,
    moving: bool = False,
    tid: int = 1
) -> TrackInfo:
    return TrackInfo(
        id=tid,
        name=name,
        box=box,
        age=10,
        moving=moving,
        dist_m=dist_m,
        ttc_s=ttc_s,
        direction=direction,
        in_corridor=True,
        truncated=truncated
    )


class TestDistanceClassification(unittest.TestCase):

    def setUp(self):
        clear_distance_history()
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # -----------------------------------------------------------------------
    # 1. Category Classification & Phrasing
    # -----------------------------------------------------------------------
    def test_distance_categories(self):
        # Very close: <= 1.0m
        t_vc = make_track("chair", dist_m=0.8, tid=1)
        d_vc = get_distance_info(t_vc)
        self.assertEqual(d_vc.category, "VERY_CLOSE")
        self.assertIn("very close", d_vc.phrase)

        # Close: 1.0 < d <= 2.0m
        t_c = make_track("bottle", dist_m=1.4, tid=2)
        d_c = get_distance_info(t_c)
        self.assertEqual(d_c.category, "CLOSE")
        self.assertIn("close distance", d_c.phrase)

        # Medium: 2.0 < d <= 3.5m
        t_m = make_track("laptop", dist_m=2.8, tid=3)
        d_m = get_distance_info(t_m)
        self.assertEqual(d_m.category, "MEDIUM")
        self.assertIn("medium distance", d_m.phrase)

        # Far: > 3.5m
        t_f = make_track("car", dist_m=4.5, tid=4)
        d_f = get_distance_info(t_f)
        self.assertEqual(d_f.category, "FAR")
        self.assertIn("far distance", d_f.phrase)

    # -----------------------------------------------------------------------
    # 2. Directions: Left, Centre, Right
    # -----------------------------------------------------------------------
    def test_directional_phrases(self):
        t_l = make_track("bottle", direction="left", dist_m=1.5)
        d_l = get_distance_info(t_l)
        self.assertEqual(d_l.direction_phrase, "on your left")

        t_c = make_track("person", direction="ahead", dist_m=0.7)
        d_c = get_distance_info(t_c)
        self.assertEqual(d_c.direction_phrase, "in the centre")

        t_r = make_track("chair", direction="right", dist_m=2.5)
        d_r = get_distance_info(t_r)
        self.assertEqual(d_r.direction_phrase, "on your right")

    # -----------------------------------------------------------------------
    # 3. Voice: "How far is my bottle?"
    # -----------------------------------------------------------------------
    def test_how_far_query(self):
        t = make_track("bottle", direction="left", dist_m=2.5)
        intent = parse("how far is my bottle", YOLO_NAMES)
        self.assertEqual(intent.kind, "how_far")

        resp = answer(intent, [t], YOLO_NAMES, frame_bgr=self.frame)
        self.assertIn("bottle is on your left", resp)
        self.assertIn("medium distance", resp)

    # -----------------------------------------------------------------------
    # 4. Voice: "How far is the person in front of me?"
    # -----------------------------------------------------------------------
    def test_how_far_person_in_front(self):
        t = make_track("person", direction="ahead", dist_m=0.8)
        intent = parse("how far is the person in front of me", YOLO_NAMES)
        self.assertEqual(intent.kind, "how_far")

        resp = answer(intent, [t], YOLO_NAMES, frame_bgr=self.frame)
        self.assertIn("person is in the centre", resp)
        self.assertIn("very close", resp)

    # -----------------------------------------------------------------------
    # 5. Voice: "Is the chair close?"
    # -----------------------------------------------------------------------
    def test_is_distance_verify_close(self):
        t_close = make_track("chair", direction="left", dist_m=1.2)
        intent = parse("is the chair close", YOLO_NAMES)
        self.assertEqual(intent.kind, "is_distance_verify")

        resp = answer(intent, [t_close], YOLO_NAMES, frame_bgr=self.frame)
        self.assertTrue(resp.startswith("Yes,"))
        self.assertIn("on your left", resp)
        self.assertIn("close distance", resp)

    # -----------------------------------------------------------------------
    # 6. Voice: "Which object is nearest?"
    # -----------------------------------------------------------------------
    def test_which_object_is_nearest(self):
        t_far = make_track("laptop", direction="right", dist_m=3.0, tid=10)
        t_near = make_track("chair", direction="left", dist_m=0.7, tid=11)
        intent = parse("which object is nearest", YOLO_NAMES)
        self.assertEqual(intent.kind, "nearest_object")

        resp = answer(intent, [t_far, t_near], YOLO_NAMES, frame_bgr=self.frame)
        self.assertIn("nearest detected object appears to be a chair", resp)
        self.assertIn("on your left", resp)
        self.assertIn("very close", resp)

    # -----------------------------------------------------------------------
    # 7. Voice: "Which object is closest to my left?"
    # -----------------------------------------------------------------------
    def test_nearest_on_left(self):
        t_left1 = make_track("bottle", direction="left", dist_m=2.2, tid=12)
        t_left2 = make_track("chair", direction="left", dist_m=1.1, tid=13)
        t_right = make_track("cup", direction="right", dist_m=0.5, tid=14)

        intent = parse("which object is closest to my left", YOLO_NAMES)
        self.assertEqual(intent.kind, "nearest_object")
        self.assertEqual(intent.direction_filter, "left")

        resp = answer(intent, [t_left1, t_left2, t_right], YOLO_NAMES, frame_bgr=self.frame)
        self.assertIn("chair", resp)
        self.assertIn("on your left", resp)

    # -----------------------------------------------------------------------
    # 8. Voice: "Compare the distance between my bottle and laptop"
    # -----------------------------------------------------------------------
    def test_compare_distance(self):
        t_bottle = make_track("bottle", direction="left", dist_m=1.2, tid=15)
        t_laptop = make_track("laptop", direction="ahead", dist_m=3.8, tid=16)

        intent = parse("compare the distance between my bottle and laptop", YOLO_NAMES)
        self.assertEqual(intent.kind, "compare_distance")

        resp = answer(intent, [t_bottle, t_laptop], YOLO_NAMES, frame_bgr=self.frame)
        self.assertIn("bottle is closer than your laptop", resp)
        self.assertIn("close distance", resp)
        self.assertIn("far distance", resp)

    # -----------------------------------------------------------------------
    # 9. Voice: "Is the person getting closer?" (Approaching)
    # -----------------------------------------------------------------------
    def test_is_person_getting_closer(self):
        # TTC indicates closing in
        t_approaching = make_track("person", direction="ahead", dist_m=2.0, ttc_s=1.8, moving=True)
        intent = parse("is the person getting closer", YOLO_NAMES)
        self.assertEqual(intent.kind, "is_approaching")

        resp = answer(intent, [t_approaching], YOLO_NAMES, frame_bgr=self.frame)
        self.assertIn("appears to be getting closer", resp)

    # -----------------------------------------------------------------------
    # 10. Combined: "How far is my blue bottle?" (Colour + Distance + Direction)
    # -----------------------------------------------------------------------
    def test_how_far_colored_object(self):
        # Paint bottle blue
        self.frame[100:200, 100:200] = (255, 0, 0)
        t_blue = make_track("bottle", direction="right", dist_m=1.3, box=(100, 100, 200, 200))

        intent = parse("how far is my blue bottle", YOLO_NAMES)
        self.assertEqual(intent.kind, "how_far")
        self.assertEqual(intent.query_color, "blue")

        resp = answer(intent, [t_blue], YOLO_NAMES, frame_bgr=self.frame)
        self.assertIn("blue bottle", resp)
        self.assertIn("on your right", resp)
        self.assertIn("close distance", resp)


if __name__ == "__main__":
    unittest.main()
