"""Unit tests for Colour Recognition voice interactions."""

import unittest
import numpy as np

from src.contracts import TrackInfo
from src.intent import parse
from src.qa import answer


YOLO_NAMES = ["person", "bottle", "cup", "backpack", "car", "chair", "cell phone"]


def make_track(name, direction="ahead", dist_m=1.2, box=(50, 50, 150, 150)):
    return TrackInfo(
        id=1,
        name=name,
        box=box,
        age=10,
        moving=False,
        dist_m=dist_m,
        ttc_s=None,
        direction=direction,
        in_corridor=True,
        truncated=False
    )


class TestColorVoiceQA(unittest.TestCase):

    def setUp(self):
        # Create a standard 200x200 canvas
        self.frame = np.zeros((200, 200, 3), dtype=np.uint8)

    # -----------------------------------------------------------------------
    # 1. Specific Object Colour ("What colour is my bottle?")
    # -----------------------------------------------------------------------
    def test_object_color_question(self):
        # Paint bottle blue
        self.frame[50:150, 50:150] = (255, 0, 0) # BGR Blue
        track = [make_track("bottle", "ahead", 1.0, (50, 50, 150, 150))]

        intent = parse("what colour is my bottle", YOLO_NAMES)
        self.assertEqual(intent.kind, "object_color")
        self.assertIn("bottle", intent.targets)

        resp = answer(intent, track, YOLO_NAMES, frame_bgr=self.frame)
        self.assertIn("bottle", resp)
        self.assertIn("blue", resp)

    # -----------------------------------------------------------------------
    # 2. Shirt / Clothing Colour ("What colour is my shirt?")
    # -----------------------------------------------------------------------
    def test_shirt_color_question(self):
        # Person box from y:10..190, x:50..150. Torso is around y:50..130
        self.frame[50:130, 60:140] = (0, 0, 255) # BGR Red shirt
        track = [make_track("person", "ahead", 1.0, (50, 10, 150, 190))]

        intent = parse("what colour is my shirt", YOLO_NAMES)
        self.assertEqual(intent.kind, "object_color")
        self.assertTrue(intent.target_clothing)

        resp = answer(intent, track, YOLO_NAMES, frame_bgr=self.frame)
        self.assertIn("shirt appears to be", resp)
        self.assertIn("red", resp)

    # -----------------------------------------------------------------------
    # 3. Colour Choice ("Is this object red or blue?")
    # -----------------------------------------------------------------------
    def test_color_choice_question(self):
        # Object is green
        self.frame[50:150, 50:150] = (0, 255, 0) # BGR Green
        track = [make_track("bottle", "ahead", 1.0, (50, 50, 150, 150))]

        intent = parse("is this object red or blue", YOLO_NAMES)
        self.assertEqual(intent.kind, "color_choice")
        self.assertIn("red", intent.color_choices)
        self.assertIn("blue", intent.color_choices)

        resp = answer(intent, track, YOLO_NAMES, frame_bgr=self.frame)
        self.assertIn("neither red nor blue", resp)
        self.assertIn("green", resp)

    # -----------------------------------------------------------------------
    # 4. Find by Colour ("Find something green")
    # -----------------------------------------------------------------------
    def test_find_by_color(self):
        # Blue bottle at box1
        self.frame[20:80, 20:80] = (255, 0, 0)
        # Green cup at box2
        self.frame[120:180, 120:180] = (0, 255, 0)

        t_bottle = make_track("bottle", "left", 1.5, (20, 20, 80, 80))
        t_cup    = make_track("cup", "right", 1.0, (120, 120, 180, 180))

        intent = parse("find something green", YOLO_NAMES)
        self.assertEqual(intent.kind, "find_by_color")
        self.assertEqual(intent.query_color, "green")

        resp = answer(intent, [t_bottle, t_cup], YOLO_NAMES, frame_bgr=self.frame)
        self.assertIn("green cup on your right", resp)

    # -----------------------------------------------------------------------
    # 5. Colour Comparison ("Are my bottle and bag the same colour?")
    # -----------------------------------------------------------------------
    def test_color_comparison(self):
        # Blue bottle
        self.frame[20:80, 20:80] = (255, 0, 0)
        # Black bag
        self.frame[120:180, 120:180] = (10, 10, 10)

        t_bottle = make_track("bottle", "left", 1.5, (20, 20, 80, 80))
        t_bag    = make_track("backpack", "right", 1.0, (120, 120, 180, 180))

        intent = parse("are my bottle and bag the same colour", YOLO_NAMES)
        self.assertEqual(intent.kind, "color_compare")

        resp = answer(intent, [t_bottle, t_bag], YOLO_NAMES, frame_bgr=self.frame)
        self.assertTrue(resp.startswith("No."))
        self.assertIn("blue", resp)
        self.assertIn("black", resp)

    # -----------------------------------------------------------------------
    # 6. Scene Dominant Colour
    # -----------------------------------------------------------------------
    def test_scene_dominant_color(self):
        self.frame[:] = (0, 255, 255) # Yellow scene
        intent = parse("what is the dominant colour in this scene", YOLO_NAMES)
        self.assertEqual(intent.kind, "scene_color")

        resp = answer(intent, [], YOLO_NAMES, frame_bgr=self.frame)
        self.assertIn("yellow", resp)


if __name__ == "__main__":
    unittest.main()
