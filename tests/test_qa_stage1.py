"""Unit tests for Stage 1: Expanded Voice Commands & Object-Location Understanding."""

import unittest
from src.contracts import TrackInfo
from src.intent import parse
from src.qa import answer, set_silenced, is_silenced


YOLO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


def make_track(name, direction="ahead", dist_m=1.5, truncated=False):
    return TrackInfo(
        id=1,
        name=name,
        box=(200, 100, 440, 400),
        age=10,
        moving=False,
        dist_m=dist_m,
        ttc_s=None,
        direction=direction,
        in_corridor=True,
        truncated=truncated
    )


class TestStage1VoiceQA(unittest.TestCase):

    def setUp(self):
        set_silenced(False)

    # -----------------------------------------------------------------------
    # 1. Natural Language Variations & Synonyms
    # -----------------------------------------------------------------------
    def test_single_object_location_synonyms(self):
        queries = [
            ("where is my bottle", "bottle"),
            ("can you find my phone", "cell phone"),
            ("where did i leave my bag", "backpack"),
            ("can you see my water bottle", "bottle"),
            ("is my laptop visible", "laptop"),
            ("where is the chair", "chair"),
            ("can you locate my cup", "cup"),
        ]
        for query, expected_target in queries:
            intent = parse(query, YOLO_NAMES)
            self.assertEqual(intent.kind, "where_object", f"Failed for {query}")
            self.assertIn(expected_target, intent.targets, f"Expected {expected_target} in {intent.targets}")

    # -----------------------------------------------------------------------
    # 2. Multi-Object Search
    # -----------------------------------------------------------------------
    def test_multi_object_search(self):
        intent = parse("can you find my bottle and phone", YOLO_NAMES)
        self.assertEqual(intent.kind, "where_object")
        self.assertIn("bottle", intent.targets)
        self.assertIn("cell phone", intent.targets)

        tracks = [
            make_track("bottle", direction="ahead", dist_m=1.2),
            make_track("cell phone", direction="right", dist_m=0.8),
        ]
        resp = answer(intent, tracks, YOLO_NAMES)
        self.assertIn("bottle is in the centre", resp)
        self.assertIn("cell phone is on your right", resp)

    # -----------------------------------------------------------------------
    # 3. Unsupported Objects
    # -----------------------------------------------------------------------
    def test_unsupported_objects_reporting(self):
        unsupported = ["where are my keys", "can you find my shoes", "where is the door"]
        for query in unsupported:
            intent = parse(query, YOLO_NAMES)
            self.assertEqual(intent.kind, "where_object")
            self.assertTrue(len(intent.unsupported_targets) > 0)

            resp = answer(intent, [], YOLO_NAMES)
            self.assertIn("cannot recognize", resp)

    # -----------------------------------------------------------------------
    # 4. Absent Objects (Never Invent Locations)
    # -----------------------------------------------------------------------
    def test_object_not_detected_reports_missing(self):
        intent = parse("where is my laptop", YOLO_NAMES)
        # Empty camera frame (no objects detected)
        resp = answer(intent, [], YOLO_NAMES)
        self.assertIn("cannot see a laptop in the current camera view", resp)

    # -----------------------------------------------------------------------
    # 5. Multiple Instances of Same Object
    # -----------------------------------------------------------------------
    def test_multiple_instances_reporting(self):
        intent = parse("where are the bottles", YOLO_NAMES)
        tracks = [
            make_track("bottle", direction="left", dist_m=1.5),
            make_track("bottle", direction="right", dist_m=2.0),
        ]
        resp = answer(intent, tracks, YOLO_NAMES)
        self.assertIn("2 bottles", resp)
        self.assertIn("on your left", resp)
        self.assertIn("on your right", resp)

    # -----------------------------------------------------------------------
    # 6. Partial Visibility / Truncation Warning
    # -----------------------------------------------------------------------
    def test_partial_visibility_warning(self):
        intent = parse("where is my laptop", YOLO_NAMES)
        tracks = [make_track("laptop", direction="left", dist_m=1.0, truncated=True)]
        resp = answer(intent, tracks, YOLO_NAMES)
        self.assertIn("partially visible", resp)
        self.assertIn("laptop is on your left", resp)

    # -----------------------------------------------------------------------
    # 7. Control Commands: Repeat, Silence, Resume, Help
    # -----------------------------------------------------------------------
    def test_repeat_command(self):
        intent_q = parse("where is my bottle", YOLO_NAMES)
        track = [make_track("bottle", direction="ahead", dist_m=1.0)]
        orig_resp = answer(intent_q, track, YOLO_NAMES)

        intent_rep = parse("repeat that", YOLO_NAMES)
        self.assertEqual(intent_rep.kind, "repeat")
        rep_resp = answer(intent_rep, [], YOLO_NAMES)
        self.assertEqual(orig_resp, rep_resp)

    def test_silence_and_resume_commands(self):
        intent_sil = parse("be quiet", YOLO_NAMES)
        self.assertEqual(intent_sil.kind, "silence")
        resp_sil = answer(intent_sil, [], YOLO_NAMES)
        self.assertIn("silenced", resp_sil)
        self.assertTrue(is_silenced())

        intent_res = parse("start listening again", YOLO_NAMES)
        self.assertEqual(intent_res.kind, "resume")
        resp_res = answer(intent_res, [], YOLO_NAMES)
        self.assertIn("resumed", resp_res)
        self.assertFalse(is_silenced())

    def test_help_command(self):
        intent_h = parse("what can you do", YOLO_NAMES)
        self.assertEqual(intent_h.kind, "help")
        resp_h = answer(intent_h, [], YOLO_NAMES)
        self.assertIn("You can ask where objects are", resp_h)


if __name__ == "__main__":
    unittest.main()
