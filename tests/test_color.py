"""Unit tests for OpenCV HSV Colour Recognition Engine."""

import unittest
import numpy as np
import cv2

from src.color import (
    analyze_patch_color,
    extract_object_color,
    get_scene_dominant_color,
    find_objects_with_color,
    ColorResult
)
from src.contracts import TrackInfo


def make_solid_patch(bgr_tuple, size=(100, 100)):
    patch = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    patch[:] = bgr_tuple
    return patch


class TestColorEngine(unittest.TestCase):

    # -----------------------------------------------------------------------
    # 1. Base Colors in BGR
    # -----------------------------------------------------------------------
    def test_pure_colors(self):
        color_samples = [
            ("blue",   (255, 0, 0)),     # BGR pure blue
            ("green",  (0, 255, 0)),     # BGR pure green
            ("red",    (0, 0, 255)),     # BGR pure red
            ("yellow", (0, 255, 255)),   # BGR pure yellow (R+G)
            ("black",  (10, 10, 10)),    # BGR very dark
            ("white",  (245, 245, 245)),# BGR bright white
            ("grey",   (128, 128, 128)),# BGR neutral grey
        ]

        for expected_name, bgr in color_samples:
            patch = make_solid_patch(bgr)
            res = analyze_patch_color(patch)
            self.assertTrue(res.is_reliable, f"Color {expected_name} should be reliable")
            self.assertEqual(res.name, expected_name, f"Expected {expected_name}, got {res.name}")

    # -----------------------------------------------------------------------
    # 2. Shades (Dark / Bright / Light)
    # -----------------------------------------------------------------------
    def test_shades(self):
        # Dark Blue
        dark_blue = make_solid_patch((120, 0, 0))
        res_db = analyze_patch_color(dark_blue)
        self.assertEqual(res_db.name, "blue")
        self.assertEqual(res_db.shade, "dark")
        self.assertEqual(res_db.display_name, "dark blue")

        # Light Blue
        light_blue = make_solid_patch((245, 200, 180))
        res_lb = analyze_patch_color(light_blue)
        self.assertEqual(res_lb.name, "blue")

    # -----------------------------------------------------------------------
    # 3. Inner-Region Sampling from Bounding Box
    # -----------------------------------------------------------------------
    def test_bounding_box_extraction(self):
        # Create a 200x200 canvas with green background
        canvas = np.zeros((200, 200, 3), dtype=np.uint8)
        canvas[:] = (0, 255, 0)  # Green background

        # Place a red object in the center (box from x:50..150, y:50..150)
        canvas[50:150, 50:150] = (0, 0, 255)  # Red center

        box = (50.0, 50.0, 150.0, 150.0)
        res = extract_object_color(canvas, box, is_person=False)
        self.assertEqual(res.name, "red")

    # -----------------------------------------------------------------------
    # 4. Poor Lighting & Guardrails
    # -----------------------------------------------------------------------
    def test_poor_lighting_handling(self):
        # Extremely dark patch
        pitch_black = make_solid_patch((5, 5, 5))
        res_dark = analyze_patch_color(pitch_black)
        self.assertEqual(res_dark.name, "black")

    # -----------------------------------------------------------------------
    # 5. Scene Dominant Color
    # -----------------------------------------------------------------------
    def test_scene_dominant_color(self):
        scene = make_solid_patch((200, 200, 200), size=(480, 640))
        res = get_scene_dominant_color(scene)
        self.assertIn(res.name, ["grey", "white"])

    # -----------------------------------------------------------------------
    # 6. Find Objects by Color
    # -----------------------------------------------------------------------
    def test_find_objects_by_color(self):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        # Blue object at box1
        frame[50:150, 20:100] = (255, 0, 0)
        # Red object at box2
        frame[50:150, 180:260] = (0, 0, 255)

        track_blue = TrackInfo(1, "bottle", (20, 50, 100, 150), 10, False, 1.2, None, "left", False, False)
        track_red  = TrackInfo(2, "cup", (180, 50, 260, 150), 10, False, 1.4, None, "right", False, False)

        matches_blue = find_objects_with_color(frame, [track_blue, track_red], "blue")
        self.assertEqual(len(matches_blue), 1)
        self.assertEqual(matches_blue[0][0].name, "bottle")

        matches_red = find_objects_with_color(frame, [track_blue, track_red], "red")
        self.assertEqual(len(matches_red), 1)
        self.assertEqual(matches_red[0][0].name, "cup")


if __name__ == "__main__":
    unittest.main()
