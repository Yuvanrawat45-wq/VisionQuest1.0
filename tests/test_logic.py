import unittest
import numpy as np
import cv2
from src.config import Config
from src.contracts import Obs, TrackInfo
from src.distance import TTCEstimator, calibrate_focal, estimate_dist
from src.pipeline import Pipeline
from src import policy


def info(**kw):
    d = dict(id=1, name="chair", box=(250, 100, 350, 300), age=10, moving=False, dist_m=1.5,
             ttc_s=None, direction="ahead", in_corridor=True, truncated=False)
    d.update(kw)
    return TrackInfo(**d)


class TestDistance(unittest.TestCase):
    def test_calibration_roundtrip(self):
        f = calibrate_focal(300, 2.0, 1.7)
        self.assertAlmostEqual(estimate_dist("person", 300, f, {"person": 1.7}, 1.0), 2.0, places=3)

    def test_ttc_closing(self):
        e, ttc = TTCEstimator(), None
        for k in range(31):
            t = k / 30
            ttc = e.update(1, t, 100 + 40 * t)
        self.assertIsNotNone(ttc)
        self.assertAlmostEqual(ttc, (100 + 40) / 40, delta=0.3)   # h/(dh/dt) at t=1

    def test_ttc_not_closing(self):
        e, ttc = TTCEstimator(), "x"
        for k in range(31):
            ttc = e.update(1, k / 30, 120.0)
        self.assertIsNone(ttc)


class TestPolicy(unittest.TestCase):
    cfg = Config()

    def test_static_close_in_corridor_alerts(self):
        a = policy.decide([info()], self.cfg, 100.0, {})
        self.assertIsNotNone(a)
        self.assertIn("chair in front of you", a.text)
        self.assertFalse(a.urgent)

    def test_far_object_silent(self):
        self.assertIsNone(policy.decide([info(dist_m=5.0)], self.cfg, 100.0, {}))

    def test_cooldown(self):
        self.assertIsNone(policy.decide([info()], self.cfg, 101.0, {1: 100.0}))
        self.assertIsNotNone(policy.decide([info()], self.cfg, 105.0, {1: 100.0}))

    def test_urgent_ttc(self):
        a = policy.decide([info(dist_m=3.0, ttc_s=1.0)], self.cfg, 100.0, {})
        self.assertTrue(a.urgent)
        self.assertTrue(a.text.startswith("Stop"))

    def test_truncated_never_trusts_distance(self):
        self.assertIsNone(policy.decide([info(truncated=True)], self.cfg, 100.0, {}))

    def test_side_mover_closing_alerts(self):
        i = info(direction="left", in_corridor=False, moving=True, dist_m=3.5, ttc_s=2.0)
        self.assertIsNotNone(policy.decide([i], self.cfg, 100.0, {}))

    def test_most_urgent_wins(self):
        a = policy.decide([info(id=1, dist_m=1.9), info(id=2, dist_m=0.8)], self.cfg, 100.0, {})
        self.assertEqual(a.tid, 2)


class TestDescribeMode(unittest.TestCase):
    cfg = Config()

    def test_announces_biggest_then_cooldown_then_zone_change(self):
        big = info(id=1, name="person", box=(100, 50, 300, 450), direction="left", dist_m=9.0)
        small = info(id=2, name="bottle", box=(400, 300, 420, 340), direction="right", dist_m=9.0)
        a = policy.decide_describe([big, small], self.cfg, 100.0, {})
        self.assertEqual(a.text, "person on your left")            # far objects still described
        last = {a.key: 100.0}
        self.assertEqual(policy.decide_describe([big, small], self.cfg, 101.0, last).tid, 2)
        moved = info(id=1, name="person", box=(250, 50, 450, 450), direction="ahead")
        self.assertEqual(policy.decide_describe([moved], self.cfg, 101.0, last).text,
                         "person in front of you")                 # zone changed => re-announce

    def test_new_track_is_announced_immediately(self):
        i = info(age=0, name="dog", direction="ahead", moving=None)
        alert = policy.decide_describe([i], self.cfg, 100.0, {})
        self.assertEqual(alert.text, "dog in front of you")

    def test_moving_object_is_announced_with_direction(self):
        i = info(name="person", direction="right", moving=True)
        alert = policy.decide_describe([i], self.cfg, 100.0, {})
        self.assertEqual(alert.text, "person is moving on your right")

    def test_track_is_reannounced_when_classified_as_moving(self):
        stationary = info(id=3, name="person", direction="ahead", moving=None)
        first = policy.decide_describe([stationary], self.cfg, 100.0, {})
        last = {first.key: 100.0}

        moving = info(id=3, name="person", direction="ahead", moving=True)
        second = policy.decide_describe([moving], self.cfg, 101.0, last)
        self.assertEqual(second.text, "person is moving in front of you")


class TestEgoMotion(unittest.TestCase):
    """Camera pans (everything shifts 3 px/frame). A static object rides with the
    background; a mover has +6 px/frame of its own motion. Naive frame-diff fails this."""

    def test_static_vs_moving_under_pan(self):
        rng = np.random.default_rng(0)
        canvas = cv2.GaussianBlur(rng.integers(0, 255, (480, 1000), dtype=np.uint8), (5, 5), 0)
        cfg = Config()
        pipe = Pipeline(cfg)
        infos = []
        for k in range(20):
            frame = cv2.cvtColor(canvas[:, 3 * k: 3 * k + 640], cv2.COLOR_GRAY2BGR)
            obs = [Obs(1, "chair", (400 - 3 * k, 200, 480 - 3 * k, 330), 0.9),            # static
                   Obs(2, "person", (100 - 9 * k + 150, 150, 160 - 9 * k + 150, 400), 0.9)]  # mover
            infos, _ = pipe.step(frame, k / 30, obs)
        by = {i.id: i for i in infos}
        self.assertIs(by[1].moving, False, f"static misclassified: {by[1]}")
        self.assertIs(by[2].moving, True, f"mover misclassified: {by[2]}")


if __name__ == "__main__":
    unittest.main()
