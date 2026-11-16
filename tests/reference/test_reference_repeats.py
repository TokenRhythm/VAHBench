"""The temporal repeat scan: loops are found, look-alikes are not."""
import unittest

import numpy as np

from stage3_evidence.intervals import repeat_evidence
from stage5_scoring import DEFAULT_CUT_POINTS
from stage5_scoring import ledger_from_reports
from stage5_scoring import score_completeness
from tests.reference import synth

synth.use_reference("visual")
from vcheck.repeats import scan_thumbs  # noqa: E402

FPS = synth.FPS


def scan(frames):
    return scan_thumbs(synth.features(frames).thumbs, FPS)


class TestLoops(unittest.TestCase):
    def test_distinct_footage_has_no_repeat(self):
        frames = np.concatenate([synth.scene(240, 1), synth.scene(240, 2)])
        out = scan(frames)
        self.assertEqual(out["abstained"], "")
        self.assertEqual(out["duplicate_union"], [])
        self.assertEqual(out["duplicate_seconds"], 0.0)

    def test_a_looped_clip_charges_only_the_later_showing(self):
        clip = synth.scene(240, 3)
        out = scan(np.concatenate([clip, clip]))
        self.assertAlmostEqual(out["duplicate_seconds"], 10.0, delta=0.3)
        self.assertEqual(out["duplicate_union"][-1][1], 20.0)        # runs to the last frame
        self.assertGreaterEqual(out["duplicate_union"][0][0], 9.0)   # first showing is genuine
        self.assertLessEqual(out["duplicate_union"][-1][1], 20.0 + 1e-6)

    def test_a_mirrored_loop_is_still_a_repeat(self):
        clip = synth.scene(240, 4)
        out = scan(np.concatenate([clip, clip[:, :, ::-1]]))
        self.assertGreaterEqual(out["duplicate_seconds"], 8.0)
        self.assertGreater(out["n_mirrored_runs"], 0)

    def test_six_copies_of_one_clip_leave_one_copy_of_footage(self):
        clip = synth.scene(96, 5)                       # 4 s
        out = scan(np.concatenate([clip] * 6))
        self.assertAlmostEqual(out["duplicate_seconds"], 20.0, delta=1.5)

    def test_a_repeat_inside_other_footage_is_located(self):
        a, b, c = synth.scene(144, 6), synth.scene(120, 7), synth.scene(96, 8)
        out = scan(np.concatenate([a, b, a[:96], c]))    # a's first 4 s return at 11 s
        self.assertEqual(out["n_spans"], 1)
        start, end = out["duplicate_union"][0]
        self.assertAlmostEqual(start, 11.0, delta=0.5)
        self.assertAlmostEqual(end, 15.0, delta=0.5)


class TestLookAlikesAreNotRepeats(unittest.TestCase):
    def test_a_static_shot_is_not_a_repeat(self):
        self.assertEqual(scan(synth.still(480, 9, noise=1.0))["duplicate_seconds"], 0.0)

    def test_a_slow_steady_camera_move_is_not_a_repeat(self):
        # Every frame resembles the one a second earlier and the motion is the
        # same throughout; only the absence of a similarity ridge tells it
        # apart from a loop.
        self.assertEqual(scan(synth.smooth_drift(480, 10, speed=0.1))["duplicate_seconds"], 0.0)

    def test_something_moving_in_front_of_a_locked_camera_is_not_a_repeat(self):
        # The background matches at every offset, so only the comparison with
        # the adjacent sample can tell this from a loop.
        for seed in range(12):
            rng = np.random.default_rng(seed)
            frames = synth.still(720, 100 + seed).copy()
            x = 30.0
            for i in range(720):
                x = float(np.clip(x + rng.normal(0, 1.5), 4, 52))
                frames[i, 24:40, int(x):int(x) + 8] = 230
            self.assertEqual(scan(frames)["duplicate_seconds"], 0.0, seed)

    def test_a_loop_of_such_a_shot_is_still_a_repeat(self):
        rng = np.random.default_rng(1)
        clip = synth.still(240, 101).copy()
        x = 30.0
        for i in range(240):
            x = float(np.clip(x + rng.normal(0, 1.5), 4, 52))
            clip[i, 24:40, int(x):int(x) + 8] = 230
        self.assertGreaterEqual(scan(np.concatenate([clip, clip]))["duplicate_seconds"], 6.0)

    def test_a_very_short_film_cannot_hold_a_repeat(self):
        out = scan(synth.scene(48, 11))
        self.assertEqual((out["duplicate_union"], out["abstained"]), ([], ""))


class TestIntoTheLedger(unittest.TestCase):
    def test_loop_seconds_reach_completeness(self):
        clip = synth.scene(240, 12)
        scores = {}
        for name, frames in (("original", synth.scene(480, 12)),
                             ("loop", np.concatenate([clip, clip]))):
            visual = {"duration": 20.0, "degeneracy": {"repeats": scan(frames),
                                                       "near_duplicate_shots": []}}
            evidence = repeat_evidence(visual, 20.0)
            self.assertEqual(evidence["abstained"], "")
            led = ledger_from_reports("v", 1, 20.0, 20.0, visual=visual)
            scores[name] = (led.duplicate_s, score_completeness([led], 1, DEFAULT_CUT_POINTS).score)
        self.assertEqual(scores["original"][0], 0.0)
        self.assertGreaterEqual(scores["loop"][0], 9.0)
        self.assertLess(scores["loop"][1], scores["original"][1])


if __name__ == "__main__":
    unittest.main()
