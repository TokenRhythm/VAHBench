"""Cut continuity on constructed motion traces."""
import unittest

import numpy as np

import stage4_checks.reports as adapters
from shared.types import Verdict
from tests.reference import synth

synth.use_reference("seam", "artifact")
import run_seam_check as seam  # noqa: E402

FPS = synth.FPS


def trace(frames):
    return synth.features(frames).mad_top


def film(*clips):
    """Clips joined with hard cuts; returns frames and the cut times."""
    times, t = [], 0
    for c in clips[:-1]:
        t += len(c)
        times.append(t / FPS)
    return np.concatenate(clips), times


def still_head(n, seed, held):
    """An image-to-video clip: it opens on a still, then starts to move."""
    moving = synth.scene(n - held, seed)
    return np.concatenate([np.repeat(moving[:1], held, axis=0), moving])


def one(report, spans=None):
    ms = adapters.from_seam(report, spans)
    assert len(ms) == 1, ms
    return ms[0]


class TestSeams(unittest.TestCase):
    def test_no_named_cut_means_nothing_to_test(self):
        rep = seam.analyse(trace(synth.scene(240, 1)), FPS, [])
        self.assertEqual(rep["status"], "no_seams")
        self.assertIs(one(rep).verdict, Verdict.NOT_APPLICABLE)

    def test_cuts_into_moving_footage_are_not_broken(self):
        frames, times = film(*[synth.scene(144, s) for s in (2, 3, 4, 5)])
        rep = seam.analyse(trace(frames), FPS, times)
        self.assertEqual(rep["summary"]["n_seams"], 3)
        self.assertEqual(rep["summary"]["n_visible_seams"], 3)
        self.assertEqual(rep["hidden_cuts"], [])
        m = one(rep)
        self.assertIs(m.verdict, Verdict.PASS)
        self.assertEqual((m.detail["k"], m.detail["n"]), (0, 3))

    def test_cuts_landing_on_a_still_head_are_broken(self):
        frames, times = film(synth.scene(144, 6), still_head(144, 7, held=30),
                             synth.scene(144, 8), still_head(144, 9, held=30))
        rep = seam.analyse(trace(frames), FPS, times)
        by_time = {round(s["t_s"], 2): s for s in rep["seams"]}
        self.assertLess(by_time[6.0]["p_head"], 0.05)
        self.assertGreater(by_time[12.0]["p_head"], 0.05)
        self.assertAlmostEqual(by_time[6.0]["dead_s"], 29 / FPS, delta=2 / FPS)
        m = one(rep)
        self.assertIs(m.verdict, Verdict.FAIL)
        self.assertEqual((m.detail["k"], m.detail["n"]), (2, 3))
        self.assertAlmostEqual(m.value, 2 / 3)

    def test_a_still_head_already_charged_in_completeness_is_not_charged_again(self):
        frames, times = film(synth.scene(144, 6), still_head(144, 7, held=30), synth.scene(144, 8))
        rep = seam.analyse(trace(frames), FPS, times)
        self.assertEqual(one(rep).detail["k"], 1)
        covered = one(rep, [(5.9, 7.5)])
        self.assertEqual(covered.detail["k"], 0)
        self.assertEqual(covered.detail["already_charged_in_C"], [6.0])

    def test_an_unnamed_cut_inside_a_clip_is_a_hidden_cut(self):
        frames, times = film(synth.scene(144, 10), synth.scene(144, 11), synth.scene(144, 12))
        named = times[:1]                               # the plan knows one cut of two
        rep = seam.analyse(trace(frames), FPS, named)
        self.assertEqual([h["t_s"] for h in rep["hidden_cuts"]], [12.0])
        m = one(rep)
        self.assertEqual((m.detail["hidden_cuts"], m.detail["k"], m.detail["n"]), (1, 1, 2))

    def test_a_head_that_is_only_slightly_calmer_is_not_a_still_head(self):
        # Very even motion makes any small difference an extreme rank. A head
        # that visibly moves must not be called still on rank alone.
        frames, times = film(synth.scene(160, 6), synth.scene(160, 7), synth.scene(160, 8))
        rep = seam.analyse(trace(frames), FPS, times)
        for row in rep["seams"]:
            self.assertGreater(row["head_motion"], 0.5 * rep["calibration"]["median_window_motion"])
            self.assertEqual(row["p_head"], 1.0)
            self.assertLess(row["p_vis"], 0.05)
        self.assertIs(one(rep).verdict, Verdict.PASS)

    def test_a_named_time_is_snapped_to_the_actual_jump(self):
        frames, times = film(synth.scene(144, 13), synth.scene(144, 14))
        rep = seam.analyse(trace(frames), FPS, [times[0] + 2 / FPS])
        self.assertEqual(rep["seams"][0]["frame"], 144)

    def test_a_cut_with_no_window_beside_it_is_not_judged(self):
        frames, times = film(*[synth.scene(144, s) for s in (2, 3, 4)])
        rep = seam.analyse(trace(frames), FPS, [*times, len(frames) / FPS + 20])
        last = rep["seams"][-1]
        self.assertIsNone(last["p_head"])
        self.assertEqual(one(rep).detail["k"], 0)

    def test_a_small_moving_region_after_a_cut_is_not_a_still_head(self):
        moving = synth.scene(144, 7)
        quiet = np.repeat(moving[:1], 144, axis=0).copy()
        rng = np.random.default_rng(0)
        for i in range(144):
            quiet[i, 8:12, 8:16] = rng.integers(60, 200)
        frames = np.concatenate([synth.scene(144, 6), quiet, synth.scene(144, 8)])
        feat = synth.features(frames)
        rep = seam.analyse(feat.mad_top, FPS, [6.0, 12.0], local=feat.mad_block)
        self.assertEqual(rep["seams"][0]["dead_s"], 0.0)

    def test_too_little_footage_to_compare_against_abstains(self):
        frames, times = film(synth.scene(20, 15), synth.scene(20, 16))
        rep = seam.analyse(trace(frames), FPS, times)
        self.assertEqual(rep["status"], "insufficient_reference")
        self.assertIs(one(rep).verdict, Verdict.ABSTAIN)

    def test_regret_is_reported_as_not_measured(self):
        frames, times = film(synth.scene(144, 17), synth.scene(144, 18))
        rep = seam.analyse(trace(frames), FPS, times)
        self.assertIsNone(rep["regret"])
        self.assertNotIn("shipped_s", one(rep).detail)


if __name__ == "__main__":
    unittest.main()
