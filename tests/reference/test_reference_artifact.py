"""Picture defects and splice structure on arrays with known content."""
import unittest

import numpy as np

import stage4_checks.reports as adapters
from shared.types import Verdict
from stage5_scoring import DEFAULT_CUT_POINTS
from stage5_scoring import ledger_from_reports
from stage5_scoring import score_completeness
from tests.reference import synth

synth.use_reference("artifact")
import artifacts  # noqa: E402

FPS = synth.FPS


def analyse(frames):
    return artifacts.analyse_features(synth.features(frames), FPS)


def measured(report):
    return {m.key: m for m in adapters.from_artifact(report)}


class TestCleanFootage(unittest.TestCase):
    def test_moving_footage_has_no_defect_and_no_cut(self):
        rep = analyse(synth.scene(240, 1))
        f = rep["findings"]
        for key in ("black_frames", "solid_frames", "frozen_runs", "tail_freeze", "join_artifacts"):
            self.assertFalse(f[key]["present"], key)
        self.assertEqual(rep["splices"]["n_splices"], 0)
        self.assertEqual(rep["splices"]["segment_frames"], [240])
        self.assertAlmostEqual(rep["meta"]["duration"], 10.0)
        ms = measured(rep)
        for key in ("artifact.black_frames", "artifact.solid_frames", "artifact.frozen_runs"):
            self.assertIs(ms[key].verdict, Verdict.PASS, key)

    def test_report_names_itself_as_a_reimplementation(self):
        stamp = analyse(synth.scene(48, 1))["implementation"]
        self.assertFalse(stamp["original_detector"])
        self.assertEqual(stamp["calibration"], "uncalibrated")
        self.assertIn("flicker", stamp["not_measured"])


class TestHeldAndBlankFrames(unittest.TestCase):
    def test_a_held_frame_is_measured_in_seconds_and_deducted(self):
        clip = synth.scene(240, 2)
        frames = np.concatenate([clip[:120], np.repeat(clip[120:121], 72, axis=0), clip[121:]])
        rep = analyse(frames)
        frozen = rep["findings"]["frozen_runs"]
        self.assertTrue(frozen["present"])
        self.assertEqual(len(frozen["segments"]), 1)
        self.assertAlmostEqual(frozen["segments"][0]["end_s"] - frozen["segments"][0]["start_s"],
                               72 / FPS, delta=2 / FPS)
        self.assertEqual(rep["splices"]["n_splices"], 0)       # a hold is not a cut
        duration = len(frames) / FPS
        led = ledger_from_reports("v", 1, duration, duration, artifact=rep)
        self.assertAlmostEqual(led.frozen_s, 3.0, delta=0.1)
        self.assertAlmostEqual(led.deductions_s, 3.0, delta=0.1)

    def test_exactly_one_second_of_identical_frames_is_a_held_run(self):
        clip = synth.scene(96, 2)
        for n_identical, expected in ((24, True), (20, False)):
            frames = np.concatenate([clip[:48], np.repeat(clip[48:49], n_identical, axis=0), clip[49:]])
            self.assertIs(analyse(frames)["findings"]["frozen_runs"]["present"], expected, n_identical)

    def test_a_small_moving_region_in_a_still_picture_is_not_frozen(self):
        # A locked-off shot where only a mouth moves: the whole-frame mean
        # barely changes, one block changes a great deal.
        frames = synth.still(120, 2).copy()
        rng = np.random.default_rng(0)
        for i in range(120):
            frames[i, 28:32, 28:36] = rng.integers(60, 200)
        f = analyse(frames)["findings"]
        self.assertFalse(f["frozen_runs"]["present"])
        self.assertFalse(f["tail_freeze"]["present"])

    def test_a_fade_out_ending_on_black_is_not_a_held_ending(self):
        clip = synth.scene(120, 5)
        fade = np.stack([(clip[-1] * (1 - k / 12)).astype(np.uint8) for k in range(12)])
        frames = np.concatenate([clip, fade, np.zeros((13, 64, 64), np.uint8)])
        rep = analyse(frames)
        self.assertFalse(rep["findings"]["tail_freeze"]["present"])
        self.assertIs(measured(rep)["artifact.tail_freeze"].verdict, Verdict.PASS)
        self.assertIs(measured(rep)["artifact.black_frames"].verdict, Verdict.PASS)   # edge exempt

    def test_a_short_last_frame_hold_is_not_padding(self):
        clip = synth.scene(120, 5)
        frames = np.concatenate([clip, np.repeat(clip[-1:], 20, axis=0)])
        self.assertFalse(analyse(frames)["findings"]["tail_freeze"]["present"])

    def test_a_short_pause_is_not_a_frozen_run(self):
        clip = synth.scene(240, 2)
        frames = np.concatenate([clip[:120], np.repeat(clip[120:121], 10, axis=0), clip[121:]])
        self.assertFalse(analyse(frames)["findings"]["frozen_runs"]["present"])

    def test_one_reencoded_keyframe_does_not_split_a_hold(self):
        clip = synth.scene(120, 2)
        hold = np.repeat(clip[60:61], 72, axis=0).astype(np.int16)
        hold[36] += 1                                         # a faint refresh
        frames = np.concatenate([clip[:60], hold.astype(np.uint8), clip[61:]])
        segments = analyse(frames)["findings"]["frozen_runs"]["segments"]
        self.assertEqual(len(segments), 1)

    def test_black_and_flat_frames_are_separate_findings(self):
        clip = synth.scene(240, 3)
        black = np.zeros((24, 64, 64), np.uint8)
        grey = np.full((24, 64, 64), 128, np.uint8)
        rep = analyse(np.concatenate([clip[:80], black, clip[80:160], grey, clip[160:]]))
        f = rep["findings"]
        self.assertEqual(f["black_frames"]["n_frames"], 24)
        self.assertEqual(f["solid_frames"]["n_frames"], 24)
        ms = measured(rep)
        self.assertIs(ms["artifact.black_frames"].verdict, Verdict.FAIL)
        self.assertIs(ms["artifact.solid_frames"].verdict, Verdict.FAIL)
        duration = rep["meta"]["duration"]
        led = ledger_from_reports("v", 1, duration, duration, artifact=rep)
        self.assertAlmostEqual(led.black_s, 2.0, delta=0.1)

    def test_dark_footage_with_a_bright_region_is_not_black(self):
        frames = (synth.scene(48, 4).astype(np.float64) * 0.02).astype(np.uint8)
        frames[:, 20:44, 20:44] = 200
        self.assertFalse(analyse(frames)["findings"]["black_frames"]["present"])

    def test_a_held_ending_is_a_tail_freeze(self):
        clip = synth.scene(120, 5)
        frames = np.concatenate([clip, np.repeat(clip[-1:], 48, axis=0)])
        tail = analyse(frames)["findings"]["tail_freeze"]
        self.assertTrue(tail["present"])
        self.assertAlmostEqual(tail["held_seconds"], 2.0, delta=0.05)
        self.assertIs(measured(analyse(frames))["artifact.tail_freeze"].verdict, Verdict.VOID)

    def test_padding_lowers_completeness(self):
        clip = synth.scene(240, 6)
        padded = np.concatenate([clip, np.repeat(clip[-1:], 240, axis=0)])
        scores = []
        for frames in (synth.scene(480, 6), padded):
            rep = analyse(frames)
            led = ledger_from_reports("v", 1, 20.0, len(frames) / FPS, artifact=rep,
                                      visual={"degeneracy": {"repeats": {"duplicate_union": []}}})
            scores.append(score_completeness([led], 1, DEFAULT_CUT_POINTS).score)
        self.assertEqual(scores[0], 1.0)
        self.assertLess(scores[1], 0.5)


class TestSplices(unittest.TestCase):
    def test_hard_cuts_are_located_and_segments_add_up(self):
        frames = np.concatenate([synth.scene(72, 10), synth.scene(120, 11), synth.scene(96, 12)])
        sp = analyse(frames)["splices"]
        self.assertEqual([s["frame"] for s in sp["splices"]], [72, 192])
        self.assertEqual(sp["segment_frames"], [72, 120, 96])
        self.assertEqual(sum(sp["segment_frames"]), len(frames))
        self.assertAlmostEqual(sp["splices"][0]["t_s"], 3.0)

    def test_a_change_in_the_subtitle_band_is_not_a_cut(self):
        frames = synth.still(96, 13, noise=0.0).copy()
        frames[48:, 52:, :] = 255                              # a caption appears
        self.assertEqual(analyse(frames)["splices"]["n_splices"], 0)

    def test_a_sliver_between_two_cuts_is_one_join_artifact(self):
        frames = np.concatenate([synth.scene(72, 20), synth.scene(4, 21), synth.scene(72, 22)])
        rep = analyse(frames)
        ja = rep["findings"]["join_artifacts"]
        self.assertEqual(ja["n_join_artifacts"], 1)
        self.assertIn("sliver", ja["join_artifacts"][0]["issues"])
        m = measured(rep)["artifact.join_artifacts"]
        self.assertIs(m.verdict, Verdict.FAIL)
        self.assertEqual(m.detail["n_splices"], 2)

    def test_one_and_two_frame_slivers_are_join_artifacts_too(self):
        for width in (1, 2, 4):
            frames = np.concatenate([synth.scene(60, 20), synth.scene(width, 21), synth.scene(60, 22)])
            ja = analyse(frames)["findings"]["join_artifacts"]
            self.assertEqual(ja["n_join_artifacts"], 1, width)
            self.assertIn("sliver", ja["join_artifacts"][0]["issues"], width)

    def test_a_dark_frame_at_a_cut_is_a_join_artifact_and_a_clean_cut_is_not(self):
        dark = np.zeros((1, 64, 64), np.uint8)
        clean = analyse(np.concatenate([synth.scene(72, 30), synth.scene(72, 31)]))
        self.assertEqual(clean["findings"]["join_artifacts"]["n_join_artifacts"], 0)
        self.assertIs(measured(clean)["artifact.join_artifacts"].verdict, Verdict.PASS)
        bad = analyse(np.concatenate([synth.scene(72, 30), dark, synth.scene(72, 31)]))
        issues = [i for j in bad["findings"]["join_artifacts"]["join_artifacts"] for i in j["issues"]]
        self.assertIn("dark", issues)
        led = ledger_from_reports("v", 1, 6.0, 145 / FPS, artifact=bad)
        self.assertGreater(led.seam_s, 0)


if __name__ == "__main__":
    unittest.main()
