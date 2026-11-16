"""The checks that closed the deterministic rubric (2026-09-16) -- O8
voice_face_match and long_silence, O4 identity_consistency -- and the
deferred / planned split that replaces an open-ended "not yet built" list.
(O7 dialogue_repetition, closed here the same day, was deleted on 2026-09-22
with the diagnostic bucket: the loop it caught is charged in C.)
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))

import stage4_checks.video.brief as expectations
import stage4_checks.reports as extract                              # noqa: E402
from stage1_inputs.brief import parse_prompts                                # noqa: E402
from stage3_evidence.collector import parse_silencedetect                       # noqa: E402
from stage5_scoring.registry import ALL_CHECKS
from stage5_scoring.registry import DEFERRED
from stage5_scoring.registry import PLANNED
from shared.types import Scale
from shared.types import Verdict
from stage5_scoring.registry import completeness_of_rubric as rubric_completeness
from stage5_scoring.capabilities import UNSUPPORTED_QUALITY_SOURCES
from tests.inputs.test_brief import P1
from tests.inputs.test_brief import P7
from tests.inputs.test_brief import FakeBus                                 # noqa: E402


def by_key(ms):
    return {m.key: m for m in ms}


class TestRubricSplit(unittest.TestCase):
    def test_nothing_is_merely_planned(self):
        self.assertEqual(PLANNED, frozenset())
        # Four measurement-side deferrals, and nothing left on the trajectory
        # side: `autonomy` was deleted (no harness here is interactive, so it
        # could only ever be N/A) and `recovery_rate` was merged into 2.1.a,
        # which now scores unrecovered attempts. See reliability.py.
        self.assertEqual(DEFERRED, {"vlm.artifact_quality", "vlm.narrative_coherence",
                                    "vlm.narration_visual_alignment", "av.lip_sync"} | set(UNSUPPORTED_QUALITY_SOURCES))
        rc = rubric_completeness()
        self.assertEqual(rc["n_deferred"], 4 + len(UNSUPPORTED_QUALITY_SOURCES))
        self.assertEqual(rc["deferred"]["av.lip_sync"], "module")
        self.assertEqual(rc["deferred"]["vlm.narrative_coherence"], "compute")
        # Unsupported quality leaves are excluded uniformly from run coverage.
        self.assertEqual(rc["n_unimplemented_leaves"], 3)
        self.assertEqual(rc["n_scored"] + rc["n_completeness_inputs"]
                         + rc["n_diagnostic"] + rc["n_deferred"], rc["n_checks"])

    def test_deferred_checks_stay_out_of_the_denominator(self):
        for k in DEFERRED:
            self.assertFalse(ALL_CHECKS[k].implemented)


class TestVoiceFaceMatch(unittest.TestCase):
    LIVE = {"rms": 0.03, "peak": 0.6, "signal_db": -20.0, "clipped_runs": 0,
            "dynamic_range_db": 14.0, "pause_fraction": 0.3,
            "n_speaker_clusters": 3, "n_speech_windows": 4}

    @staticmethod
    def _win(verdict, people=2):
        return {"start": 0, "end": 10, "people": people, "sounds": [["Speech", 0.9]],
                "verdicts": [{"sound": "Speech", "verdict": verdict}]}

    def test_voice_type_counts_are_diagnostic_only(self):
        rep = {"cleanliness": self.LIVE, "presence": {"windows": [
            self._win("plausible"), self._win("plausible"), self._win("implausible"),
            self._win("offscreen_voice", people=0), self._win("unknown")]}}
        m = by_key(extract.from_av(rep))["av.voice_face_match"]
        self.assertIs(m.verdict, Verdict.ABSTAIN)
        self.assertIsNone(m.value)
        self.assertEqual(m.detail["implausible"], 1)
        self.assertEqual((m.detail["plausible"], m.detail["offscreen_voice"]), (2, 1))

    def test_unresolved_windows_abstain(self):
        rep = {"cleanliness": self.LIVE, "presence": {"windows": [
            self._win("offscreen_voice", people=0), self._win("unknown")]}}
        self.assertIs(by_key(extract.from_av(rep))["av.voice_face_match"].verdict,
                      Verdict.ABSTAIN)
        self.assertIs(by_key(extract.from_av({"cleanliness": self.LIVE}))["av.voice_face_match"].verdict,
                      Verdict.ABSTAIN)

    def test_silent_track_suppresses_the_new_audio_checks_too(self):
        silent = {"cleanliness": {**self.LIVE, "rms": 0.0, "peak": 0.0, "signal_db": -200.0}}
        d = by_key(extract.from_av(silent))
        for k in ("av.voice_face_match", "av.long_silence"):
            self.assertIs(d[k].verdict, Verdict.NOT_APPLICABLE, k)


class TestLongSilence(unittest.TestCase):
    LOG = ("[silencedetect @ 000001] silence_start: 10.5\n"
           "[silencedetect @ 000001] silence_end: 17.5 | silence_duration: 7\n"
           "[silencedetect @ 000001] silence_start: 30\n"
           "[silencedetect @ 000001] silence_end: 33.25 | silence_duration: 3.25\n"
           "[silencedetect @ 000001] silence_start: 58\n")          # open at the end: dropped

    def test_parse_ffmpeg_log(self):
        runs = parse_silencedetect(self.LOG)
        self.assertEqual([(r["start"], r["duration"]) for r in runs], [(10.5, 7.0), (30.0, 3.25)])

    def test_holes_of_five_seconds_are_the_rate(self):
        runs = parse_silencedetect(self.LOG)
        m = extract.from_silence(runs, 60.0)
        self.assertIs(m.verdict, Verdict.FAIL)
        self.assertAlmostEqual(m.value, 7 / 60)
        self.assertEqual(m.detail["n_long"], 1)
        self.assertIs(extract.from_silence(runs[1:], 60.0).verdict, Verdict.PASS)
        self.assertIs(extract.from_silence(None, 60.0).verdict, Verdict.ABSTAIN)


class SegBus(FakeBus):
    """Transcripts with real segment structure: text split on ' | '."""

    def transcript(self, v):
        t = self._t.get(str(v))
        if t is None:
            return None
        parts = [p for p in t.split(" | ") if p]
        return {"silent": self._silent, "n_words": len(t.replace(" | ", " ").split()),
                "text": " ".join(parts), "model": "fake",
                "segments": [{"start": 5.0 * i, "end": 5.0 * i + 4, "text": p}
                             for i, p in enumerate(parts)]}


class TestIdentityConsistency(unittest.TestCase):
    def test_ratio_over_the_modules_within_shot_floor(self):
        rep = {"n_shots": 6, "character_identity": {"per_identity_consistency": {
            "comparison_scope": "cross_shot", "mean": 0.62, "worst": 0.55, "intra_shot": 0.90}}}
        m = by_key(extract.from_visual(rep))["visual.identity_consistency"]
        self.assertIs(m.verdict, Verdict.FAIL)
        self.assertEqual(m.scale, Scale.RATIO)
        self.assertAlmostEqual(m.value, 0.45 / 0.10, places=6)

    def test_without_the_floor_it_abstains(self):
        rep = {"n_shots": 6, "character_identity": {"per_identity_consistency": {
            "mean": 0.62, "worst": 0.55}}}
        m = by_key(extract.from_visual(rep))["visual.identity_consistency"]
        self.assertIs(m.verdict, Verdict.ABSTAIN)
        self.assertIn("噪声底", m.reason)


if __name__ == "__main__":
    unittest.main()
