"""Regression cases for scoring scope, timing, and missing quality evidence.

Small deterministic evidence fixtures; no detector models or network needed.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


from stage1_inputs.brief import BriefFacts
from stage1_inputs.brief import Character
import stage4_checks.reports as adapters
import stage4_checks.video.brief as expectations
from stage4_checks.video.subtitles import align_cues
from stage4_checks.video.subtitles import text_matches
from stage4_checks.video.subtitles import score_alignment
from shared.types import Measurement
from shared.types import Verdict
from shared.types import Scale
from stage5_scoring.node import DEFAULT_CUT_POINTS
from stage5_scoring.node import check_score
from stage5_scoring.node import score_category
from stage5_scoring.node import score_leaf
from stage5_scoring.registry import SCORED
from stage5_scoring.capabilities import UNSUPPORTED_QUALITY_SOURCES
from stage5_scoring.score import score_case
from stage5_scoring.video.completeness import EpisodeLedger
from stage5_scoring.video.quality.consistency import CONSISTENCY
from stage5_scoring.video.quality.plausibility import PLAUSIBILITY
from run import _category
from run import _preserve_quality_episode_gaps
import tests.inputs.test_brief as fixtures
from tests.inputs.test_brief import FakeBus
from tests.inputs.test_brief import _b1


def measurement(key, verdict=Verdict.PASS):
    return Measurement(key, verdict)


class TestCastFraction(unittest.TestCase):
    def test_unparsed_cast_is_not_inapplicable(self):
        b = BriefFacts(99, declared_character_count=4)
        self.assertIs(expectations.characters_anchored(b, FakeBus(), [Path("ep1.mp4")]).verdict,
                      Verdict.ABSTAIN)

    def test_all_missing_is_zero_for_every_cast_size(self):
        for n in (1, 2, 4, 7):
            with self.subTest(n=n):
                b = BriefFacts(99, characters=[Character(f"Role{i}", f"role{i}") for i in range(n)])
                bus = FakeBus({"ep1.mp4": "x"},
                              sheets={c.first: Path(f"{c.first}.png") for c in b.characters},
                              visual_refs=fixtures.TestExpectations._vis([False] * n))
                m = expectations.characters_anchored(b, bus, bus.delivered_videos())
                self.assertEqual(m.value, 1)
                self.assertEqual(check_score(m, DEFAULT_CUT_POINTS, "fraction"), 0)

    def test_partial_sheets_do_not_shrink_the_cast(self):
        b = _b1()
        bus = FakeBus({"ep1.mp4": "x"}, sheets={"maya": Path("maya.png")},
                      visual_refs=fixtures.TestExpectations._vis([True]))
        m = expectations.characters_anchored(b, bus, bus.delivered_videos())
        self.assertIs(m.verdict, Verdict.ABSTAIN)
        self.assertEqual(len(m.detail["missing_sheets"]), 3)

    def test_summary_counts_cannot_be_assigned_to_names(self):
        bus = FakeBus({"ep1.mp4": "x"}, sheets=fixtures.TestExpectations.SHEETS,
                      visual_refs={"character_identity": {"reference_identity": {
                          "n_references": 4, "n_matched": 4}}})
        self.assertIs(expectations.characters_anchored(_b1(), bus, bus.delivered_videos()).verdict,
                      Verdict.ABSTAIN)

    def test_missing_episode_does_not_prove_character_absence(self):
        class PartialBus(FakeBus):
            def visual(self, v, reference_images=None):
                return fixtures.TestExpectations._vis([False] * 4) if v.name == "ep1.mp4" else None
        bus = PartialBus({"ep1.mp4": "x", "ep2.mp4": "x"}, sheets=fixtures.TestExpectations.SHEETS)
        self.assertIs(expectations.characters_anchored(_b1(), bus, bus.delivered_videos()).verdict,
                      Verdict.ABSTAIN)


class TestCueTiming(unittest.TestCase):
    LINE = "You presented the numbers I kept the receipts"

    def test_merging_a_whole_episode_cannot_hide_internal_timing_errors(self):
        segs = [{"start": i, "end": i + 2, "text": f"line number {i}"} for i in range(0, 60, 5)]
        cues = [{**s, "start": s["start"] + 1, "end": s["end"] + 1} for s in segs]
        t = {"segments": segs}
        merged = [{"start": cues[0]["start"], "end": cues[-1]["end"],
                   "text": " ".join(c["text"] for c in cues)}]
        self.assertLess(score_alignment(merged, t)["score"], score_alignment(cues, t)["score"])
        self.assertLess(score_alignment(merged, t)["score"], .15)

    def test_repeated_dialogue_with_matching_times_is_still_dialogue(self):
        cues = [{"start": i, "end": i + 1, "text": "I know."} for i in (0, 2, 4)]
        bus = FakeBus({"ep1.mp4": "I know. I know. I know."},
                      subtitles={"ep1.mp4": (Path("ep1.srt"), cues)}, asr_segments={"ep1.mp4": cues})
        m = expectations.subtitles_aligned(_b1(), bus, bus.delivered_videos(), [])
        self.assertIs(m.verdict, Verdict.PASS)

    def test_repeated_asr_at_wrong_times_does_not_prove_dialogue(self):
        cues = [{"start": 0, "end": 1, "text": "I know."}]
        segs = [{**cues[0], "start": i, "end": i + 1} for i in (10, 20, 30)]
        bus = FakeBus({"ep1.mp4": "I know. I know. I know."},
                      subtitles={"ep1.mp4": (Path("ep1.srt"), cues)}, asr_segments={"ep1.mp4": segs})
        m = expectations.subtitles_aligned(_b1(), bus, bus.delivered_videos(), [])
        self.assertEqual(m.value, 1)

    def result(self, cue_start=0, cue_end=3, asr_start=0, asr_end=3):
        cues = [{"start": cue_start, "end": cue_end, "text": self.LINE}]
        transcript = {"segments": [{"start": asr_start, "end": asr_end, "text": self.LINE}]}
        return align_cues(cues, transcript)[0]["status"]

    def test_forty_seconds_early_is_mistimed(self):
        self.assertEqual(self.result(asr_start=40, asr_end=43), "mistimed")

    def test_aligned_and_small_asr_jitter(self):
        self.assertEqual(self.result(), "aligned")
        self.assertEqual(self.result(asr_start=0.3, asr_end=3.3), "aligned")

    def test_subtitle_held_for_whole_video_is_mistimed(self):
        self.assertEqual(self.result(cue_end=60, asr_start=20, asr_end=23), "mistimed")

    def test_coarse_segment_is_unknown(self):
        self.assertEqual(self.result(asr_end=60), "unknown")

    def test_word_timestamps_locate_a_phrase_inside_coarse_segment(self):
        cues = [{"start": 4, "end": 6, "text": "hello friend"}]
        transcript = {"segments": [{"start": 0, "end": 60, "text": "hello friend",
                                   "words": [{"start": 4, "end": 5, "word": "hello"},
                                             {"start": 5, "end": 6, "word": "friend"}]}]}
        self.assertEqual(align_cues(cues, transcript)[0]["status"], "aligned")

    def test_repeated_line_can_match_its_later_occurrence(self):
        cue = {"start": 40, "end": 43, "text": self.LINE}
        t = {"segments": [{**cue, "start": 0, "end": 3}, cue]}
        self.assertEqual(align_cues([cue], t)[0]["status"], "aligned")

    def test_point_word_timestamps_and_partial_overlap_are_supported(self):
        t = {"segments": [{"text": "Then prove it", "start": 34.46, "end": 35.68,
                           "words": [{"word": "Then", "start": 34.46, "end": 35.1},
                                     {"word": "prove", "start": 35.1, "end": 35.1},
                                     {"word": "it", "start": 35.1, "end": 35.68}]}]}
        cue = {"start": 33, "end": 35, "text": "Then prove it"}
        self.assertEqual(align_cues([cue], t)[0]["status"], "mistimed")

    def test_order_and_word_multiplicity_matter(self):
        self.assertFalse(text_matches("one two three", [{"text": "three two one"}]))
        self.assertFalse(text_matches("no no no no no", [{"text": "no yes maybe please thanks"}]))
        self.assertTrue(text_matches("我保留了凭据", [{"text": "我保留了凭据。"}]))

    def test_missing_invalid_times_are_unknown(self):
        for seg in ({"text": self.LINE}, {"text": self.LINE, "start": 0, "end": float("nan")}):
            rows = align_cues([{"start": 0, "end": 3, "text": self.LINE}], {"segments": [seg]})
            self.assertEqual(rows[0]["status"], "unknown")

    def test_integrated_mistimed_cue_scores_zero(self):
        cue = {"start": 0, "end": 3, "text": self.LINE}
        bus = FakeBus({"ep1.mp4": self.LINE}, subtitles={"ep1.mp4": (Path("ep1.srt"), [cue])},
                      asr_segments={"ep1.mp4": [{**cue, "start": 40, "end": 43}]})
        m = expectations.subtitles_aligned(_b1(), bus, bus.delivered_videos(), [])
        self.assertEqual(m.detail["n_mistimed"], 1)
        self.assertEqual(check_score(m, DEFAULT_CUT_POINTS), 0)

    def test_an_episode_without_timed_subtitles_is_not_skipped(self):
        cue = {"start": 45, "end": 48, "text": self.LINE}
        bus = FakeBus({"ep1.mp4": self.LINE, "ep2.mp4": self.LINE},
                      subtitles={"ep1.mp4": (Path("ep1.srt"), [cue])})
        self.assertIs(expectations.subtitles_aligned(_b1(), bus, bus.delivered_videos(), []).verdict,
                      Verdict.VOID)


class TestBidirectionalAlignment(unittest.TestCase):
    @staticmethod
    def transcript(*cues):
        return {"text": " ".join(c["text"] for c in cues),
                "n_words": sum(len(c["text"].split()) for c in cues), "segments": list(cues)}

    def test_one_correct_cue_does_not_hide_uncaptioned_speech(self):
        a = {"start": 0, "end": 2, "text": "hello my friend"}
        b = {"start": 5, "end": 7, "text": "bring the receipts"}
        t = self.transcript(a, b)
        self.assertEqual(score_alignment([a, b], t)["score"], 1)
        only = score_alignment([a], t)
        self.assertEqual(only["precision"], 1)
        self.assertEqual(only["recall"], 0.5)
        self.assertAlmostEqual(only["score"], 2 / 3)

    def test_continuous_alignment_reaches_the_quality_score(self):
        key = "brief.subtitles_aligned"
        for aligned in (0.2, 0.45, 0.7, 0.95):
            m = Measurement(key, Verdict.FAIL, Scale.RATE, 1 - aligned)
            leaf = next(lf for lf in PLAUSIBILITY.leaves if lf.key == "av_alignment")
            self.assertEqual(score_leaf(leaf, {key: m}, DEFAULT_CUT_POINTS).score, aligned)

    def test_duplicate_cue_cannot_reuse_one_spoken_line(self):
        a = {"start": 0, "end": 2, "text": "hello my friend"}
        result = score_alignment([a, a], self.transcript(a))
        self.assertEqual(result["credited_words"], 3)
        self.assertAlmostEqual(result["score"], 2 / 3)

    def test_time_penalty_is_continuous(self):
        a = {"start": 5, "end": 8, "text": "hello my friend"}
        t = self.transcript(a)
        scores = [score_alignment([{**a, "start": 5 + d, "end": 8 + d}], t)["score"]
                  for d in (0, 0.5, 0.6, 0.8, 1, 2, 4)]
        self.assertEqual(scores[0:2], [1, 1])
        self.assertTrue(all(a > b for a, b in zip(scores[1:], scores[2:])), scores)
        self.assertEqual(scores[-1], 0)

    def test_short_lines_and_ordinary_repetitions_survive_filtering(self):
        cues = [{"start": i * 3, "end": i * 3 + 1, "text": t}
                for i, t in enumerate(("Careful.", "No.", "No.", "No."))]
        speech, _ = expectations._dialogue_cues(cues)
        self.assertEqual(speech, cues)
        self.assertEqual(score_alignment(speech, self.transcript(*cues))["score"], 1)

    def test_one_word_episode_with_timed_evidence_is_not_silence(self):
        cue = {"start": 5, "end": 6, "text": "Careful."}
        bus = FakeBus({"ep1.mp4": "Careful."}, subtitles={"ep1.mp4": (Path("ep1.srt"), [cue])},
                      asr_segments={"ep1.mp4": [cue]})
        m = expectations.subtitles_aligned(_b1(), bus, bus.delivered_videos(), [])
        self.assertIs(m.verdict, Verdict.PASS)

    def test_screen_credit_is_not_dialogue_and_speaker_tags_are_stripped(self):
        self.assertFalse(expectations._spoken("CRISIS FORECAST — V. HALE"))
        cue = {"start": 1, "end": 3, "text": "[MAYA] Keep the receipts."}
        clean = {**cue, "text": "Keep the receipts."}
        speech, _ = expectations._dialogue_cues([cue])
        self.assertEqual(speech, [clean])
        self.assertEqual(score_alignment(speech, self.transcript(clean))["score"], 1)

    def test_coarse_substring_does_not_invent_word_times(self):
        cue = {"start": 10, "end": 12, "text": "hello friend"}
        segment = {"start": 0, "end": 60, "text": "please say hello friend and then go away"}
        self.assertIsNone(score_alignment([cue], self.transcript(segment))["score"])

    def test_one_unknown_cue_does_not_discard_other_measured_words(self):
        a = {"start": 0, "end": 2, "text": "hello my friend"}
        b = {"start": 5, "end": 7, "text": "bring the receipts"}
        bad = {**b, "end": float("nan")}
        result = score_alignment([a, bad], self.transcript(a, b))
        self.assertEqual(result["score"], 0.5)

    def test_chinese_single_word_and_punctuation(self):
        a = {"start": 0, "end": 1, "text": "小心！"}
        self.assertTrue(expectations._spoken(a["text"]))
        self.assertEqual(score_alignment([a], self.transcript(a))["score"], 1)

    def test_typographic_punctuation_does_not_create_missing_words(self):
        cue = {"start": 0, "end": 2, "text": "It’s ＭＹ office."}
        speech = {**cue, "text": "it's my office"}
        self.assertEqual(score_alignment([cue], self.transcript(speech))["score"], 1)


class TestPhysicalPopulation(unittest.TestCase):
    def test_legacy_flat_boundary_flags_require_remeasurement(self):
        r = {"penetration": {"coverage": {"pairs_evaluated": 20, "pairs_judged": 20,
             "frame_contact_rate": 1}, "unresolvable": {"n": 0},
             "flagged": {"n": 10, "by_reason": {"no_depth_step": 10}}}}
        self.assertIs(adapters.from_physical(r)[0].verdict, Verdict.ABSTAIN)

    @staticmethod
    def measure(evaluated=265, judged=225, unresolved=89, flagged=40):
        report = {"penetration": {"coverage": {"pairs_evaluated": evaluated,
                    "pairs_judged": judged, "frame_contact_rate": 0.9},
                    "unresolvable": {"n": unresolved}, "flagged": {"n": flagged}}}
        return {m.key: m for m in adapters.from_physical(report)}["physical.penetration"]

    def test_real_case_uses_only_resolved_nonattachment_pairs(self):
        m = self.measure()
        self.assertAlmostEqual(m.value, 40 / 136)
        self.assertEqual(m.detail["pairs_resolved"], 136)

    def test_adding_attachments_does_not_improve_score(self):
        self.assertEqual(self.measure(evaluated=1000).value, self.measure().value)

    def test_adding_unresolved_pairs_does_not_improve_score(self):
        self.assertEqual(self.measure(evaluated=365, judged=325, unresolved=189).value,
                         self.measure().value)

    def test_absent_or_inconsistent_counts_are_not_clean(self):
        for kwargs in ({"flagged": None}, {"unresolved": None}, {"flagged": 140}, {"judged": 300}):
            self.assertIs(self.measure(**kwargs).verdict, Verdict.ABSTAIN)


class TestProxyScope(unittest.TestCase):
    def test_legacy_within_cluster_faces_cannot_certify_cross_shot_identity(self):
        r = {"character_identity": {"per_identity_consistency": {"worst": .99, "intra_shot": .9}}}
        self.assertIs(adapters.from_visual(r)[0].verdict, Verdict.ABSTAIN)

    def test_cluster_count_cannot_establish_role_voice_consistency(self):
        for n in (1, 2, 4, 6, 9):
            report = {"cleanliness": {"n_speaker_clusters": n, "n_speech_windows": 12, "rms": 0.1}}
            m = {m.key: m for m in adapters.from_av(report, expected_speakers=4)}["av.speaker_consistency"]
            self.assertIs(m.verdict, Verdict.ABSTAIN)
            self.assertEqual(m.detail["n_clusters"], n)

    def test_class_drift_cannot_establish_instance_drift(self):
        for ratio in (0.1, 1, 10):
            report = {"scale_drift": {"drift_over_noise": {"value": ratio}}}
            m = {m.key: m for m in adapters.from_physical(report)}["visual.scale_consistency"]
            self.assertIs(m.verdict, Verdict.ABSTAIN)
            self.assertEqual(m.detail["drift_over_noise"], ratio)

    def test_compatible_visible_face_is_not_a_speaking_face(self):
        report = {"presence": {"windows": [{"verdicts": [{"sound": "Speech", "verdict": "plausible"}]}]}}
        m = {m.key: m for m in adapters.from_av(report)}["av.voice_face_match"]
        self.assertIs(m.verdict, Verdict.ABSTAIN)
        self.assertEqual(m.detail["plausible"], 1)


class TestEvidenceGate(unittest.TestCase):
    def test_detector_limitations_are_reported_at_rubric_level(self):
        from stage5_scoring.registry import completeness_of_rubric
        limits = completeness_of_rubric()["evidence_limitations"]
        self.assertEqual(set(limits), set(UNSUPPORTED_QUALITY_SOURCES))
        self.assertFalse(set(limits) & set(SCORED))

    def test_one_measured_leaf_no_longer_makes_a_perfect_dimension(self):
        key = "av.speaker_consistency"
        r = score_category(CONSISTENCY, {key: measurement(key)}, DEFAULT_CUT_POINTS)
        self.assertIsNone(r.score)
        self.assertEqual((r.observed_score, r.coverage, r.source_coverage), (None, 0, 0))
        self.assertEqual(set(r.missing_sources), {"visual.identity_consistency", "seam.cut_continuity"})

    def test_missing_supported_source_lowers_coverage_not_score(self):
        ms = {k: measurement(k) for k in PLAUSIBILITY.source_keys}
        del ms["physical.penetration"]
        r = score_category(PLAUSIBILITY, ms, DEFAULT_CUT_POINTS)
        self.assertAlmostEqual(r.coverage, 2 / 3)
        self.assertLess(r.source_coverage, 1)
        self.assertEqual(r.score, 1.0)
        self.assertIn("physical.penetration", r.missing_sources)

    def test_explicit_inapplicability_is_not_a_gap(self):
        ms = {k: measurement(k) for k in CONSISTENCY.source_keys}
        ms["seam.cut_continuity"] = measurement("seam.cut_continuity", Verdict.NOT_APPLICABLE)
        r = score_category(CONSISTENCY, ms, DEFAULT_CUT_POINTS)
        self.assertEqual((r.score, r.coverage, r.source_coverage), (1, 1, 1))
        self.assertEqual(r.n_applicable, 1)

    def test_abstention_is_not_inapplicability(self):
        ms = {k: measurement(k) for k in CONSISTENCY.source_keys}
        ms["seam.cut_continuity"] = measurement("seam.cut_continuity", Verdict.ABSTAIN)
        r = score_category(CONSISTENCY, ms, DEFAULT_CUT_POINTS)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(r.n_applicable, 1)

    def test_disabled_proxies_cannot_change_final_score(self):
        ms = {k: measurement(k) for k in SCORED}
        result = score_case(ms, [EpisodeLedger("ep1", target_s=60, delivered_s=60)], n_required=1, delivered=True)
        self.assertGreater(result.coverage, 0.6)
        self.assertEqual(result.S, 100)
        self.assertTrue(result.sufficient)
        for verdict in Verdict:
            changed = {**ms, **{k: measurement(k, verdict) for k in UNSUPPORTED_QUALITY_SOURCES}}
            self.assertEqual(score_case(changed, [EpisodeLedger("ep1", target_s=60, delivered_s=60)],
                                        n_required=1, delivered=True).S, result.S)

    def test_missing_measurement_marks_total_as_conditional(self):
        ms = {k: measurement(k) for k in SCORED}
        ms["seam.cut_continuity"] = measurement("seam.cut_continuity", Verdict.FAIL)
        before = score_case(ms, [EpisodeLedger("ep1", target_s=60, delivered_s=60)], n_required=1, delivered=True)
        del ms["seam.cut_continuity"]
        after = score_case(ms, [EpisodeLedger("ep1", target_s=60, delivered_s=60)], n_required=1, delivered=True)
        self.assertIsNotNone(after.S)
        self.assertTrue(after.score_is_partial)
        self.assertFalse(before.score_is_partial)

    def test_missing_episode_does_not_disappear_in_worst_merge(self):
        key = "visual.identity_consistency"
        acc = {key: measurement(key)}
        episodes = [{"video": "ep1", "measurements": acc.copy()}, {"video": "ep2", "measurements": {}}]
        _preserve_quality_episode_gaps(acc, episodes)
        self.assertIs(acc[key].verdict, Verdict.PASS)
        self.assertEqual(acc[key].detail["missing_episodes"], ["ep2"])


class TestDetectorCache(unittest.TestCase):
    def test_changed_detectors_never_reuse_pre_fix_output(self):
        import json
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        import stage4_checks.dispatch as runners
        with tempfile.TemporaryDirectory() as tmp:
            for module in ("visual", "physical", "av", "seam"):
                with self.subTest(module=module):
                    (Path(tmp) / f"{module}.json").write_text('{"stale": true}', encoding="utf8")
                    def compute(cmd, **kw):
                        Path(cmd[cmd.index("--out") + 1]).write_text('{"fresh": true}', encoding="utf8")
                        return SimpleNamespace(returncode=0, stderr="", stdout="")
                    script = Path(tmp) / module / "run.py"
                    script.parent.mkdir()
                    script.write_text("# isolated cache test entrypoint\n")
                    with patch.dict(runners.SCRIPTS, {module: script}), \
                            patch.object(runners, "_interpreter", return_value=sys.executable), \
                            patch.object(runners.subprocess, "run", side_effect=compute) as run:
                        r = runners.run_module(module, Path("video.mp4"), Path(tmp))
                        self.assertEqual(r.report, {"fresh": True})
                        self.assertTrue(json.loads((Path(tmp) / f"{module}.json").read_text())["stale"])
                        self.assertEqual(runners.run_module(module, Path("video.mp4"), Path(tmp)).report, r.report)
                        self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
