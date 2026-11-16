"""The scoring tree: pure dict-in / number-out. No videos, no models, no I/O.

Covers, from leaf to root: the measurement -> [0, 1] mapping, leaf and
category reduction, the completeness ledger (the worked cases from
SCORING_SYSTEM.md: honest short, padded, looped, over-delivered, missing
episode), Q's geometric mean, P's mean, and the composition S with its three
non-numeric outcomes (no delivery, Q undefined, P unmeasured).
"""
from __future__ import annotations

import os
import sys
import unittest


from stage5_scoring.registry import ALL_CHECKS
from stage5_scoring.registry import SCORED
from shared.types import Measurement
from shared.types import Scale
from shared.types import Verdict   # noqa: E402
import stage5_scoring as rubric                                                    # noqa: E402
from stage5_scoring import DEFAULT_CUT_POINTS
from stage5_scoring import Leaf
from stage5_scoring import EpisodeLedger
from stage5_scoring import check_score
from stage5_scoring import score_leaf
from stage5_scoring import score_completeness
from stage5_scoring import score_quality
from stage5_scoring import score_process
from stage5_scoring import score_case
from stage5_scoring import ledger_from_reports
from stage5_scoring.registry import COMPLETENESS_INPUTS
from stage5_scoring.registry import DIAGNOSTIC          # noqa: E402

CUTS = DEFAULT_CUT_POINTS


def m(key, verdict, scale=Scale.BOOLEAN, value=None, reason="x", **detail):
    return Measurement(key=key, verdict=verdict, scale=scale, value=value,
                       reason=reason, detail=dict(detail))


def all_scored(verdict=Verdict.PASS):
    return {k: m(k, verdict) for k in SCORED}


def led(video="ep1.mp4", T=60.0, d=60.0, **ded):
    return EpisodeLedger(video=video, target_s=T, delivered_s=d, **ded)


# --------------------------------------------------------------- leaves --
class TestCheckScore(unittest.TestCase):
    def test_pass_void_abstain(self):
        self.assertEqual(check_score(m("k", Verdict.PASS), CUTS), 1.0)
        self.assertEqual(check_score(m("k", Verdict.VOID), CUTS), 0.0)
        self.assertIsNone(check_score(m("k", Verdict.ABSTAIN), CUTS))
        self.assertIsNone(check_score(m("k", Verdict.NOT_APPLICABLE), CUTS))
        self.assertIsNone(check_score(None, CUTS))

    def test_boolean_fail_is_zero(self):
        self.assertEqual(check_score(m("k", Verdict.FAIL), CUTS), 0.0)

    def test_rate_fail_is_linear_between_the_cuts(self):
        clean, suspect = CUTS["defect_rate"]["clean"], CUTS["defect_rate"]["suspect"]
        mid = (clean + suspect) / 2
        self.assertEqual(check_score(m("k", Verdict.FAIL, Scale.RATE, clean), CUTS), 1.0)
        self.assertAlmostEqual(check_score(m("k", Verdict.FAIL, Scale.RATE, mid), CUTS), 0.5)
        self.assertEqual(check_score(m("k", Verdict.FAIL, Scale.RATE, suspect), CUTS), 0.0)
        self.assertEqual(check_score(m("k", Verdict.FAIL, Scale.RATE, 0.9), CUTS), 0.0)

    def test_hard_rate_uses_its_own_band(self):
        v = 0.004                    # between hard clean 0.002 and suspect 0.01
        s = check_score(m("k", Verdict.FAIL, Scale.HARD_RATE, v), CUTS)
        self.assertGreater(s, 0.0)
        self.assertLess(s, 1.0)
        # the same number on the statistical band would be clean
        self.assertEqual(check_score(m("k", Verdict.FAIL, Scale.RATE, v), CUTS), 1.0)

    def test_fraction_mapping_is_one_minus_value(self):
        self.assertAlmostEqual(
            check_score(m("k", Verdict.FAIL, Scale.RATE, 0.25), CUTS, "fraction"), 0.75)
        self.assertEqual(check_score(m("k", Verdict.PASS, Scale.RATE, 0.0), CUTS, "fraction"), 1.0)

    def test_jeffreys_shrinks_sparse_records(self):
        two_calls = m("k", Verdict.PASS, Scale.RATE, 0.0, k=0, n=2)
        many = m("k", Verdict.PASS, Scale.RATE, 0.0, k=0, n=100)
        self.assertAlmostEqual(check_score(two_calls, CUTS, "jeffreys"), 1 - 0.5 / 3)
        self.assertAlmostEqual(check_score(many, CUTS, "jeffreys"), 1 - 0.5 / 101)
        # No counts on the measurement: a pass is still a pass, a fail uses the band.
        self.assertEqual(check_score(m("k", Verdict.PASS), CUTS, "jeffreys"), 1.0)
        self.assertEqual(check_score(m("k", Verdict.FAIL, Scale.RATE, 0.5), CUTS, "jeffreys"), 0.0)


class TestLeaf(unittest.TestCase):
    L = Leaf("x", "x", "x", ("a", "b", "c"))

    def test_mean_over_measured_sources_only(self):
        ms = {"a": m("a", Verdict.PASS), "b": m("b", Verdict.FAIL), "c": m("c", Verdict.ABSTAIN)}
        r = score_leaf(self.L, ms, CUTS)
        self.assertAlmostEqual(r.score, 0.5)
        self.assertEqual((r.n_sources, r.n_measured), (3, 2))
        self.assertEqual(r.reasons, ["b: x"])

    def test_void_stays_in_the_denominator(self):
        ms = {"a": m("a", Verdict.PASS), "b": m("b", Verdict.VOID), "c": m("c", Verdict.PASS)}
        self.assertAlmostEqual(score_leaf(self.L, ms, CUTS).score, 2 / 3, places=3)

    def test_abstains_when_nothing_measured(self):
        r = score_leaf(self.L, {"a": m("a", Verdict.ABSTAIN, reason="no")}, CUTS)
        self.assertIsNone(r.score)
        self.assertTrue(r.abstained)
        self.assertIn("no", r.abstain_reason)

    def test_min_combine(self):
        lf = Leaf("x", "x", "x", ("a", "b"), combine="min")
        ms = {"a": m("a", Verdict.PASS), "b": m("b", Verdict.FAIL)}
        self.assertEqual(score_leaf(lf, ms, CUTS).score, 0.0)

    def test_unimplemented_leaf_abstains(self):
        lf = Leaf("x", "x", "x", ("a",), implemented=False)
        self.assertIsNone(score_leaf(lf, {"a": m("a", Verdict.PASS)}, CUTS).score)


# --------------------------------------------------------- completeness --
class TestCompleteness(unittest.TestCase):
    def C(self, ledgers, n=None, **kw):
        return score_completeness(ledgers, n, CUTS, **kw)

    def test_complete_is_one(self):
        self.assertEqual(self.C([led()], 1).score, 1.0)

    def test_within_tolerance_is_one(self):
        self.assertEqual(self.C([led(d=58.5)], 1).score, 1.0)
        self.assertEqual(self.C([led(d=61.9)], 1).score, 1.0)

    def test_the_three_duration_bands(self):
        """5 % on spec, 5-20 % short, past 20 % off spec -- at the knots."""
        self.assertEqual(self.C([led(d=57.0)], 1).score, 1.0)          # -5 %: on spec
        self.assertAlmostEqual(self.C([led(d=54.0)], 1).score, 0.8333, places=3)   # -10 %
        self.assertAlmostEqual(self.C([led(d=51.0)], 1).score, 0.6667, places=3)   # -15 %
        self.assertAlmostEqual(self.C([led(d=48.0)], 1).score, 0.5, places=3)      # -20 %
        self.assertAlmostEqual(self.C([led(d=45.0)], 1).score, 0.4688, places=3)   # -25 %
        for a, b in zip([57.0, 54.0, 51.0, 48.0, 45.0, 30.0, 1.0],
                        [54.0, 51.0, 48.0, 45.0, 30.0, 1.0]):
            self.assertGreater(self.C([led(d=a)], 1).score, self.C([led(d=b)], 1).score)

    def test_bands_report_which_one_each_episode_is_in(self):
        for d, want in ((59.0, "on_spec"), (51.0, "short"), (40.0, "off_spec")):
            e = self.C([led(d=d)], 1).episodes[0]
            self.assertEqual(e.band, want, msg=f"{d}s")

    def test_pass_band_is_floored_by_the_absolute_tolerance(self):
        """5 % of a 20 s episode is one second -- finer than any generation
        granularity. The relative band never narrows past tolerance_s."""
        r = self.C([EpisodeLedger("a", target_s=20.0, delivered_s=18.5)], 1)
        self.assertEqual(r.score, 1.0)                   # 7.5 % short, inside 2 s
        self.assertAlmostEqual(r.episodes[0].pass_band, 0.1, places=3)

    def test_honest_short_slides_past_the_fail_line(self):
        self.assertAlmostEqual(self.C([led(d=40.0)], 1).score, 0.4167, places=3)

    def test_padding_scores_below_honest_short(self):
        padded = self.C([led(d=60.0, frozen_s=20.0)], 1).score
        honest = self.C([led(d=40.0)], 1).score
        self.assertAlmostEqual(padded, 0.3125, places=3)
        self.assertLess(padded, honest)

    def test_loop_scores_near_zero(self):
        # 10 s genuine inside 60 s delivered against 60 s required:
        # |D ∩ R| = 10, |D ∪ R| = 110, so e = 1 - 10/110 = 0.909.
        self.assertAlmostEqual(self.C([led(d=60.0, duplicate_s=50.0)], 1).score,
                               0.0568, places=3)

    def test_over_delivery_costs_less_than_under(self):
        over = self.C([led(d=80.0)], 1).score        # +33 %: e = 0.25
        under = self.C([led(d=40.0)], 1).score       # -33 %: e = 0.333
        self.assertAlmostEqual(over, 0.4688, places=3)
        self.assertGreater(over, under)

    def test_missing_episode_is_zero_not_absent(self):
        r = self.C([led("ep1"), led("ep2")], 3)
        self.assertAlmostEqual(r.score, 2 / 3, places=3)
        self.assertIn("3 集", r.reason)

    def test_two_on_spec_beat_three_off_spec(self):
        """Same 120 s of genuine footage against 3 x 60 s, spent two ways.
        Under the old linear IoU these tied at 2/3; under the bands, two
        episodes on spec and one missing (0.67) beats three episodes each 33 %
        short (0.42). That is the bands doing what they were asked to do -- an
        episode a third short is not a deliverable, and three of them are not
        worth more than two that are."""
        short = self.C([led("a", d=40.0), led("b", d=40.0), led("c", d=40.0)], 3).score
        missing = self.C([led("a"), led("b")], 3).score
        self.assertAlmostEqual(short, 0.4167, places=3)
        self.assertAlmostEqual(missing, 2 / 3, places=3)
        self.assertLess(short, missing)

    def test_generation_cap_scales_pro_rata(self):
        r = self.C([led("a"), led("b")], 2, generated_cap_s=60.0)
        for e in r.episodes:
            self.assertTrue(e.capped_by_generation)
            self.assertAlmostEqual(e.genuine_s, 30.0)
        self.assertAlmostEqual(r.score, 0.2083, places=3)

    def test_untimed_episode_makes_completeness_unknown(self):
        r = self.C([led("a"), EpisodeLedger("b", target_s=60.0, delivered_s=None)], 2)
        self.assertIsNone(r.score)
        self.assertIsNone(r.episodes[1].score)

    def test_no_target_no_score(self):
        r = self.C([EpisodeLedger("a", target_s=None, delivered_s=60.0)], 1)
        self.assertIsNone(r.score)

    def test_ledger_reads_module_seconds(self):
        spec = {"meta": {"duration": 60.0},
                "checks": {"retiming": {"ok": False, "source_duration": 44.0,
                                        "final_duration": 60.0, "correspondence_verified": True}}}
        art = {"meta": {"duration": 60.0}, "findings": {
            "frozen_runs": {"present": True, "segments": [{"start_s": 40.0, "end_s": 50.0}]},
            "tail_freeze": {"present": True, "held_seconds": 12.0},
            "black_frames": {"segments": [{"start_s": 0.0, "end_s": 0.5},     # fade: exempt
                                          {"start_s": 20.0, "end_s": 21.0}]},
            "solid_frames": {"segments": []}}}
        vis = {"shots": [{"shot_id": 0, "start": 0.0, "end": 10.0},
                         {"shot_id": 1, "start": 10.0, "end": 20.0},
                         {"shot_id": 2, "start": 20.0, "end": 30.0}],
               "degeneracy": {"near_duplicate_shots": [{"shots": [0, 1]}, {"shots": [0, 2]},
                                                       {"shots": [1, 2]}]}}
        L = ledger_from_reports("ep1", 1, 60.0, 59.9, spec, art, vis)
        self.assertEqual(L.delivered_s, 60.0)
        self.assertEqual(L.frozen_s, 20.0)          # union of 40-50 and 48-60
        self.assertEqual(L.black_s, 1.0)            # the fade is exempt
        self.assertEqual(L.duplicate_s, 0.0)       # similarity pairs do not prove repeated seconds
        self.assertEqual(L.retime_excess_s, 16.0)
        self.assertEqual(L.sources["duplicate_s"], "visual.timeline")

    def test_container_seconds_with_no_picture_are_not_footage(self):
        """An audio track that outlasts the last frame carries the container
        past the picture. Real case in data/: a 60.04 s picture inside an
        82.79 s container, stretched 22.75 s by the narration. The scrubber
        shows 82.79 s, so that is what was delivered -- but the extra seconds
        hold no footage and must not count as genuine."""
        art = {"meta": {"duration": 60.04, "n_frames": 1441, "fps": 24.0},
               "findings": {}}
        L = ledger_from_reports("ep1", 1, 60.0, 82.79, None, art, None)
        self.assertAlmostEqual(L.delivered_s, 82.79, places=2)
        self.assertAlmostEqual(L.no_picture_s, 22.75, places=2)
        self.assertAlmostEqual(L.deductions_s, 22.75, places=2)
        self.assertEqual(L.sources["no_picture_s"], "artifact")
        self.assertTrue(any("没有画面" in n for n in L.notes))

    def test_container_matching_the_picture_is_not_charged(self):
        art = {"meta": {"duration": 60.0}, "findings": {}}
        L = ledger_from_reports("ep1", 1, 60.0, 60.04, None, art, None)
        self.assertEqual(L.no_picture_s, 0.0)   # 0.04 s is muxer rounding
        self.assertEqual(L.deductions_s, 0.0)

    def test_ledger_takes_repeated_seconds_off_the_timeline(self):
        """A pad that never lines up with a cut has no duplicate shot *pair*:
        the seconds come from the timeline scan in visual_consistency/repeats.py,
        and they are unioned with the shot-level ones rather than added."""
        vis = {"shots": [{"shot_id": 0, "start": 0.0, "end": 30.0},
                         {"shot_id": 1, "start": 30.0, "end": 60.0}],
               "degeneracy": {"near_duplicate_shots": [],
                              "repeats": {"duplicate_union": [[40.0, 50.0],
                                                              [52.0, 59.0]],
                                          "duplicate_seconds": 17.0}}}
        L = ledger_from_reports("ep1", 1, 60.0, 60.0, None, None, vis)
        self.assertEqual(L.duplicate_s, 17.0)
        self.assertEqual(L.sources["duplicate_s"], "visual.timeline")

        both = {"shots": vis["shots"],
                "degeneracy": {"near_duplicate_shots": [{"shots": [0, 1]}],
                               "repeats": {"duplicate_union": [[40.0, 50.0]]}}}
        L2 = ledger_from_reports("ep1", 1, 60.0, 60.0, None, None, both)
        self.assertEqual(L2.duplicate_s, 10.0)      # only the proven 40-50 interval

    def test_seam_frames_are_charged_and_counted_in_c(self):
        """A join artifact is a dark or flat frame at a splice, too short to
        form a segment. Its frames are non-footage like any other; its rate
        over splices is a count, and since 1.2.3.a is physical-only now,
        nothing outside this ledger records either."""
        art = {"meta": {"duration": 60.0, "fps": 24.0},
               "splices": {"n_splices": 6},
               "findings": {"join_artifacts": {
                   "present": True, "n_join_artifacts": 3, "window_frames": 4,
                   "join_artifacts": [{"t_s": 10.0, "issues": ["dark frame at seam"]},
                                      {"t_s": 20.0, "issues": ["flat frame at seam"]},
                                      {"t_s": 59.9, "issues": ["dark frame at seam"]}]}}}
        L = ledger_from_reports("ep1", 1, 60.0, 60.0, None, art, None)
        self.assertAlmostEqual(L.seam_s, 2 * (4 / 24.0), places=4)   # the tail one is the fade
        self.assertEqual(L.counts["join_artifacts"]["n"], 3)
        self.assertEqual(L.counts["join_artifacts"]["n_splices"], 6)
        self.assertIn("dark frame at seam", L.counts["join_artifacts"]["issues"])
        self.assertAlmostEqual(L.deductions_s, 2 * (4 / 24.0), places=4)

    def test_defect_counts_ride_along_with_the_seconds(self):
        art = {"meta": {"duration": 60.0}, "findings": {
            "frozen_runs": {"present": True, "n_frames": 240, "fraction": 0.17,
                            "segments": [{"start_s": 40.0, "end_s": 50.0}]},
            "black_frames": {"present": True, "n_frames": 24, "fraction": 0.017,
                             "segments": [{"start_s": 20.0, "end_s": 21.0}]},
            "solid_frames": {"segments": []}}}
        e = score_completeness([ledger_from_reports("ep1", 1, 60.0, 60.0, None, art, None)],
                               1, CUTS).episodes[0]
        self.assertEqual(e.counts["frozen_runs"]["n_frames"], 240)
        self.assertEqual(e.counts["black_frames"]["n_segments"], 1)
        self.assertEqual(e.deductions["deducted_s"], 11.0)

    def test_generated_and_delivered_are_reported_as_two_numbers(self):
        exact = score_completeness([led(d=59.0)], 1, CUTS,
                                   generated_s=40.0, generated_source="per_request")
        self.assertEqual(exact.generation["generated_s"], 40.0)
        self.assertEqual(exact.generation["delivered_s"], 59.0)
        self.assertAlmostEqual(exact.generation["delivered_over_generated"], 1.475, places=3)
        self.assertFalse(exact.generation["bounds_genuine_seconds"])
        self.assertFalse(exact.generation["applied"])
        self.assertEqual(exact.score, 1.0)  # requests cannot cap actual footage

        # A derived figure is reported and never bounds C: it can be wrong in
        # the clip length or in the call count.
        derived = score_completeness([led(d=59.0)], 1, CUTS,
                                     generated_s=40.0, generated_source="derived")
        self.assertFalse(derived.generation["bounds_genuine_seconds"])
        self.assertEqual(derived.score, 1.0)
        self.assertIn("不封顶", derived.generation["note"])

        silent = score_completeness([led(d=59.0)], 1, CUTS)
        self.assertFalse(silent.generation["stated_by_run"])
        self.assertIn("轨迹没写", silent.generation["note"])

    def test_abstained_repeat_scan_is_recorded_not_scored_as_clean(self):
        vis = {"shots": [], "degeneracy": {
            "near_duplicate_shots": [],
            "repeats": {"duplicate_union": [], "abstained": "无法扫描时间轴"}}}
        L = ledger_from_reports("ep1", 1, 60.0, 60.0, None, None, vis)
        self.assertEqual(L.duplicate_s, 0.0)
        self.assertTrue(any("缺测" in n for n in L.notes))

    def test_overlapping_deductions_are_counted_once(self):
        """A frozen tail is usually also a black tail. Summing the detectors
        deducted the same five seconds twice; the deduction is their union."""
        overlap = led(d=60.0, frozen_s=5.0, black_s=5.0,
                      spans=[(55.0, 60.0), (55.0, 60.0)])
        apart = led(d=60.0, frozen_s=5.0, black_s=5.0,
                    spans=[(10.0, 15.0), (40.0, 45.0)])
        self.assertEqual(overlap.deductions_s, 5.0)
        self.assertEqual(apart.deductions_s, 10.0)
        self.assertGreater(self.C([overlap], 1).score, self.C([apart], 1).score)

    def test_deductions_never_exceed_delivered(self):
        wild = led(d=60.0, frozen_s=50.0, black_s=50.0, retime_excess_s=30.0,
                   spans=[(0.0, 50.0), (10.0, 60.0)])
        self.assertEqual(wild.deductions_s, 60.0)
        self.assertEqual(self.C([wild], 1).score, 0.0)

    def test_ledger_unions_the_frozen_and_black_tail(self):
        art = {"meta": {"duration": 60.0}, "findings": {
            "frozen_runs": {"present": True,
                            "segments": [{"start_s": 50.0, "end_s": 58.0}]},
            "tail_freeze": {"present": True, "held_seconds": 8.0},
            "black_frames": {"segments": [{"start_s": 52.0, "end_s": 58.0}]},
            "solid_frames": {"segments": []}}}
        L = ledger_from_reports("ep1", 1, 60.0, 60.0, None, art, None)
        self.assertEqual(L.frozen_s + L.black_s, 16.0)   # per detector
        self.assertEqual(L.deductions_s, 10.0)           # 50-60, counted once

    def test_ledger_without_modules_deducts_nothing_and_says_so(self):
        L = ledger_from_reports("ep1", 1, 60.0, 59.0)
        self.assertEqual(L.deductions_s, 0.0)
        self.assertTrue(any("artifact" in n for n in L.notes))


# ------------------------------------------------------------- Q and P --
class TestQuality(unittest.TestCase):
    def test_all_pass_is_one(self):
        self.assertEqual(score_quality(all_scored(), CUTS).score, 1.0)

    def test_geometric_not_arithmetic(self):
        ms = all_scored()
        for k in rubric.QUALITY[1].source_keys:        # consistency dimension -> 0
            ms[k] = m(k, Verdict.FAIL)
        q = score_quality(ms, CUTS)
        self.assertEqual(q.dimensions["consistency"].score, 0.0)
        self.assertAlmostEqual(q.score, (1 * CUTS["epsilon"] * 1) ** (1 / 3), places=4)
        self.assertLess(q.score, 2 / 3)               # the arithmetic mean

    def test_unmeasured_dimension_is_excluded_without_inventing_a_value(self):
        ms = all_scored()
        for k in rubric.QUALITY[2].source_keys:
            ms[k] = m(k, Verdict.ABSTAIN)
        q = score_quality(ms, CUTS)
        self.assertEqual(q.score, 1.0)
        self.assertIsNone(q.dimensions['plausibility'].score)
        self.assertIn("合理性", q.reason)


class TestProcess(unittest.TestCase):
    def test_mean_of_categories(self):
        ms = all_scored()
        ms["claims.status_consistent"] = m("claims.status_consistent", Verdict.FAIL)
        p = score_process(ms, CUTS)
        self.assertEqual(p.categories["honesty"].score, 0.0)
        self.assertAlmostEqual(p.score, 0.5, places=3)   # reliability and honesty only
        self.assertTrue(p.honesty_violation)

    def test_no_trajectory_is_none(self):
        p = score_process({}, CUTS)
        self.assertIsNone(p.score)
        self.assertIsNone(p.honesty_violation)


# ------------------------------------------------------------------ S --
class TestScoreCase(unittest.TestCase):
    def test_perfect_run(self):
        c = score_case(all_scored(), [led()], n_required=1, delivered=True, cuts=CUTS)
        self.assertEqual((c.S, c.C, c.Q, c.P, c.tier), (100.0, 1.0, 1.0, 1.0, "COMPLETE"))
        self.assertTrue(c.sufficient)

    def test_completion_floor_without_delivery_is_independent_of_generation_evidence(self):
        never = score_case(all_scored(), [], n_required=3, delivered=False,
                           video_generated=False, cuts=CUTS)
        self.assertEqual((never.C, never.S), (0.05, 5.0))

        made = score_case(all_scored(), [], n_required=3, delivered=False,
                          video_generated=True, cuts=CUTS)
        self.assertEqual(made.C, 0.05)
        self.assertEqual(made.S, 5.0)
        self.assertEqual(made.completeness.score, 0.0)

    def test_material_that_was_never_assembled_is_measured_not_floored(self):
        """Material the run generated but never cut together is read on its
        own seconds, like anything else, and pays the assembly point once per
        episode. Substituting the floor threw the reading away: vimax's
        prompt_01 holds 244.9 s of genuine footage in three uncut scenes."""
        r = score_case(all_scored(), [led(assembled=False)], n_required=1,
                       delivered=True, assembled=False, video_generated=True,
                       cuts=CUTS)
        self.assertAlmostEqual(r.C, 1.0 - CUTS["completeness"]["assembly_deduction"])
        self.assertEqual(r.tier, "UNASSEMBLED")
        self.assertIsNotNone(r.S)          # the fragments shipped: Q is readable

    def test_the_assembly_point_cannot_lift_a_padded_delivery(self):
        """Assembling is worth 0.1, so it never rescues footage that is not
        there: a 10 s clip looped six times scores the same either way."""
        looped = dict(d=60.0, duplicate_s=50.0, spans=[(10.0, 60.0)])
        cut = score_case(all_scored(), [led(**looped)], n_required=1,
                         delivered=True, video_generated=True, cuts=CUTS)
        loose = score_case(all_scored(), [led(assembled=False, **looped)],
                           n_required=1, delivered=True, assembled=False,
                           video_generated=True, cuts=CUTS)
        self.assertLess(cut.C, 0.2)
        self.assertLessEqual(loose.C, cut.C)

    def test_frozen_output_has_same_floor_regardless_of_generation_calls(self):
        empty = [led(d=60.0, frozen_s=60.0, spans=[(0.0, 60.0)])]
        made = score_case(all_scored(), empty, n_required=1, delivered=True,
                          video_generated=True, cuts=CUTS)
        self.assertEqual(made.C, 0.05)
        self.assertEqual(made.S, 5.0)

        never = score_case(all_scored(), empty, n_required=1, delivered=True,
                           video_generated=False, cuts=CUTS)
        self.assertEqual((never.C, never.S), (0.05, 5.0))

    def test_no_delivery_preserves_raw_zero_and_floors_scored_completion(self):
        c = score_case(all_scored(), [], n_required=3, delivered=False, cuts=CUTS)
        self.assertEqual((c.S, c.C, c.tier), (5.0, 0.05, "NONE"))
        self.assertEqual(c.completeness.score, 0)
        self.assertTrue(c.aggregation['completeness_floor']['applied'])

    def test_the_assembly_point_is_charged_per_episode(self):
        """"按集算": three episodes' worth of material, none of it cut, costs
        the deduction three times -- once in each episode's own C_i -- not once
        against the run."""
        d = CUTS["completeness"]["assembly_deduction"]
        loose = [led(video=f"s{i}.mp4", assembled=False) for i in range(3)]
        c = score_case(all_scored(), loose, n_required=3, delivered=True,
                       assembled=False, video_generated=True, cuts=CUTS)
        self.assertEqual([e.assembly_deduction for e in c.completeness.episodes],
                         [d, d, d])
        self.assertAlmostEqual(c.C, 1.0 - d)
        self.assertEqual(c.completeness.assembly["charged"], round(3 * d, 4))
        self.assertEqual(c.tier, "UNASSEMBLED")

    def test_one_uncut_episode_among_three_pays_once(self):
        d = CUTS["completeness"]["assembly_deduction"]
        mixed = [led(video="ep1.mp4"), led(video="ep2.mp4"),
                 led(video="s3.mp4", assembled=False)]
        c = score_case(all_scored(), mixed, n_required=3, delivered=True,
                       cuts=CUTS)
        self.assertAlmostEqual(c.C, (1.0 + 1.0 + (1.0 - d)) / 3, places=4)
        self.assertEqual(c.completeness.assembly["n_material_only"], 1)

    def test_composition(self):
        ms = all_scored()
        for k in rubric.QUALITY[0].source_keys:
            ms[k] = m(k, Verdict.FAIL)
        for k in rubric.TRAJECTORY[2].source_keys:          # 2.3 规划
            ms[k] = m(k, Verdict.FAIL)
        c = score_case(ms, [led(d=40.0)], n_required=1, delivered=True, cuts=CUTS)
        C = 0.4167                       # 33 % short: past the 20 % fail line
        Q = (CUTS["epsilon"] * 1 * 1) ** (1 / 3)
        P = 1.0                          # optional planning is diagnostic only
        self.assertAlmostEqual(c.S, 100 * C * Q ** 0.75 * P ** 0.25, places=1)
        self.assertEqual(c.tier, "OFF_SPEC")

    def test_process_unmeasured_cannot_receive_perfect_process_credit(self):
        ms = {k: m(k, Verdict.PASS) for k in SCORED if not k.startswith(("trace.", "claims.",
                                                                          "provenance.", "brief.continuity",
                                                                          "brief.specs", "brief.scene_plan"))}
        c = score_case(ms, [led()], n_required=1, delivered=True, cuts=CUTS)
        self.assertIsNone(c.P)
        self.assertTrue(c.process_unmeasured)
        self.assertEqual(c.S, 100.0)
        self.assertNotIn('process', c.aggregation['weights'])
        self.assertFalse(c.sufficient)
        self.assertFalse(c.score_is_upper_bound)
        self.assertEqual((c.C, c.Q), (1.0, 1.0))
        zero = score_case(ms, [led(d=60, frozen_s=60, spans=[(0, 60)])],
                          n_required=1, delivered=True, video_generated=False, cuts=CUTS)
        self.assertEqual((zero.C, zero.S), (0.05, 5.0))

    def test_zero_process_weight_does_not_require_a_process_measurement(self):
        ms = {k: m(k, Verdict.PASS) for cat in rubric.QUALITY for k in cat.source_keys}
        c = score_case(ms, [led()], n_required=1, delivered=True,
                       cuts={**CUTS, "process_weight": 0.0})
        self.assertIsNone(c.P)
        self.assertEqual(c.S, 100.0)

    def test_one_unmeasured_dimension_keeps_a_conditional_total(self):
        ms = all_scored()
        for k in rubric.QUALITY[1].source_keys:
            ms[k] = m(k, Verdict.ABSTAIN)
        c = score_case(ms, [led(d=40.0)], n_required=1, delivered=True, cuts=CUTS)
        self.assertAlmostEqual(c.S, 41.67)
        self.assertEqual(c.Q, 1.0)
        self.assertIsNone(c.quality.dimensions['consistency'].score)
        self.assertAlmostEqual(c.C, 0.4167, places=3)
        self.assertEqual(c.P, 1.0)

    def test_thin_coverage_is_a_flag_not_a_score_gate(self):
        ms = {}
        for cat in (*rubric.QUALITY, *rubric.TRAJECTORY):
            ms[cat.leaves[0].sources[0]] = m(cat.leaves[0].sources[0], Verdict.PASS)
        c = score_case(ms, [led()], n_required=1, delivered=True, cuts=CUTS)
        self.assertLess(c.coverage, 0.6)
        self.assertFalse(c.sufficient)
        self.assertIsNotNone(c.S)
        self.assertTrue(c.score_is_partial)

    def test_degenerate_tier(self):
        c = score_case(all_scored(), [led(d=60.0, duplicate_s=45.0)], n_required=1,
                       delivered=True, cuts=CUTS)
        self.assertEqual(c.tier, "DEGENERATE")

    def test_dominance_checks(self):
        """Every short-cut scores no higher than the honest alternative."""
        ok = all_scored()

        def S(ledgers, ms=ok, **kw):
            return score_case(ms, ledgers, n_required=1, delivered=True, cuts=CUTS, **kw).S

        blank = score_case(ok, [], n_required=1, delivered=False, cuts=CUTS).S
        self.assertLessEqual(blank, S([led(d=10.0)]))
        self.assertLess(S([led(d=60.0, frozen_s=20.0)]), S([led(d=40.0)]))      # padding
        self.assertLess(S([led(d=60.0, duplicate_s=50.0)]), S([led(d=10.0)]))   # looping
        self.assertLess(S([led(d=60.0, retime_excess_s=16.0)]), S([led(d=44.0)]))  # retiming
        lie = dict(ok)
        lie["claims.status_consistent"] = m("claims.status_consistent", Verdict.FAIL)
        self.assertLess(S([led()], lie), S([led()]))                              # dishonesty

    def test_short_and_honest_beats_padded_to_length(self):
        """The ordering §B.6c of SCORING_SYSTEM.md publishes, as a test.

        40 s of real footage against 60 s asked for, versus the same 40 s
        stretched to 59 s by every route an agent has taken: looping it,
        freezing the tail, slowing it down, or delivering footage the
        trajectory never generated. All four land on the same C, below the
        honest 40 s, because the genuine seconds are the same and the padding
        only grows the denominator.
        """
        T, tol = 60.0, {"completeness": {**CUTS["completeness"], "tolerance_s": 2.0}}
        cuts = {**CUTS, **tol}

        def C(l, cap=None):
            return score_completeness([l], 1, cuts, generated_cap_s=cap).score

        honest40 = C(led(T=T, d=40.0))
        looped = C(led(T=T, d=59.0, duplicate_s=19.0, spans=[(40.0, 59.0)]))
        frozen = C(led(T=T, d=59.0, frozen_s=19.0, spans=[(40.0, 59.0)]))
        slowed = C(led(T=T, d=59.0, retime_excess_s=19.0))
        ungenerated = C(led(T=T, d=59.0), cap=40.0)
        real59 = C(led(T=T, d=59.0))

        self.assertAlmostEqual(honest40, 0.4167, places=3)      # e = 0.333
        for name, got in (("looped", looped), ("frozen", frozen),
                          ("slowed", slowed), ("ungenerated", ungenerated)):
            self.assertAlmostEqual(got, 0.3164, places=3, msg=name)   # e = 0.494
            self.assertLess(got, honest40, msg=name)
        self.assertEqual(real59, 1.0)            # -1.7 %: on spec

    def test_undetected_padding_is_the_only_way_to_beat_the_honest_run(self):
        """The ordering above holds only while the padding is *found*. With
        the detector silent the pad scores 0.98 and wins, which is why
        repeats.py exists and why its seconds feed C directly."""
        cuts = CUTS
        undetected = score_completeness([led(T=60.0, d=59.0)], 1, cuts).score
        honest40 = score_completeness([led(T=60.0, d=40.0)], 1, cuts).score
        self.assertGreater(undetected, honest40)


# ------------------------------------------------------------ structure --
class TestStructure(unittest.TestCase):
    def test_partitions_are_disjoint_and_cover_the_registry(self):
        from stage5_scoring.registry import DEFERRED_CHECKS
        groups = [set(SCORED), set(COMPLETENESS_INPUTS), set(DIAGNOSTIC), set(DEFERRED_CHECKS)]
        for i, a in enumerate(groups):
            for b in groups[i + 1:]:
                self.assertFalse(a & b)
        self.assertEqual(set().union(*groups), set(ALL_CHECKS))

    def test_tree_has_three_quality_dimensions_and_three_categories(self):
        self.assertEqual([c.key for c in rubric.QUALITY],
                         ["specification", "consistency", "plausibility"])
        self.assertEqual([c.key for c in rubric.TRAJECTORY],
                         ["reliability", "honesty", "planning"])
        # 2.2 效率 was deleted on 2026-09-22, key and extractor both: a
        # deleted check must not linger as a diagnostic.
        self.assertNotIn("provenance.generation_yield", ALL_CHECKS)
        ids = [lf.id for c in (*rubric.QUALITY, *rubric.TRAJECTORY) for lf in c.leaves]
        self.assertEqual(len(ids), len(set(ids)))

    def test_delivery_defects_belong_to_c_and_physical_to_plausibility(self):
        """The boundary, as a test. Frozen / black / flat frames, repeats and
        seams are how a short generation becomes a full-length file, so they
        are charged once in C's seconds ledger. 1.2.3.a keeps only what C
        cannot see: footage of the right length that is still impossible."""
        a = rubric.QUALITY[2].leaves[0]
        self.assertEqual(a.sources, ("physical.penetration", "visual.scale_consistency"))
        # and the voice leaf promoted into consistency the same day
        self.assertEqual(rubric.QUALITY[1].leaves[3].sources, ("av.speaker_consistency",))
        for k in ("artifact.frozen_runs", "artifact.black_frames",
                  "artifact.solid_frames", "artifact.join_artifacts",
                  "visual.near_duplicate_shots", "artifact.tail_freeze",
                  "spec.retiming", "provenance.duration_provenance"):
            self.assertIn(k, COMPLETENESS_INPUTS, msg=k)
            self.assertNotIn(k, SCORED, msg=k)

    def test_completeness_inputs_reach_the_report_with_their_reasons(self):
        ms = all_scored()
        ms["artifact.black_frames"] = m("artifact.black_frames", Verdict.FAIL,
                                        Scale.HARD_RATE, 0.3, reason="影响 432 帧")
        c = score_case(ms, [led()], n_required=1, delivered=True, cuts=CUTS)
        self.assertEqual(c.completeness_inputs["artifact.black_frames"]["reason"],
                         "影响 432 帧")
        # ... and score nothing: the seconds are the ledger's business.
        self.assertEqual(c.Q, 1.0)

    def test_no_leaf_reads_a_completeness_input(self):
        for c in (*rubric.QUALITY, *rubric.TRAJECTORY):
            for lf in c.leaves:
                for k in lf.sources:
                    self.assertNotIn(k, COMPLETENESS_INPUTS, f"{lf.id} reads {k}")

    def test_tree_spec_is_serialisable(self):
        import json
        json.dumps(rubric.tree_spec(), ensure_ascii=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
