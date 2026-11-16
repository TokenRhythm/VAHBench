"""Scoring invariants: same evidence once, uniform mappings, missing is unknown."""
import copy
import itertools
import json
from dataclasses import asdict
import unittest

from shared.types import Measurement as M
from shared.types import Verdict as V
from shared.types import Scale as S
from stage3_evidence.intervals import union_seconds
from stage3_evidence.intervals import repeat_evidence
import stage4_checks.reports as adapters
import stage4_checks.video.brief as expectations
import stage4_checks.cross_checks as flows
from stage4_checks.penalties import normalize_episode
from stage5_scoring.node import DEFAULT_CUT_POINTS as CUTS
from stage5_scoring.node import check_score
from stage5_scoring.node import score_category
from stage5_scoring.registry import SCORED
from stage5_scoring.registry import SOURCE_MAPPINGS
from stage5_scoring.video.quality import QUALITY
from stage5_scoring.process import TRAJECTORY
from stage5_scoring.video.completeness import ledger_from_reports
from stage5_scoring.video.completeness import EpisodeLedger
from stage5_scoring.score import score_case
from run import _merge_worst
from run import _preserve_quality_episode_gaps


def visual(spans, pairs=1, abstained=""):
    return {"duration": 60, "shots": [{"shot_id": 0, "start": 0, "end": 30},
                                     {"shot_id": 1, "start": 30, "end": 60}],
            "degeneracy": {"near_duplicate_shots": [{"shots": [0, 1]}] * pairs,
                           "verdict": {"degenerate": True},
                           "repeats": {"duplicate_union": spans, "abstained": abstained}}}


class TestTemporalEvidence(unittest.TestCase):
    def test_similarity_only_never_manufactures_repeated_seconds(self):
        for n in (0, 1, 40):
            rep = visual([], n)
            led = ledger_from_reports("ep", 1, 60, 60, visual=rep)
            self.assertEqual(led.duplicate_s, 0)
            m = {x.key: x for x in adapters.from_visual(rep)}
            self.assertIs(m['visual.near_duplicate_shots'].verdict, V.PASS)
            self.assertIs(m['visual.identity_consistency'].verdict, V.ABSTAIN)

    def test_real_repeat_spans_are_charged_once_regardless_of_pair_count(self):
        for n in (0, 1, 40):
            rep = visual([[30, 45], [40, 50], [30, 45]], n)
            led = ledger_from_reports("ep", 1, 60, 60, visual=rep)
            self.assertEqual(led.duplicate_s, 20)
            self.assertEqual(led.deductions_s, 20)

    def test_abstained_or_malformed_scan_cannot_charge_stale_spans(self):
        for rep in (visual([[20, 40]], abstained="failed"), visual([[40, 20]]),
                    visual([[float('nan'), 40]]), visual([[90, 100]])):
            led = ledger_from_reports("ep", 1, 60, 60, visual=rep)
            self.assertEqual(led.duplicate_s, 0)
            self.assertIn('timeline_repeats_unmeasured', led.evidence_gaps)

    def test_interval_bounds_and_order_do_not_change_union(self):
        spans = [[-2, 4], [3, 8], [8, 80], [5, 2], [0, float('inf')]]
        for ordered in (spans, list(reversed(spans)), spans * 2):
            self.assertEqual(union_seconds(ordered, 60), 60)

    def test_freeze_tail_and_black_are_separately_unioned_then_combined(self):
        art = {"meta": {"duration": 60}, "findings": {
            "frozen_runs": {"segments": [{"start_s": 10, "end_s": 20}]},
            "tail_freeze": {"present": True, "held_seconds": 10},
            "black_frames": {"segments": [{"start_s": 10, "end_s": 20}]},
            "solid_frames": {"segments": [{"start_s": 15, "end_s": 25}]}}}
        led = ledger_from_reports("ep", 1, 60, 60, artifact=art, visual=visual([[52, 58]]))
        self.assertEqual((led.frozen_s, led.black_s, led.duplicate_s), (20, 15, 6))
        self.assertEqual(led.deductions_s, 25)

    def test_unlocated_counts_do_not_prove_disjoint_defects(self):
        led = EpisodeLedger("ep", delivered_s=60, frozen_s=10, duplicate_s=10, black_s=10)
        self.assertEqual(led.deductions_s, 10)
        led.spans = [(0, 5)]
        self.assertEqual(led.deductions_s, 10)

    def test_static_cut_already_in_C_is_not_a_second_defect(self):
        rep = {"summary": {"n_seams": 2, "n_visible_seams": 2}, "seams": [
            {"t_s": 10, "dead_s": 2, "p_head": 0.001},
            {"t_s": 20, "dead_s": 1, "p_head": 0.001}]}
        before = adapters.from_seam(rep)[0]
        after = adapters.from_seam(rep, [(10, 12)])[0]
        self.assertEqual((before.detail['k'], after.detail['k']), (2, 1))
        self.assertEqual(after.detail['already_charged_in_C'], [10])
        self.assertEqual(after.detail['broken_at'], [20])
        self.assertEqual(after.detail['dead_heads'], 1)
        self.assertIn('1 个切在', after.reason)


class TestUniformReduction(unittest.TestCase):
    def test_numeric_labels_never_change_same_value_score(self):
        for scale in (S.RATE, S.HARD_RATE, S.RATIO):
            for value in (0, .005, .1, .5, 2, 4):
                for mapping in ('band', 'fraction'):
                    scores = [check_score(M('x', v, scale, value), CUTS, mapping) for v in (V.PASS, V.FAIL)]
                    self.assertEqual(scores[0], scores[1])

    def test_worst_merge_uses_leaf_mapping_and_is_order_independent(self):
        for key in ('brief.subtitles_aligned', 'brief.word_cap', 'brief.continuity_contract'):
            rows = [M(key, V.FAIL, S.RATE, v) for v in (.3, .5, .9)]
            for ordered in itertools.permutations(rows):
                merged = {}
                for row in ordered:
                    _merge_worst(merged, {key: row}, CUTS)
                self.assertEqual(merged[key].value, .9)
                self.assertAlmostEqual(check_score(merged[key], CUTS, SOURCE_MAPPINGS[key]), .1)

    def test_nonfinite_numeric_values_are_unknown_not_zero_or_full_credit(self):
        for value in (float('nan'), float('inf'), -1):
            self.assertIsNone(check_score(M('x', V.PASS, S.RATE, value), CUTS))

    def test_all_branches_share_missing_evidence_semantics(self):
        for cat in (*QUALITY, *TRAJECTORY):
            ms = {k: M(k, V.PASS) for k in cat.source_keys if k in SCORED}
            if len(ms) < 2:
                continue
            key = next(iter(ms))
            ms[key] = M(key, V.ABSTAIN)
            result = score_category(cat, ms, CUTS)
            self.assertEqual(result.score, 1)
            self.assertIn(key, result.missing_sources)
            self.assertLess(result.source_coverage, 1)

    def test_known_bad_episode_survives_unknown_episode(self):
        key = 'visual.identity_consistency'
        m = M(key, V.FAIL, S.RATIO, 4)
        acc = {key: m}
        _preserve_quality_episode_gaps(acc, [{'video': 'bad', 'measurements': {key: m}},
                                           {'video': 'unknown', 'measurements': {}}])
        self.assertIs(acc[key].verdict, V.FAIL)
        self.assertEqual(acc[key].detail['missing_episodes'], ['unknown'])
        self.assertEqual(check_score(acc[key], CUTS), 0)
        # Episode evidence and the selected result must not form a cycle.
        json.dumps(asdict(acc[key]))

    def test_missing_is_conditional_and_zero_coverage_is_completeness_only(self):
        ms = {k: M(k, V.PASS) for k in SCORED}
        ms['physical.penetration'] = M('physical.penetration', V.ABSTAIN)
        result = score_case(ms, [EpisodeLedger('ep', target_s=60, delivered_s=60)], n_required=1, delivered=True)
        self.assertTrue(result.score_is_partial)
        self.assertTrue(any('条件分' in n for n in result.notes))
        empty = score_case({}, [EpisodeLedger('ep', target_s=60, delivered_s=60)], n_required=1, delivered=True)
        self.assertEqual(empty.S, 100)
        self.assertEqual(empty.aggregation['basis'], 'completeness_only')
        self.assertIsNone(empty.Q)
        self.assertIsNone(empty.P)


class TestPrerequisiteOwnership(unittest.TestCase):
    def test_silent_and_absent_audio_share_one_owner(self):
        reports = [{'audio_stream': {'present': False}}, {'cleanliness': {'rms': 0}}]
        for rep in reports:
            ms = {m.key: m for m in adapters.from_av(rep)}
            normalize_episode(ms)
            failed = [k for k, m in ms.items() if m.verdict in (V.FAIL, V.VOID)]
            self.assertEqual(failed, ['spec.audio_track'])
            for key in ('av.clipping', 'av.dynamic_range', 'av.long_silence'):
                self.assertEqual(ms[key].detail['suppressed_by'], 'spec.audio_track')

    def test_silent_episode_does_not_hide_clipping_in_audible_episode(self):
        silent = {m.key: m for m in adapters.from_av({'cleanliness': {'rms': 0}})}
        loud = {'av.clipping': M('av.clipping', V.FAIL)}
        for rows in (silent, loud), (loud, silent):
            acc = {}
            for row in rows:
                _merge_worst(acc, row, CUTS)
            self.assertIs(acc['av.clipping'].verdict, V.FAIL)

    def test_geometry_failure_is_not_two_penalties(self):
        ms = {k: M(k, V.FAIL) for k in ('spec.resolution', 'spec.aspect')}
        normalize_episode(ms)
        self.assertIs(ms['spec.resolution'].verdict, V.FAIL)
        self.assertEqual(ms['spec.aspect'].detail['suppressed_by'], 'spec.resolution')

    def test_intermediate_requests_do_not_double_charge_final_specs(self):
        ms = {k: M(k, V.PASS) for k in SCORED}
        baseline = score_case(ms, [EpisodeLedger('ep', target_s=60, delivered_s=60)],
                              n_required=1, delivered=True)
        ms['o2.requested_params'] = M('o2.requested_params', V.FAIL)
        measured = score_case(ms, [EpisodeLedger('ep', target_s=60, delivered_s=60)],
                              n_required=1, delivered=True)
        self.assertEqual(measured.S, baseline.S)

    def test_unmeasured_planning_cannot_be_voided_by_output_drift(self):
        ms = {'brief.continuity_contract': M('brief.continuity_contract', V.ABSTAIN),
              'visual.identity_consistency': M('visual.identity_consistency', V.FAIL, S.RATIO, 4)}
        flows.contract_explains_consistency(ms, flows.Ledger())
        self.assertIs(ms['brief.continuity_contract'].verdict, V.ABSTAIN)


if __name__ == '__main__':
    unittest.main()
