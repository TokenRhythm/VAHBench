"""Output equivalence, honest missingness and extraction-omission controls."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shared.types import Measurement
from shared.types import Verdict
from shared.types import Scale
from stage5_scoring import score_case
from stage5_scoring import EpisodeLedger
from stage5_scoring.registry import SCORED
from stage5_scoring.capabilities import METHOD_DIAGNOSTIC_SOURCES
from stage5_scoring.score import aggregate_total
from stage5_scoring.score import completion_factor
from stage1_inputs.schema import TrajectoryFacts
import stage1_inputs.extractors as extractors
from stage1_inputs.observability import recheck_facts
from run import _finalize_evaluation_status


def measured(key, verdict=Verdict.PASS, value=None, scale=Scale.BOOLEAN):
    return Measurement(key=key, verdict=verdict, value=value, scale=scale)


class MethodNeutrality(unittest.TestCase):
    def score(self, measurements, **kwargs):
        return score_case(measurements, [EpisodeLedger('episode', target_s=60, delivered_s=60)],
                          n_required=1, delivered=True, **kwargs)

    def test_same_output_different_optional_workflows_has_identical_score_and_coverage(self):
        ms = {k: measured(k) for k in SCORED}
        ms['av.clipping'] = measured('av.clipping', Verdict.FAIL)
        baseline = self.score(ms)
        for verdict in Verdict:
            variant = dict(ms)
            variant.update({k: measured(k, verdict) for k in METHOD_DIAGNOSTIC_SOURCES})
            result = self.score(variant)
            self.assertEqual((result.S, result.C, result.Q, result.P, result.coverage),
                             (baseline.S, baseline.C, baseline.Q, baseline.P, baseline.coverage))
            self.assertFalse(result.process.categories['planning'].missing_sources)
            self.assertEqual(result.process.categories['planning'].n_applicable, 0)

    def test_missing_process_is_not_imputed_and_weights_are_renormalised(self):
        score, audit = aggregate_total(.8, {'specification': .5, 'consistency': .5, 'plausibility': .5}, None, {})
        self.assertEqual(score, 40)
        self.assertEqual(set(audit['weights']), {'specification', 'consistency', 'plausibility'})
        self.assertAlmostEqual(sum(audit['weights'].values()), 1)
        self.assertFalse(audit['missing_values_imputed'])

    def test_unknown_and_not_applicable_have_same_score_but_different_coverage(self):
        ms = {k: measured(k) for k in SCORED}
        ms['physical.penetration'] = measured('physical.penetration', Verdict.ABSTAIN)
        unknown = self.score(ms)
        ms['physical.penetration'] = measured('physical.penetration', Verdict.NOT_APPLICABLE)
        inapplicable = self.score(ms)
        self.assertEqual(unknown.S, inapplicable.S)
        self.assertLess(unknown.quality.dimensions['plausibility'].source_coverage,
                        inapplicable.quality.dimensions['plausibility'].source_coverage)

    def test_completeness_only_is_labelled_not_fabricated_quality(self):
        result = self.score({})
        self.assertEqual(result.S, 100)
        self.assertIsNone(result.Q)
        self.assertIsNone(result.P)
        self.assertEqual(result.aggregation['basis'], 'completeness_only')
        self.assertTrue(result.score_is_partial)

    def test_no_delivery_uses_floor_regardless_of_generation_evidence(self):
        for called in [False, True, None]:
            result = score_case({}, [], n_required=3, delivered=False, video_generated=called)
            self.assertEqual((result.C, result.S), (.05, 5))
            self.assertEqual(result.completeness.score, 0)
            self.assertTrue(result.aggregation['completeness_floor']['applied'])

    def test_same_frozen_output_has_same_score_with_or_without_a_generator_call(self):
        results = [score_case({}, [EpisodeLedger('v', target_s=60, delivered_s=60,
                      frozen_s=60, spans=[(0, 60)])], n_required=1,
                      delivered=True, video_generated=called) for called in [False, True]]
        self.assertEqual([(r.C, r.S) for r in results], [(.05, 5), (.05, 5)])

    def test_floor_does_not_fabricate_unknown_completion_or_lower_valid_score(self):
        for raw, expected in [(None, None), (0, .05), (.02, .05), (.05, .05), (.8, .8)]:
            actual, audit = completion_factor(raw, delivered=True, video_generated=False, epsilon=.05)
            self.assertEqual(actual, expected)
            self.assertEqual(audit['raw'], raw)
            self.assertEqual(audit['applied'], raw is not None and raw < .05)

    def test_low_coverage_retains_real_defect(self):
        good = self.score({'spec.resolution': measured('spec.resolution')})
        bad = self.score({'spec.resolution': measured('spec.resolution', Verdict.FAIL)})
        self.assertIsNotNone(bad.S)
        self.assertLess(bad.S, good.S)
        self.assertFalse(bad.sufficient)


class Observability(unittest.TestCase):
    def prepared(self, root):
        f = TrajectoryFacts()
        extractors._lm_parts(root, f)
        f.lm_audit.update(chunks_total=1, chunks_valid=1, input_complete=True)
        return f

    def test_structured_positive_overturns_empty_checklist_not_marked_unused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'generate.py').write_text("request = {'prompt': 'A woman in a blue coat walks across a busy street at sunset'}", encoding='utf8')
            facts = self.prepared(root)
            facts.requested.prompts = []
            result = recheck_facts(root, facts)
            self.assertTrue(result['passed'])
            self.assertIn('requested.prompts', result['recovered_fields'])
            self.assertTrue(facts.requested.prompts)
            self.assertEqual(result['method_fields']['requested.prompts'], 'recovered_extraction_omission')

    def test_source_absence_is_not_proof_of_method_nonuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'agent.log').write_text('Finished rendering.', encoding='utf8')
            facts = self.prepared(root)
            result = recheck_facts(root, facts)
            self.assertTrue(result['source_inventory_verified'])
            self.assertFalse(result['semantic_absence_proven'])
            self.assertEqual(result['method_fields']['conditioning'], 'not_observed_after_source_recheck')

    def test_added_raw_evidence_invalidates_cached_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'agent.log').write_text('Rendering.', encoding='utf8')
            facts = self.prepared(root)
            (root/'another.log').write_text('A new final claim.', encoding='utf8')
            result = recheck_facts(root, facts)
            self.assertFalse(result['passed'])
            self.assertIn('extraction_sources_changed', result['errors'])
            report = {'S': 80, 'C': 1}
            _finalize_evaluation_status(report, facts)
            self.assertEqual(report['evaluation_status'], 'extraction_failed')
            self.assertIsNone(report['S'])

    def test_failed_extraction_never_becomes_method_inapplicability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            facts = self.prepared(root)
            facts.lm_audit['chunks_valid'] = 0
            result = recheck_facts(root, facts)
            self.assertFalse(result['passed'])
            self.assertEqual(result['method_fields']['conditioning'], 'extraction_unverified')


if __name__ == '__main__':
    unittest.main()
