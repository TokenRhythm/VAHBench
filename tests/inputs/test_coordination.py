"""Production coordinator regressions, including unmodified real log records.

Injected responses test validation/merging, not model accuracy. Real files are
only used as extraction inputs; no test calls a provider.
"""

from shared.paths import REPO_ROOT
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import stage1_inputs.extractors as e
from stage1_inputs.schema import TrajectoryFacts
from run import _merge_facts

REPO = REPO_ROOT


def reply(field, value, quote, source='TRAJECTORY.md'):
    group, _, key = field.partition('.')
    p = {group: {key: value} if key else value}
    p['evidence'] = {field: [{'source': source, 'quote': quote}]}
    if field.startswith('claims.'):
        p['field_basis'] = {field: {'source_kind': 'agent_final_claim', 'scope': 'final_delivery'}}
    return p


def coordinate(parts, payloads, focus, **kwargs):
    answers = iter(payloads)
    with patch.object(e, '_lm_parts', return_value=parts):
        return e.lm_extract(Path('.'), model='test', revision='test',
                            call_fn=lambda _: json.dumps(next(answers)),
                            focus_fields=focus, **kwargs)


class TestCoordinator(unittest.TestCase):
    def test_error_counts_use_the_same_evidence_path_as_other_fields(self):
        key = 'errors.n_attempts'
        f = coordinate(['### SOURCE: TRAJECTORY.md\n3 production tasks'],
                       [reply(key, 3, '3 production tasks')], [key])
        self.assertEqual(_merge_facts(TrajectoryFacts(), f).errors.n_attempts, 3)
        self.assertIn(key, f.lm_evidence)
        for p in ({'errors': {'n_attempts': 3}}, reply(key, 3, 'invented')):
            f = coordinate(['### SOURCE: TRAJECTORY.md\n3 production tasks'], [p], [key])
            self.assertIsNone(f.errors.n_attempts)

    def test_scope_is_enforced_even_if_model_ignores_prompt(self):
        p = reply('claims.self_reported_status', 'partial', 'only two delivered')
        p['calls'] = {'total': 999}
        f = coordinate(['### SOURCE: TRAJECTORY.md\nonly two delivered'], [p],
                       ['claims.self_reported_status'])
        self.assertEqual(f.claims.self_reported_status, 'partial')
        self.assertIsNone(f.calls.total)
        self.assertIn('claims.self_reported_status', f.lm_evidence)

    def test_missing_fabricated_and_wrong_source_quotes_are_rejected(self):
        for p in ({'calls': {'total': 3}}, reply('calls.total', 3, 'invented'),
                  reply('calls.total', 3, '3 calls', 'elsewhere.md')):
            f = coordinate(['### SOURCE: TRAJECTORY.md\n3 calls'], [p], ['calls.total'])
            self.assertIsNone(f.calls.total)

    def test_partial_global_counts_are_not_summed_or_treated_as_complete(self):
        parts = ['### SOURCE: TRAJECTORY.md\n3 calls\n' + 'x' * 80,
                 '### SOURCE: TRAJECTORY.md\n4 calls\n' + 'y' * 80]
        ps = [reply('calls.total', n, f'{n} calls') for n in (3, 4)]
        f = coordinate(parts, ps, ['calls.total'], max_chars=130)
        self.assertIsNone(f.calls.total)
        f = coordinate(parts, ps[:1], ['calls.total'], max_chars=130, max_calls=1)
        self.assertEqual(f.lm_audit['chunks_total'], 2)
        self.assertFalse(f.lm_audit['input_complete'])
        self.assertIsNone(f.calls.total)

    def test_conflicting_status_does_not_use_longest_string(self):
        parts = [f'### SOURCE: TRAJECTORY.md\nstatus {s}\n' + 'x' * 70
                 for s in ('ok', 'failed')]
        ps = [reply('claims.self_reported_status', s, 'status ' + s) for s in ('ok', 'failed')]
        f = coordinate(parts, ps, ['claims.self_reported_status'], max_chars=130)
        self.assertIsNone(f.claims.self_reported_status)
        self.assertIn('claims.self_reported_status', f.lm_audit['conflicts'])

    def test_explicit_empty_conditioning_survives_but_invalid_entry_does_not(self):
        for value, expected in (([], []), ([{}], None)):
            f = coordinate(['### SOURCE: TRAJECTORY.md\nall calls used no references'],
                           [reply('conditioning', value, 'all calls used no references')], ['conditioning'])
            self.assertEqual(f.conditioning, expected)

    def test_episode_counts_preserve_repeated_values_and_order(self):
        f = coordinate(['### SOURCE: TRAJECTORY.md\n6 clips per episode, three episodes'],
                       [reply('claims.declared_clips_per_episode', [6, 6, 6],
                              '6 clips per episode, three episodes')],
                       ['claims.declared_clips_per_episode'])
        self.assertEqual(f.claims.declared_clips_per_episode, [6, 6, 6])

    def test_derived_duration_uses_rules_count_after_merge(self):
        det = TrajectoryFacts()
        det.calls.video_gen = 3
        lm = coordinate(['### SOURCE: TRAJECTORY.md\neach clip is 10 seconds'],
                        [reply('requested.clip_durations', [10], 'each clip is 10 seconds')],
                        ['requested.clip_durations'], base_facts=det)
        self.assertIsNone(lm.requested.generated_seconds)
        f = _merge_facts(det, lm)
        self.assertEqual(f.requested.generated_seconds, 30)
        self.assertEqual(f.requested.generated_seconds_source, 'derived')

    def test_lm_error_count_cannot_exceed_rules_attempts(self):
        det, lm = TrajectoryFacts(), TrajectoryFacts()
        det.errors.n_attempts = 3
        lm.errors.n_unrecovered = 10
        f = _merge_facts(det, lm)
        self.assertEqual(f.errors.n_attempts, 3)
        self.assertIsNone(f.errors.n_unrecovered)

    def test_invalid_numbers_never_become_valid_zero_or_empty(self):
        for v in (-1, float('nan'), float('inf'), True, 'NaN'):
            f = TrajectoryFacts()
            e._apply({'calls': {'total': v}}, f)
            self.assertIsNone(f.calls.total)
        f = TrajectoryFacts()
        e._apply({'requested': {'clip_durations': [-1]},
                  'claims': {'declared_clips_per_episode': [1.5]},
                  'errors': {'n_attempts': 2, 'n_unrecovered': 3}}, f)
        self.assertIsNone(f.requested.clip_durations)
        self.assertIsNone(f.claims.declared_clips_per_episode)
        self.assertIsNone(f.errors.n_unrecovered)

    def test_self_checks_are_supported_and_do_not_convert_string_false(self):
        body = '### SOURCE: TRAJECTORY.md\naudioOk=false durationOk=true'
        p = reply('claims.self_checks', {'audioOk': False, 'durationOk': True}, 'audioOk=false durationOk=true')
        f = coordinate([body], [p], ['claims.self_checks'])
        self.assertEqual(f.claims.self_checks, {'audioOk': False, 'durationOk': True})
        self.assertIn('self_checks', e._SCHEMA_HINT)

    def test_audit_roundtrip_does_not_change_coverage(self):
        f = coordinate(['### SOURCE: TRAJECTORY.md\n3 calls'],
                       [reply('calls.total', 3, '3 calls')], ['calls.total'])
        restored = TrajectoryFacts.from_dict(f.to_dict())
        self.assertEqual(restored.to_dict(), f.to_dict())
        self.assertEqual(f.observable_fields, 1)

    def test_chunk_headers_and_budget_survive_splitting(self):
        chunks = e._pack(['### SOURCE: a.md\n' + 'abcdefghij\n' * 30], 80)
        self.assertTrue(all(len(c) <= 80 for c in chunks))
        self.assertTrue(all(c.startswith('### SOURCE: a.md\n') for c in chunks))
        self.assertEqual(sum(c.count('abcdefghij') for c in chunks), 30)


class TestRealInputs(unittest.TestCase):
    def prepared(self, path):
        from stage1_inputs.preprocess import InputPreprocessor
        cleaner = InputPreprocessor()
        part = cleaner.prepare(path.name, path.read_text(encoding='utf-8'))
        audit = cleaner.finalize([part])
        self.assertEqual(audit['validation']['status'], 'passed')
        return part.partition('\n')[2]

    def root(self, harness, layouts=('harness',)):
        # 'harness' is the layout of the original working corpus, which the two
        # codex tests below name specific records of. 's_harness_data' is the
        # layout of the published dataset.
        for layout in layouts:
            paths = sorted((REPO / 'data' / layout / harness).glob('*/prompt_*'))
            if paths:
                return paths[0]
        self.skipTest('real trajectory corpus not installed')

    def test_codex_cost_records_keep_generation_kind_and_job_id(self):
        p = self.root('codex') / 'costs.skill.jsonl'
        if not p.exists():
            self.skipTest('fixture not installed')
        records = [json.loads(s) for s in self.prepared(p).splitlines()]
        self.assertEqual(len(records), 3)
        self.assertTrue(all(r['kind'] == 'video' and r['id'] for r in records))

    def test_hermes_long_records_are_valid_and_keep_their_tails(self):
        p = next(self.root('hermes', ('harness', 's_harness_data')).rglob('events.jsonl'), None)
        if p is None:
            self.skipTest('fixture not installed')
        original = [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
        distilled = [json.loads(s) for s in self.prepared(p).splitlines()]
        self.assertTrue(any(len(json.dumps(r)) > 4000 for r in original))
        self.assertEqual(distilled, original)

    def test_actual_codex_duration_evidence_reaches_merged_checklist(self):
        root = self.root('codex')
        facts = TrajectoryFacts()
        parts = e._lm_parts(root, facts)
        quote = 'Real skill-generated clips: 3 × 10 seconds'
        self.assertTrue(any(quote in s for s in parts))
        f = coordinate(parts, [reply('requested.generated_seconds', 30, quote)],
                       ['requested.generated_seconds'])
        self.assertEqual(f.requested.generated_seconds, 30)
        self.assertEqual(f.requested.generated_seconds_source, 'stated')


if __name__ == '__main__':
    unittest.main()
