"""Declarations survive storage; no truth/quality assessment occurs here."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import stage1_inputs.extractors as extractors
from stage1_inputs.schema import TrajectoryFacts
from run import _merge_facts
from stage1_inputs.statements import collect_statements
from stage1_inputs.statements import document_statements


def statement(field, value, stage, subject=None):
    return dict(field=field, value=value, stage=stage, subject=subject,
                evidence=[dict(source="trajectory.md", quote="the original declaration")])


def extract(rows):
    with patch.object(extractors, '_lm_parts', return_value=[
            '### SOURCE: trajectory.md\nthe original declaration']):
        return extractors.lm_extract(Path('.'), model='test', revision='test',
            focus_fields=list(dict.fromkeys(r['field'] for r in rows)),
            call_fn=lambda _: json.dumps({'statements': rows}))


class StatementExtractionTests(unittest.TestCase):
    def test_single_asset_and_inventory_statements_merge_without_losing_files(self):
        one = {'path': 'a.mp4', 'kind': 'video'}
        two = {'path': 'b.png', 'kind': 'image'}
        rows = [statement('produced', one, 'generation', 'a.mp4'),
                statement('produced', [one, two], 'generation')]
        f = extract(rows)
        self.assertEqual([p.path for p in f.produced], ['a.mp4', 'b.png'])
        self.assertEqual(f.statements[0]['value'], one)
        self.assertEqual(f.statements[1]['value'], [one, two])

    def test_single_reference_use_and_inventory_merge_without_losing_targets(self):
        one = {'target': 'a.mp4', 'references': ['maya.png'], 'kind': 'character'}
        two = {'target': 'b.mp4', 'references': ['room.png'], 'kind': 'scene'}
        f = extract([statement('conditioning', one, 'request', 'a.mp4'),
                     statement('conditioning', [one, two], 'request')])
        self.assertEqual([c.target for c in f.conditioning], ['a.mp4', 'b.mp4'])
        self.assertEqual(f.conditioning[0].references, ['maya.png'])

    def test_unmatched_model_quote_is_retained_in_audit_but_not_scored(self):
        bad = statement('claims.self_reported_status', 'ok', 'delivery')
        bad['evidence'][0]['source'] = 'another.md'
        f = extract([bad])
        self.assertEqual(f.statements, [])
        self.assertIsNone(f.claims.self_reported_status)
        self.assertEqual(f.lm_audit['unverified_statements'][0]['statement']['value'], 'ok')
        self.assertEqual(f.lm_audit['chunks_valid'], 1)

    def test_invented_or_empty_quote_cannot_fill_global_count(self):
        for quote in ('invented text', ''):
            bad = statement('errors.n_unrecovered', 0, 'generation')
            bad['evidence'][0]['quote'] = quote
            f = extract([bad])
            self.assertIsNone(f.errors.n_unrecovered)
            self.assertEqual(len(f.lm_audit['unverified_statements']), 1)

    def test_escaped_json_quote_and_whitespace_are_traceable(self):
        row = statement('calls.total', 2, 'generation')
        row['evidence'][0]['quote'] = 'two\n calls'
        f = TrajectoryFacts()
        collect_statements({'statements': [row]}, ['calls.total'], f, 1,
                           body='### SOURCE: trajectory.md\n{"text":"two\\n calls"}')
        self.assertEqual(len(f.statements), 1)

    def test_document_fragments_preserve_all_characters_and_source_offsets(self):
        for text in ('# Characters\n' + '角色衣着、道具与场景。\n' * 1300 + 'FINAL DETAIL',
                     'x' * 18003, 'a\n\n  b\n'):
            rows = document_statements('continuity.text', 'story/bible.md', text)
            self.assertEqual(''.join(r['value'] for r in rows), text)
            for i, row in enumerate(rows, 1):
                self.assertEqual(row['chunk'], i)
                self.assertLessEqual(len(row['value']), 8000)
                ref = row['evidence'][0]
                self.assertEqual(ref['source'], 'story/bible.md')
                self.assertEqual(text[ref['char_start']:ref['char_end']], row['value'])
            f = TrajectoryFacts(statements=rows)
            self.assertEqual(TrajectoryFacts.from_dict(f.to_dict()).statements, rows)

    def test_multiple_continuity_passages_remain_in_compatibility_view(self):
        f = extract([statement('continuity.text', 'Maya wears navy.', 'planning', 'ep1'),
                     statement('continuity.text', 'The pencil stays silver.', 'planning', 'ep2')])
        self.assertIn('Maya wears navy.', f.continuity.text)
        self.assertIn('The pencil stays silver.', f.continuity.text)
        self.assertEqual([r['subject'] for r in f.statements], ['ep1', 'ep2'])

    def test_error_statements_fill_checklist_gaps(self):
        f = extract([statement('errors.n_attempts', 3, 'generation'),
                     statement('errors.n_unrecovered', 1, 'generation')])
        merged = _merge_facts(TrajectoryFacts(), f)
        self.assertEqual(merged.errors.n_attempts, 3)
        self.assertEqual(merged.errors.n_unrecovered, 1)
        self.assertEqual(len(merged.statements), 2)
        self.assertIn('errors.n_attempts', merged.lm_evidence)

    def test_related_known_field_is_saved_even_if_not_in_gaps(self):
        f = TrajectoryFacts()
        collect_statements({'statements': [statement('calls.video_gen', 3, 'generation')]},
                           ['claims.self_reported_status'], f, 1)
        self.assertEqual(f.statements[0]['value'], 3)

    def test_progress_and_delivery_are_both_saved(self):
        f = extract([statement('claims.self_reported_status', 'partial', 'generation'),
                     statement('claims.self_reported_status', 'ok', 'delivery')])
        self.assertEqual([r['value'] for r in f.statements], ['partial', 'ok'])
        self.assertEqual(f.claims.self_reported_status, 'ok')
        self.assertEqual(f.lm_audit['conflicts'], [])

    def test_competing_declarations_never_disappear(self):
        f = extract([statement('claims.self_reported_status', 'partial', 'delivery'),
                     statement('claims.self_reported_status', 'ok', 'delivery')])
        self.assertEqual(len(f.statements), 2)
        self.assertIsNone(f.claims.self_reported_status)
        restored = TrajectoryFacts.from_dict(f.to_dict())
        self.assertEqual(restored.to_dict(), f.to_dict())

    def test_missing_files_are_still_saved_as_declared(self):
        f = extract([statement('claims.declared_deliverables', ['missing/*.mp4'], 'delivery')])
        self.assertEqual(f.claims.declared_deliverables, ['missing/*.mp4'])
        self.assertEqual(f.statements[0]['value'], ['missing/*.mp4'])

    def test_checked_is_not_passed_and_per_file_false_is_retained(self):
        f = extract([statement('claims.self_checks', {'checked': ['audio']}, 'verification'),
                     statement('claims.self_checks', {'audioOk': False}, 'verification', 'ep1.mp4')])
        self.assertIsNone(f.claims.self_checks)
        self.assertEqual(f.statements[0]['value'], {'checked': ['audio']})
        self.assertEqual(f.statements[1]['subject'], 'ep1.mp4')
        self.assertIs(f.statements[1]['value']['audioOk'], False)

    def test_early_limitations_are_not_erased_or_promoted_to_delivery(self):
        f = extract([statement('claims.deviations', ['only drafts so far'], 'generation'),
                     statement('claims.deviations', ['TTS dialogue'], 'delivery')])
        self.assertEqual(len(f.statements), 2)
        self.assertEqual(f.claims.deviations, ['TTS dialogue'])

    def test_merge_keeps_statement_even_if_rule_slot_is_already_filled(self):
        base = TrajectoryFacts()
        base.claims.self_reported_status = 'partial'
        lm = extract([statement('claims.self_reported_status', 'ok', 'delivery')])
        f = _merge_facts(base, copy.deepcopy(lm))
        self.assertEqual(f.claims.self_reported_status, 'partial')
        self.assertEqual(f.statements[0]['value'], 'ok')

    def test_partial_slice_counts_are_saved_without_summing(self):
        rows = [statement('calls.total', 3, 'generation', 'episode 1'),
                statement('calls.total', 4, 'generation', 'episode 2')]
        f = extract(rows)
        self.assertEqual([r['value'] for r in f.statements], [3, 4])
        self.assertIsNone(f.calls.total)
        single = extract(rows[:1])
        self.assertIsNone(single.calls.total)
        self.assertEqual(single.statements[0]['value'], 3)

    def test_no_statement_is_not_a_failure_or_a_zero(self):
        f = extract([])
        self.assertEqual(f.statements, [])
        self.assertIsNone(f.calls.total)
