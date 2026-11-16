"""Pipeline integration and invariant failures, without a remote model."""

from shared.paths import REPO_ROOT
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import stage1_inputs.extractors as extractors
import stage1_inputs.client as llm
from stage2_delivery.discover import Run
from stage2_delivery.discover import Target
from stage1_inputs.contract import policy_hash
from stage1_inputs.contract import validate_preparation
from stage1_inputs.preprocess import InputPreprocessor
from stage1_inputs.schema import TrajectoryFacts
from run import gather_facts


class TestContract(unittest.TestCase):
    def prepared(self):
        cleaner = InputPreprocessor()
        text = json.dumps({'type': 'system', 'subtype': 'thinking_tokens', 'count': 3}) + '\n'
        text += '{"type":"tool.result","id":"a","error":"bad reference"}\n'
        parts = [cleaner.prepare('events.jsonl', text)]
        parts.append(cleaner.prepare('copy.jsonl', text))
        return cleaner.finalize(parts), parts

    def test_reproducible_manifest_and_policy_identity(self):
        first, parts = self.prepared()
        second, again = self.prepared()
        self.assertEqual(first, second)
        self.assertEqual(parts, again)
        self.assertEqual(first['validation']['status'], 'passed')
        self.assertEqual(first['policy_sha256'], policy_hash())
        self.assertEqual(first['rule_hits']['P001'], 1)
        self.assertEqual(first['rule_hits']['P003'], 1)

    def test_changed_output_missing_line_and_broken_alias_are_detected(self):
        audit, parts = self.prepared()
        changed = list(parts)
        changed[0] = changed[0].replace('bad reference', 'success')
        self.assertIn('output_hashes_and_sources', validate_preparation(audit, changed)['errors'])
        broken = copy.deepcopy(audit)
        broken['files'][0]['output_lines'] = []
        self.assertIn('source_line_coverage_and_order', validate_preparation(broken, parts)['errors'])
        broken = copy.deepcopy(audit)
        broken['duplicate_sources']['copy.jsonl'] = 'missing.jsonl'
        self.assertIn('document_references', validate_preparation(broken, parts)['errors'])

    def test_dangling_history_reference_is_detected(self):
        cleaner = InputPreprocessor()
        row = {'type': 'llm_call', 'input': {'messages': [{'content': 'original contract'}]}}
        parts = [cleaner.prepare('events.jsonl', json.dumps(row) + '\n' + json.dumps(row))]
        audit = cleaner.finalize(parts)
        broken = copy.deepcopy(audit)
        broken['history_definitions'] = []
        self.assertIn('history_and_context_references', validate_preparation(broken, parts)['errors'])

    def test_malformed_source_is_preserved_not_rejected(self):
        cleaner = InputPreprocessor()
        text = '{broken record\nFINAL: failed\n'
        parts = [cleaner.prepare('stdout.log', text)]
        audit = cleaner.finalize(parts)
        self.assertEqual(audit['validation']['status'], 'passed')
        self.assertEqual(audit['rule_hits']['P006'], 1)
        self.assertIn(text, parts[0])

    def test_standard_schema_is_shipped_with_manifest_fields(self):
        schema = json.loads((REPO_ROOT / 'stage1_inputs/schemas/preprocessing-audit.schema.json').read_text())
        audit, _ = self.prepared()
        self.assertEqual(schema['$schema'], 'https://json-schema.org/draft/2020-12/schema')
        self.assertEqual(set(audit), set(schema['required']))
        self.assertEqual(set(audit['files'][0]), set(schema['$defs']['file']['required']))

    def test_benchmark_gather_facts_automatically_prepares_and_records_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text = json.dumps({'type': 'system', 'subtype': 'thinking_tokens', 'count': 123}) + '\n'
            text += json.dumps({'type': 'assistant', 'content': 'FINAL delivered ep1.mp4'}) + '\n'
            path = root / 'stdout.log'
            path.write_text(text, encoding='utf-8')
            raw_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            prompts = []
            def call(prompt):
                prompts.append(prompt)
                return json.dumps({'statements': [{'field': 'claims.declared_deliverables',
                    'stage': 'delivery', 'subject': None, 'value': ['ep1.mp4'],
                    'evidence': [{'source': 'stdout.log', 'quote': 'FINAL delivered ep1.mp4'}]}]})
            run = Run(root, 'test', 'case', None, Target())
            with patch.object(llm, 'make_call_fn', return_value=call):
                facts, _ = gather_facts(run, model='test', provider='test', max_chars=120000,
                    max_tokens=1000, call_log=llm.CallLog(), base_facts=TrajectoryFacts())
            self.assertEqual(len(prompts), 1)
            self.assertNotIn('thinking_tokens', prompts[0])
            self.assertIn('FINAL delivered ep1.mp4', prompts[0])
            self.assertEqual(facts.claims.declared_deliverables, ['ep1.mp4'])
            audit = facts.lm_audit['preprocessing']
            self.assertEqual(audit['validation']['status'], 'passed')
            self.assertEqual(audit['rule_hits']['P001'], 1)
            self.assertEqual(audit['files'][0]['raw_sha256'], raw_hash)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), raw_hash)
            self.assertEqual(TrajectoryFacts.from_dict(facts.to_dict()).lm_audit, facts.lm_audit)

    def test_invalid_utf8_is_visible_not_silently_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'notes.txt').write_bytes(b'FINAL \xff failed')
            f = TrajectoryFacts()
            parts = extractors._lm_parts(root, f)
            self.assertIn('\ufffd', parts[0])
            self.assertEqual(f.lm_audit['preprocessing']['files'][0]['decoding'], 'utf-8-with-replacement')


if __name__ == '__main__':
    unittest.main()
