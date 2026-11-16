"""Contract coverage and scoped serialization; model accuracy uses live audits."""
import json
from pathlib import Path
import sys
import unittest

from stage1_inputs.prompt import FIELD_CONTRACTS
from stage1_inputs.prompt import build_prompt
from stage1_inputs.schema import CONSUMERS
from stage1_inputs.schema import TrajectoryFacts
from stage1_inputs.schema import _flatten


class TestFieldContracts(unittest.TestCase):
    def test_every_checklist_field_has_a_contract(self):
        self.assertEqual(set(FIELD_CONTRACTS), {k for k, _ in _flatten(TrajectoryFacts())})
        self.assertEqual(set(FIELD_CONTRACTS), set(CONSUMERS))
        for key, (kind, purpose, evidence) in FIELD_CONTRACTS.items():
            self.assertTrue(kind and purpose and evidence, key)

    def test_template_contains_only_requested_model_owned_fields(self):
        prompt = build_prompt('### SOURCE: a.md\nhello', 1, 1,
                              ['claims.self_checks', 'produced', 'requested.generated_seconds_source'])
        serialized = prompt.split('OUTPUT TEMPLATE:\n')[1].split('\n\nEXECUTION EVIDENCE:')[0]
        self.assertEqual(json.loads(serialized), {'statements': []})
        requested = prompt.split('REQUESTED FIELDS:\n')[1].split('\n\nOUTPUT TEMPLATE:')[0]
        self.assertIn('claims.self_checks:', requested)
        self.assertIn('produced:', requested)
        self.assertNotIn('requested.generated_seconds_source:', requested)
        self.assertNotIn('calls.total:', requested)
        self.assertIn('Do not evaluate truth', prompt)

    def test_unknown_field_fails_instead_of_silently_missing_instructions(self):
        with self.assertRaises(ValueError):
            build_prompt('', 1, 1, ['new.unexplained_field'])

    def test_draft_is_valid_json_and_evidence_is_preserved(self):
        base = TrajectoryFacts()
        base.continuity.text = 'quoted "text"\n' * 3000
        base.requested.prompts = ['long prompt' * 1000] * 10
        evidence = '### SOURCE: odd.md\n"No guesses"\n## heading\nend'
        prompt = build_prompt(evidence, 1, 1, ['calls.total'], base)
        draft = prompt.split('DETERMINISTIC DRAFT (context, NOT evidence to cite):\n')[1].split('\n\nOUTPUT TEMPLATE')[0]
        parsed = json.loads(draft)
        self.assertEqual(parsed['requested']['prompts']['count'], 10)
        self.assertLess(len(draft), 5000)
        self.assertIn(evidence, prompt)


if __name__ == '__main__':
    unittest.main()
