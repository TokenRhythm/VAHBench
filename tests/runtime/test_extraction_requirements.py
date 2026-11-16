"""The model is the second half of checklist extraction, not an option.

Rules first, the model on whatever is still None, and no mode in which the
second step is quietly skipped: with gaps and no model reachable the sweep
refuses to start, a run that reaches extraction in that state errors rather
than reporting a rules-only checklist as complete, and every report says what
the model was asked for and what it still could not read.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))

import run as run_eval                                                        # noqa: E402
from stage2_delivery.discover import Run
from stage2_delivery.discover import Target                               # noqa: E402
from stage1_inputs.schema import ConditioningUse
from stage1_inputs.schema import ProducedArtifact
from stage1_inputs.schema import TrajectoryFacts
from tests.inputs.test_extraction import TRACE
from tests.inputs.test_extraction import write_run                           # noqa: E402


def _run(root: Path) -> Run:
    return Run(root=root, harness="h", case="prompt_01", model="m",
               target=Target(), rel=str(root))


def _opts(call_log):
    return dict(model="m", provider="openrouter", max_chars=1000, max_tokens=10,
                call_log=call_log)


class TestGatherFacts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        # An HTTP trace: exact on counts, and by construction unable to supply
        # produced / conditioning / prompts -- the gaps the model exists for.
        self.root = write_run(Path(self.tmp.name), trace=TRACE)
        self.run = _run(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_gaps_with_no_model_reachable_is_an_error_not_a_quiet_score(self):
        with mock.patch.object(run_eval.llm, "make_call_fn", return_value=None):
            with self.assertRaises(RuntimeError) as cm:
                run_eval.gather_facts(self.run, **_opts(run_eval.llm.CallLog()))
        self.assertIn("conditioning", str(cm.exception))

    def test_no_gaps_means_no_model_and_says_so(self):
        from tests.inputs.test_extraction import TestNeedsLM
        full = TestNeedsLM._complete()        # every field the tree reads
        with mock.patch.object(run_eval.llm, "make_call_fn") as mk:
            facts, lm = run_eval.gather_facts(self.run, base_facts=full,
                                              **_opts(run_eval.llm.CallLog()))
        mk.assert_not_called()
        self.assertIsNone(lm)
        self.assertIn("规则抽取已填满清单，未调用 LM", facts.notes)

    def test_model_fills_only_the_gaps_and_the_report_shows_before_and_after(self):
        lm_facts = TrajectoryFacts(backend="lm:m@r1")
        lm_facts.produced = [ProducedArtifact(path="clip_e1s1.mp4")]
        lm_facts.conditioning = [ConditioningUse(references=["maya.png"])]
        lm_facts.errors.n_attempts = 99              # injected extraction fills the missing field
        with mock.patch.object(run_eval.llm, "make_call_fn", return_value=lambda p: ""), \
                mock.patch.object(run_eval.extractors, "lm_extract", return_value=lm_facts):
            facts, lm = run_eval.gather_facts(self.run, **_opts(run_eval.llm.CallLog()))
        self.assertIs(lm, lm_facts)
        self.assertEqual(facts.errors.n_attempts, 99)
        self.assertEqual(len(facts.produced), 1)                  # the model filled
        self.assertEqual(facts.backend, "deterministic+lm:m@r1")
        before = next(n for n in facts.notes if n.startswith("LM 补填前缺口"))
        after = next(n for n in facts.notes if n.startswith("LM 补填后仍缺"))
        self.assertIn("conditioning", before)
        self.assertIn("errors.n_attempts", before)
        self.assertNotIn("errors.n_attempts", after)
        self.assertNotIn("conditioning", after)                  # it got filled
        self.assertIn("requested.prompts", after)                # it did not

    def test_discover_only_skips_the_model_and_names_the_gaps(self):
        with mock.patch.object(run_eval.llm, "make_call_fn") as mk:
            facts, lm = run_eval.gather_facts(self.run, discover_only=True,
                                              **_opts(run_eval.llm.CallLog()))
        mk.assert_not_called()
        self.assertIsNone(lm)
        self.assertTrue(any(n.startswith("--discover-only：未调用 LM") and "conditioning" in n
                            for n in facts.notes))

    def test_precomputed_facts_are_reused_not_recomputed(self):
        pre = TrajectoryFacts(backend="deterministic")
        pre.errors.n_attempts = 7
        with mock.patch.object(run_eval.extractors, "deterministic_extract") as de, \
                mock.patch.object(run_eval.llm, "make_call_fn", return_value=None):
            with self.assertRaises(RuntimeError):
                run_eval.gather_facts(self.run, base_facts=pre,
                                      **_opts(run_eval.llm.CallLog()))
        de.assert_not_called()


class TestLmGate(unittest.TestCase):
    def test_ready_model_never_refuses(self):
        self.assertIsNone(run_eval.lm_gate({"a": ["conditioning"], "b": []}, True))

    def test_no_gaps_proceeds_without_a_model(self):
        self.assertIsNone(run_eval.lm_gate({"a": [], "b": []}, False))

    def test_gaps_without_a_model_refuse_and_name_the_runs(self):
        msg = run_eval.lm_gate({"h/prompt_01": ["conditioning", "requested.prompts"],
                                "h/prompt_03": []}, False)
        self.assertIsNotNone(msg)
        self.assertIn("1/2", msg)
        self.assertIn("h/prompt_01: conditioning、requested.prompts", msg)
        self.assertNotIn("h/prompt_03", msg)
        self.assertIn("不是可选项", msg)


if __name__ == "__main__":
    unittest.main()
