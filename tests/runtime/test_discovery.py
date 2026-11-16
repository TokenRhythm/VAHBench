"""find_runs across every directory shape data/ has actually been in.

No ffprobe, no videos: discovery of *runs* only needs directory names and run
records. The 2026-09-15 merge with VAHBench put the model level above the case
(`<group>/<agent>/<model>/<prompt>/`) while openmontage stayed on the old
`<agent>/<prompt>/<model>/`; a walker that assumed one shape reported the
merged tree as 18 runs.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


from stage2_delivery.discover import find_runs  # noqa: E402


def mk(root: Path, rel: str, files: dict[str, str] | None = None) -> Path:
    d = root / rel
    d.mkdir(parents=True, exist_ok=True)
    for name, body in (files or {}).items():
        (d / name).write_text(body, encoding="utf-8")
    return d


RUN = {"run.json": json.dumps({"status": "ok", "target": {"episodes": 3}})}


class TestLayouts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        r = Path(self.tmp.name)
        # new shape: <group>/<agent>/<model>/<case>/
        mk(r, "agents/univa/deepseek-flash/prompt_01", RUN)
        mk(r, "agents/univa/deepseek-flash/prompt_03", RUN)
        mk(r, "harness/codex/gpt6/prompt_01", {"verification.json": "{}"})
        # a VAHBench-style case with no run record at all, plus a hidden dir
        mk(r, "harness/hermes/glm/prompt_01/receipts_series/final")
        mk(r, "harness/hermes/glm/prompt_01/.hermes/trajectory-exports")
        # empty placeholder model dirs (dsh)
        mk(r, "harness/dsh/gemini", {".gitkeep": ""})
        # old shape: <agent>/<case>/<model>/, still coexisting
        mk(r, "openmontage/prompt_01/deepseek-flash", RUN)
        mk(r, "openmontage/prompt_01/claude-opus", RUN)
        # old shape with a variant level: <agent>/<variant>/<case>/<model>/
        mk(r, "openmontage/codex/prompt_05/deepseek-flash", RUN)
        # a case that IS the run and keeps its output in `artifacts/`. That
        # folder holds a nested `artifacts/`, so `_is_run_dir` says yes and the
        # old-shape branch used to emit it as the run -- one level below the
        # trajectory, which then read as missing on 15 real runs.
        mk(r, "harness/claudecode/openmontage__flash/prompt_07",
           {**RUN, "trace.jsonl": "", "stdout.log": ""})
        mk(r, "harness/claudecode/openmontage__flash/prompt_07/artifacts/artifacts")
        # things that must never be entered
        mk(r, "_runner/adapters", {"x.py": ""})
        mk(r, "_case_conflicts/harness/hermes/glm/prompt_03", RUN)
        mk(r, "harness/claudecode/_shared/probe_out", {"a.txt": ""})
        self.runs = find_runs(r)
        self.by_rel = {x.rel: x for x in self.runs}

    def tearDown(self):
        self.tmp.cleanup()

    def test_every_case_found_once(self):
        self.assertEqual(sorted(self.by_rel), [
            "agents/univa/deepseek-flash/prompt_01",
            "agents/univa/deepseek-flash/prompt_03",
            "harness/claudecode/openmontage__flash/prompt_07",
            "harness/codex/gpt6/prompt_01",
            "harness/hermes/glm/prompt_01",
            "openmontage/codex/prompt_05/deepseek-flash",
            "openmontage/prompt_01/claude-opus",
            "openmontage/prompt_01/deepseek-flash",
        ])

    def test_a_case_holding_its_own_record_is_the_run(self):
        """The run root has to be where the trajectory is.

        `prompt_07/` holds run.json, trace.jsonl and stdout.log; `artifacts/` is
        its output. Emitting `artifacts/` as the run put the root one level
        below the log, and every trajectory reader looks at the root -- so the
        whole trajectory layer abstained on runs whose log was right there. 15
        of the 80 runs in `data/` were in this shape and none of them could be
        scored on 2.1.a until the walker stopped descending.
        """
        r = self.by_rel["harness/claudecode/openmontage__flash/prompt_07"]
        self.assertEqual(r.root.name, "prompt_07")
        self.assertTrue((r.root / "trace.jsonl").is_file())
        self.assertEqual((r.harness, r.model), ("claudecode", "openmontage__flash"))
        self.assertNotIn("harness/claudecode/openmontage__flash/prompt_07/artifacts",
                         self.by_rel)

    def test_new_shape_labels(self):
        r = self.by_rel["agents/univa/deepseek-flash/prompt_01"]
        self.assertEqual((r.group, r.harness, r.model, r.case),
                         ("agents", "univa", "deepseek-flash", "prompt_01"))
        self.assertEqual(r.root.name, "prompt_01")

    def test_old_shape_labels(self):
        r = self.by_rel["openmontage/prompt_01/claude-opus"]
        self.assertEqual((r.group, r.harness, r.model), ("", "openmontage", "claude-opus"))
        self.assertEqual(r.root.name, "claude-opus")

    def test_old_shape_with_variant_level(self):
        r = self.by_rel["openmontage/codex/prompt_05/deepseek-flash"]
        self.assertEqual(r.harness, "openmontage")
        self.assertEqual(r.model, "codex/deepseek-flash")

    def test_case_without_run_record_is_still_a_run(self):
        """VAHBench cases carry no run.json; the directory is the run."""
        r = self.by_rel["harness/hermes/glm/prompt_01"]
        self.assertEqual((r.harness, r.model), ("hermes", "glm"))
        self.assertIsNone(r.run_json)

    def test_hidden_and_underscore_dirs_are_never_entered(self):
        for rel in self.by_rel:
            self.assertNotIn("_case_conflicts", rel)
            self.assertNotIn("_runner", rel)
            self.assertNotIn("_shared", rel)
            self.assertNotIn(".hermes", rel)

    def test_empty_model_dir_yields_nothing(self):
        self.assertFalse(any("dsh" in rel for rel in self.by_rel))

    def test_run_id_is_the_relative_path(self):
        r = self.by_rel["harness/codex/gpt6/prompt_01"]
        self.assertEqual(r.run_id, "harness/codex/gpt6/prompt_01")

    def test_record_less_case_inherits_the_prompts_target(self):
        """The brief belongs to the prompt. hermes' prompt_01 has no run
        record; univa's and openmontage's prompt_01 both say 3 episodes."""
        r = self.by_rel["harness/hermes/glm/prompt_01"]
        self.assertEqual(r.target.episodes, 3)
        self.assertEqual(r.target.source, "sibling_prompt")

    def test_conflicting_sibling_targets_are_not_inherited(self):
        with tempfile.TemporaryDirectory() as t:
            r = Path(t)
            mk(r, "a/m/prompt_02", {"run.json": json.dumps({"target": {"episodes": 3}})})
            mk(r, "b/m/prompt_02", {"run.json": json.dumps({"target": {"episodes": 5}})})
            mk(r, "c/m/prompt_02")
            orphan = [x for x in find_runs(r) if x.harness == "c"][0]
        self.assertFalse(orphan.target.known)

    def test_harness_filter_matches_harness_or_group(self):
        r = Path(self.tmp.name)
        self.assertEqual({x.harness for x in find_runs(r, {"univa"})}, {"univa"})
        self.assertTrue(all(x.group == "harness" for x in find_runs(r, {"harness"})))


if __name__ == "__main__":
    unittest.main(verbosity=2)
