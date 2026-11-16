"""The benchmark's only inputs are a trajectory and a video.

Everything else is derived, and this file pins the derivations that used to
be asked of the user or of the directory layout:

  * a run is found under any directory name, and --data may be the run;
  * the brief is matched from the text the trajectory embeds when the case
    name says nothing, and it fills a target no record supplied;
  * module interpreters are resolved by an import probe, with an
    environment override, instead of two hard-coded paths;
  * the plain visual pass and the physical pass run on the same
    splice-derived shot boundaries as the seeded pass.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


import stage1_inputs.brief as brief
import stage4_checks.dispatch as runners                                    # noqa: E402
from stage1_inputs.brief import BriefFacts
from stage1_inputs.brief import match_brief
from stage1_inputs.brief import apply_brief_targets  # noqa: E402
from stage2_delivery.discover import Target
from stage2_delivery.discover import find_runs                        # noqa: E402
from stage3_evidence.collector import EvidenceBus                              # noqa: E402

PROMPT_MD = """# Prompts

### Prompt 1: The Woman Behind the Numbers

[Basic Specifications]
- **Format**: Multi-episode series (3 episodes)
- **Single-episode duration**: 60 seconds
- **Aspect ratio**: 9:16 (vertical screen)
- **Characters**: Four principal characters. Maya Chen, 29, an analyst; Victor Hale, 41, a department head; Lena Ortiz, 31, a colleague; and Evelyn Shaw, 52, the CFO.
- **Setting**: An office, including an open-plan workspace, glass conference room, elevator lobby, and executive suite.
- **Dialogue Norms**: Keep dialogue below 125 spoken words per episode. No explanatory voiceover.
- **Timing Allocation**: For each episode, use 0–3 seconds for the hook, 4–18 seconds for conflict setup, 19–38 seconds for escalation, 39–53 seconds for the reversal, and 54–60 seconds for the payoff.
- **Final Output Checklist**: Deliver all three finished episodes, a series cover frame, episode titles.

### Prompt 2: The Parcel That Kept Returning

- **Format**: Multi-episode series (3 episodes)
- **Single-episode duration**: 60 seconds
- **Aspect ratio**: 9:16 (vertical screen)
- **Characters**: Five recurring characters. Noah Reed, 24, a courier; Mrs. Bell, 68, a seamstress; Jay, 10, her grandson; Omar Singh, 36, the caretaker; and M. Bell, the unknown sender.
- **Setting**: An apartment block, using the entrance desk, corridor, elevator, and kitchen.
- **Dialogue Norms**: Keep dialogue below 110 spoken words per episode.
- **Timing Allocation**: For each episode, use 0–3 seconds for the hook, 4–18 seconds for conflict setup, 19–38 seconds for escalation, 39–53 seconds for the reversal, and 54–60 seconds for the payoff.
- **Final Output Checklist**: Deliver all three finished episodes.
"""


def _body(prompt_no: int) -> str:
    return brief.parse_prompts(PROMPT_MD)[prompt_no].raw and PROMPT_MD.split(
        f"### Prompt {prompt_no}:")[1].split("### Prompt")[0]


class TestRunsUnderAnyName(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.r = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _mk(self, rel, files=None):
        d = self.r / rel
        d.mkdir(parents=True, exist_ok=True)
        for name, body in (files or {}).items():
            (d / name).write_bytes(body if isinstance(body, bytes) else body.encode("utf-8"))
        return d

    def test_a_trajectory_under_any_name_is_a_run(self):
        self._mk("drops/my_agent_take3", {"trace.jsonl": "{}\n", "final.mp4": b"x"})
        self._mk("drops/just_a_video", {"result.mp4": b"x"})
        self._mk("drops/just_a_log", {"events.jsonl": "{}\n"})
        self._mk("drops/nothing_here", {"notes.txt": "hi"})
        runs = {x.rel: x for x in find_runs(self.r)}
        self.assertEqual(set(runs), {"drops/my_agent_take3", "drops/just_a_video",
                                     "drops/just_a_log"})
        self.assertEqual(runs["drops/my_agent_take3"].case, "my_agent_take3")
        self.assertEqual(runs["drops/my_agent_take3"].harness, "drops")
        self.assertFalse(runs["drops/just_a_video"].target.known)   # never guessed

    def test_a_run_with_a_hidden_trajectory_and_subfolders_is_one_run(self):
        """hermes: everything under `.hermes/` (hidden) and `short_drama/`;
        nothing sits directly in the run directory. Its clips/ and output/
        folders hold videos and must not become runs of their own."""
        self._mk("drop/take3/.hermes/trajectory-exports/trajectory", {"events.jsonl": "{}\n"})
        self._mk("drop/take3/short_drama/clips/ep1", {"clip01.mp4": b"x"})
        self._mk("drop/take3/short_drama/output", {"ep1_stitched.mp4": b"x"})
        runs = find_runs(self.r)
        self.assertEqual([x.rel for x in runs], ["drop/take3"])
        self.assertEqual(runs[0].case, "take3")

    def test_data_may_point_at_the_run_itself(self):
        d = self._mk("one_run", {"trace.jsonl": "{}\n", "ep1.mp4": b"x"})
        runs = find_runs(d)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].root, d)
        self.assertEqual(runs[0].run_id, "one_run")        # not "."

    def test_prompt_layouts_are_unchanged(self):
        """The old and new dataset shapes still resolve to the same runs; a
        prompt directory's subfolders are never mistaken for runs."""
        self._mk("harness/x/m/prompt_01/output", {"ep1.mp4": b"x"})
        self._mk("harness/x/m/prompt_01", {"run.json": json.dumps({"target": {"episodes": 3}})})
        self._mk("old/prompt_03/model-a", {"run.json": "{}"})
        rels = sorted(x.rel for x in find_runs(self.r))
        self.assertEqual(rels, ["harness/x/m/prompt_01", "old/prompt_03/model-a"])


class TestBriefFromTheRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.r = Path(self.tmp.name)
        self.briefs = brief.parse_prompts(PROMPT_MD)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, name, files):
        d = self.r / name
        d.mkdir(parents=True, exist_ok=True)
        for n, body in files.items():
            p = d / n
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body, encoding="utf-8")
        return SimpleNamespace(case=name, root=d, run_id=name,
                               target=Target())

    def test_case_name_wins_when_it_names_a_prompt(self):
        run = self._run("prompt_02_anything", {})
        bf, how = match_brief(self.briefs, run)
        self.assertEqual((bf.prompt_no, how), (2, "case_name"))

    def test_json_escaped_brief_in_the_trajectory_is_parsed(self):
        body = _body(1)
        escaped = json.dumps({"role": "user", "content": body})
        run = self._run("take7", {".agent/exports/prompts.json": escaped})
        bf, how = match_brief(self.briefs, run)
        self.assertEqual(how, "embedded_content")          # matched back to prompt 1
        self.assertEqual(bf.prompt_no, 1)
        self.assertEqual([c.first for c in bf.characters][:2], ["maya", "victor"])

    def test_numbered_heading_in_the_trajectory_matches_by_number(self):
        run = self._run("take8", {"TRAJECTORY.md": "notes\n\n### Prompt 2: whatever\n" + _body(2)})
        bf, how = match_brief(self.briefs, run)
        self.assertEqual((bf.prompt_no, how), (2, "embedded_number"))

    def test_unknown_brief_is_used_as_is(self):
        body = _body(1).replace("Maya Chen, 29, an analyst", "Ada Lin, 30, a pilot")
        run = self._run("take9", {"log.txt": body})
        bf, how = match_brief({}, run)
        self.assertEqual(how, "embedded_text")
        self.assertEqual(bf.prompt_no, 0)
        self.assertEqual(bf.characters[0].first, "ada")
        self.assertEqual(bf.word_cap, 125)

    def test_footprints_point_at_a_brief_without_quoting_it(self):
        """claudecode: the brief is compiled into series.js and named in
        TRAJECTORY.md; the character sheets carry the cast."""
        run = self._run("take12", {
            "TRAJECTORY.md": '# Trajectory — Prompt 1, "The Woman Behind the Numbers"\n'
                             "| Hard constraints | <125 spoken words/ep; no voiceover; "
                             'mandated line "You presented the numbers. I kept the receipts." |\n',
            "series.js": "const cast = [{name: 'Maya Chen'}, {name: 'Victor Hale'}];\n"})
        for stem in ("maya", "victor", "lena", "evelyn"):
            (self.r / "take12/assets/characters" / f"{stem}.png").parent.mkdir(
                parents=True, exist_ok=True)
            (self.r / "take12/assets/characters" / f"{stem}.png").write_bytes(b"x")
        bf, how = match_brief(self.briefs, run)
        self.assertEqual((bf.prompt_no, how), (1, "content_match"))

    def test_footprints_rebuild_a_partial_brief_without_prompt_md(self):
        run = self._run("take13", {
            "TRAJECTORY.md": '# Trajectory — Prompt 1, "The Woman Behind the Numbers"\n'
                             "| Hard constraints | <125 spoken words/ep; no voiceover; "
                             'mandated line "You presented the numbers. I kept the receipts." |\n'
                             "Maya Chen builds the forecast; Victor Hale presents it.\n"})
        for stem in ("maya", "victor"):
            (self.r / "take13/assets/characters").mkdir(parents=True, exist_ok=True)
            (self.r / "take13/assets/characters" / f"{stem}.png").write_bytes(b"x")
        bf, how = match_brief({}, run)
        self.assertEqual(how, "reconstructed")
        self.assertEqual([c.name for c in bf.characters], ["Maya Chen", "Victor Hale"])
        self.assertEqual(bf.word_cap, 125)
        self.assertIs(bf.voiceover_allowed, False)
        self.assertEqual(bf.mandated_lines, ["You presented the numbers. I kept the receipts."])
        self.assertEqual(bf.title, "The Woman Behind the Numbers")
        self.assertEqual(bf.timing_bands, [])                 # nothing stated, nothing invented
        self.assertIsNone(bf.episodes)

    def test_weak_footprints_do_not_pick_a_brief(self):
        """One first name in a log is not evidence for a brief."""
        run = self._run("take14", {"log.txt": "Maya said hi"})
        bf, how = match_brief(self.briefs, run)
        self.assertNotEqual(how, "content_match")

    def test_nothing_to_match_is_none(self):
        run = self._run("take10", {"log.txt": "the agent said hello"})
        self.assertEqual(match_brief(self.briefs, run), (None, "none"))

    def test_brief_fills_a_target_no_record_supplied(self):
        runs = [self._run("take11", {"log.txt": _body(1)}),
                self._run("prompt_02_x", {})]
        runs[1].target = Target(episodes=5, seconds_per_episode=30.0, source="run.json")
        matched = {r.run_id: match_brief(self.briefs, r) for r in runs}
        self.assertEqual(apply_brief_targets(runs, matched), 1)
        self.assertEqual((runs[0].target.episodes, runs[0].target.seconds_per_episode,
                          runs[0].target.aspect_ratio, runs[0].target.source),
                         (3, 60.0, "9:16", "brief"))
        self.assertEqual(runs[1].target.source, "run.json")     # a record is never overridden


class TestInterpreterResolution(unittest.TestCase):
    def setUp(self):
        runners._probe_cache.clear()
        self._env = os.environ.get(runners.ENV_OVERRIDE["spec"])

    def tearDown(self):
        runners._probe_cache.clear()
        if self._env is None:
            os.environ.pop(runners.ENV_OVERRIDE["spec"], None)
        else:
            os.environ[runners.ENV_OVERRIDE["spec"]] = self._env

    def test_environment_override_is_first_when_it_can_import(self):
        os.environ[runners.ENV_OVERRIDE["spec"]] = sys.executable
        exe, how = runners.resolve_interpreter("spec", probes={"spec": "import sys"})
        self.assertEqual((exe, how), (sys.executable, "env"))

    def test_a_candidate_that_cannot_import_is_passed_over(self):
        os.environ[runners.ENV_OVERRIDE["spec"]] = sys.executable
        exe, how = runners.resolve_interpreter(
            "spec", probes={"spec": "import no_such_module_for_this_probe"})
        self.assertNotEqual(how, "env")
        # whatever else exists on this machine either imports it or nothing does
        self.assertIn(how, ("repo_venv", "default", "current", "none"))

    def test_explicit_override_skips_the_probe(self):
        self.assertEqual(runners.resolve_interpreter("av", overrides={"av": "/x/python"}),
                         ("/x/python", "override"))


class TestOneShotDefinition(unittest.TestCase):
    """physical and the plain visual pass are run on the splice-derived
    shots when there are any -- the same file the seeded pass uses."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name) / "run"
        self.v = root / "output/ep1_final.mp4"
        self.v.parent.mkdir(parents=True)
        self.v.write_bytes(b"x")
        vids = [SimpleNamespace(info=SimpleNamespace(path=self.v), episode=1)]
        self.bus = EvidenceBus(SimpleNamespace(root=root, run_id="t", target=None),
                               SimpleNamespace(videos=vids, found=True),
                               Path(self.tmp.name) / "_out")
        self.calls = []

        def fake_module(name, video, **kw):
            self.calls.append((name, kw.get("tag", ""), kw.get("shots_json")))
            return {"n_shots": 2}
        self.bus.module = fake_module

    def tearDown(self):
        self.tmp.cleanup()

    def test_with_splices_every_module_gets_the_same_shots_file(self):
        self.bus.segments = lambda video: [{"start": 0.0, "end": 10.0, "frames": 240},
                                           {"start": 10.0, "end": 20.0, "frames": 240}]
        self.bus.physical(self.v)
        self.bus.visual(self.v)
        (n1, t1, s1), (n2, t2, s2) = self.calls
        self.assertEqual((n1, n2), ("physical", "visual"))
        self.assertTrue(t1.startswith("_shots_") and t1 == t2)
        self.assertEqual(s1, s2)
        self.assertEqual(json.loads(Path(s1).read_text(encoding="utf-8"))["shots"][1]["start"], 10.0)
        self.assertEqual(self.bus.status("physical", self.v, None), "not_run")   # stubbed module
        self.assertEqual(self.bus.tags[("visual", str(self.v))], t1)
        self.assertEqual(sum(1 for e in self.bus.ledger.entries
                             if e.rule == "splices_seed_shots"), 2)

    def test_without_splices_modules_run_plain(self):
        self.bus.segments = lambda video: None
        self.bus.physical(self.v)
        self.assertEqual(self.calls, [("physical", "", None)])
        self.assertEqual(self.bus.tags[("physical", str(self.v))], "")


if __name__ == "__main__":
    unittest.main()
