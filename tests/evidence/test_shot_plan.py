"""The shot plan: parsing it out of prose, and placing it in the brief's world.

Three harnesses write a shot plan in three layouts and two harnesses write
none. What these pin is the shape-blind reader that handles all three, and the
four false-positive classes real data produced -- a subtitle cue list, a build
log's stage table, a beat heading swallowing its own shot, and three episodes
collapsing onto one clock.

    python -m unittest tests.test_shot_plan -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path


import stage4_checks.video.brief as expectations                                   # noqa: E402
from stage1_inputs.brief import BriefFacts                                # noqa: E402
from stage1_inputs.extractors import _scan_shot_plan
from stage1_inputs.extractors import deterministic_extract  # noqa: E402
from shared.types import Verdict                                   # noqa: E402
from stage1_inputs.schema import TrajectoryFacts               # noqa: E402

HERMES = """# EPISODE 1 - "The Author of Record"
## HOOK (0:00-0:03) - IMMEDIATE HUMILIATION
SHOT 1 (0:00-0:03) INT. GLASS CONFERENCE ROOM / THROUGH GLASS - 2-shot from outside.
Maya watches through the glass from the corridor, motionless.

## CONFLICT SETUP (0:04-0:18)
SHOT 2 (0:04-0:07) on Victor at the podium, warm, arms opening to the room.
"""

OPENCLAW = """# EPISODE 1 - "BORROWED NUMBERS"
## Timing Map
| Block | Time | Beat |
|-------|------|------|
| Hook | 0-3s | Victor's applause under Maya's stolen slide, glass conference room |
| Conflict | 4-18s | Maya is erased from the credit; Lena clocks the injustice |
"""

OPENSQUILLA = """# EPISODE 1 - "The Forecast He Stole"
SHOT PLAN
- **S1 (0-4s, Hook)**: Victor applauded inside glass room beneath Maya's slide.
- **S2 (4-18s, Setup)**: Victor basks in applause, gestures at the slide.

# EPISODE 2 - "The Numbers That Lie"
SHOT PLAN
- **S1 (0-4s, Hook)**: Victor alone in the elevator lobby, phone buzzing.
"""


class TestShotPlanParsing(unittest.TestCase):

    def test_screenplay_table_and_bullet_all_parse(self):
        """One reader, three layouts. A parser per harness does not scale.

        hermes yields two, not three: its "## CONFLICT SETUP (0:04-0:18)"
        heading is immediately followed by SHOT 2's own span, leaving the
        heading nothing but "SHOT 2 (" to describe itself with. Fewer than
        four words is not a shot, so it drops -- which is the behaviour that
        keeps a beat heading from occupying the slot of the shot beneath it.
        """
        for name, body, want in (("hermes", HERMES, 2),
                                 ("openclaw", OPENCLAW, 2),
                                 ("opensquilla", OPENSQUILLA, 3)):
            got = _scan_shot_plan(body)
            self.assertEqual(len(got), want, f"{name}: {[s.text[:30] for s in got]}")

    def test_beat_heading_does_not_displace_its_shot(self):
        """hermes repeats a span: once on the beat heading, once on the shot
        under it. Taking the first occurrence keeps "IMMEDIATE HUMILIATION"
        and throws away the slug line that names the room."""
        first = _scan_shot_plan(HERMES)[0]
        self.assertIn("GLASS CONFERENCE ROOM", first.text)

    def test_每_episode_restarts_the_clock(self):
        """Keyed on the span alone, opensquilla's three episodes collapse to
        five shots and episode 3's text is served as episode 1's."""
        plan = _scan_shot_plan(OPENSQUILLA)
        self.assertEqual([s.episode for s in plan], [1, 1, 2])
        ep2 = [s for s in plan if s.episode == 2][0]
        self.assertEqual(ep2.start, 0.0)                 # same span as ep1's S1
        self.assertIn("elevator lobby", ep2.text)        # ...different text

    def test_subtitle_cues_are_not_a_shot_plan(self):
        """claudecode embeds its cue list in series.js. A pattern that accepts
        a bare "2-15" reads the whole track as twenty-six shots."""
        cues = "\n".join(f"[0:{i:02d}] Some line of dialogue here." for i in range(0, 40, 4))
        self.assertEqual(_scan_shot_plan(cues), [])

    def test_a_build_log_stage_table_is_not_a_shot_plan(self):
        """Real spans, real clock notation, nine minutes into a 60s episode.
        What gives them away is that a shot plan opens at the top."""
        log = ("| Stage | Time |\n"
               "| refs | 5:15-5:19 | 4 characters and 4 locations staged |\n"
               "| keys | 5:26-5:30 | 18 keyframes, each conditioned on refs |\n")
        self.assertEqual(_scan_shot_plan(log), [])

    def test_prose_ranges_are_not_spans(self):
        body = "Ep 1-2 share a palette. Real clips: 3 x 10 seconds each.\n"
        self.assertEqual(_scan_shot_plan(body), [])

    def test_per_episode_files_merge_rather_than_compete(self):
        """hermes writes one file per episode. Picking the richest file
        returned episode 3 and silently dropped the other two."""
        d = Path(tempfile.mkdtemp())
        (d / "01_episode1.md").write_text(HERMES, encoding="utf-8")
        (d / "02_episode2.md").write_text(
            OPENSQUILLA.split("# EPISODE 2")[1].join(["# EPISODE 2", ""]),
            encoding="utf-8")
        plan = deterministic_extract(d, d).continuity.shot_plan
        self.assertEqual({s.episode for s in plan}, {1, 2})


class TestScenePlanExpectation(unittest.TestCase):

    def _brief(self):
        b = BriefFacts(prompt_no=1)
        b.locations = ["an open-plan workspace", "glass conference room",
                       "elevator lobby", "executive presentation suite"]
        return b

    def _facts(self, body):
        f = TrajectoryFacts()
        f.continuity.shot_plan = _scan_shot_plan(body) or None
        return f

    def test_distinctive_tokens_come_from_the_brief(self):
        """"room" sits in one of these four and resolves it; hand-listing the
        vocabulary would need editing for every new brief."""
        d = expectations._distinctive_location_tokens(self._brief())
        self.assertIn("conference", d["glass conference room"])
        self.assertIn("elevator", d["elevator lobby"])
        self.assertNotIn("an", d["an open-plan workspace"])

    def test_plan_visiting_every_location_passes(self):
        body = ("# EPISODE 1\n"
                "| Hook | 0-3s | glass conference room, applause |\n"
                "| Two | 4-18s | the open-plan workspace, desks |\n"
                "| Three | 19-38s | elevator lobby, doors closing |\n"
                "| Four | 39-53s | executive presentation suite, the board |\n")
        m = expectations.scene_plan(self._brief(), self._facts(body))
        self.assertIs(m.verdict, Verdict.PASS)
        self.assertEqual(m.value, 0.0)

    def test_a_beat_naming_no_room_is_not_a_failure(self):
        """"Maya at her desk, silver pencil behind ear" continues the previous
        location and is supposed to. The rate is over the brief's locations,
        not over the agent's beats -- the characters_anchored shape."""
        body = ("# EPISODE 1\n"
                "| Hook | 0-3s | glass conference room, applause |\n"
                "| Two | 4-18s | Maya at her desk, silver pencil behind ear |\n"
                "| Three | 19-38s | she opens the original file, unedited |\n")
        m = expectations.scene_plan(self._brief(), self._facts(body))
        self.assertEqual(m.detail["n_shots_placed"], 1)     # only one beat placed
        self.assertEqual(m.value, 0.75)                     # 3 of 4 rooms unplanned
        self.assertIn("glass conference room", m.detail["locations_used"])

    def test_no_plan_abstains(self):
        m = expectations.scene_plan(self._brief(), TrajectoryFacts())
        self.assertIs(m.verdict, Verdict.ABSTAIN)

    def test_brief_without_locations_is_not_applicable(self):
        b = BriefFacts(prompt_no=1)
        self.assertIsNone(expectations.scene_plan(b, self._facts(OPENCLAW)))


if __name__ == "__main__":
    unittest.main()
