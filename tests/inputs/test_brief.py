"""The brief compiler and the expectations built on it. Offline: the
expectations are driven through a fake evidence bus.

Fixtures quote the real prompts' phrasings, because the first parser lost the
first character of three prompts ("Five central characters." was not
"principal"), read "2 minutes" as no duration, and filed Mrs. Alvarez under
"mrs." while the trajectory had drawn her as `alvarez.png`.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


import stage4_checks.video.brief as expectations                                  # noqa: E402
from stage1_inputs.brief import BriefFacts
from stage1_inputs.brief import Character
from stage1_inputs.brief import parse_prompts
from stage1_inputs.brief import prompt_number
from stage4_checks.cross_checks import Ledger                                  # noqa: E402
from shared.types import Scale
from shared.types import Verdict                          # noqa: E402
from stage1_inputs.schema import TrajectoryFacts             # noqa: E402
from stage1_inputs.schema import ConditioningUse

P1 = """### Prompt 1: The Woman Behind the Numbers

- **Format**: Multi-episode series (3 episodes)
- **Single-episode duration**: 60 seconds
- **Aspect ratio**: 9:16 (vertical screen)
- **Characters**: Four principal characters. Maya Chen, 29, a quiet analyst; Victor Hale, 41, a department head; Lena Ortiz, 31, Maya's colleague; and Evelyn Shaw, 52, the CFO.
- **Setting**: A contemporary office, including an open-plan workspace, glass conference room, elevator lobby, and executive presentation suite.
- **Character Visual Design**: Maya wears restrained navy officewear and a silver mechanical pencil behind one ear; Victor wears immaculate charcoal suits and a red tie; Lena carries a bright yellow notebook; Evelyn is distinguished by a sculptural white bob.
- **Dialogue Norms**: Keep dialogue below 125 spoken words per episode. No explanatory voiceover. Place one memorable line near the final reversal: “You presented the numbers. I kept the receipts.”
- **Timing Allocation**: For each episode, use 0–3 seconds for the hook, 4–18 seconds for conflict setup, 19–38 seconds for escalation, 39–53 seconds for the reversal, and 54–60 seconds for the cliff-hanger or payoff.
- **Final Output Checklist**: Deliver all three finished episodes, a series cover frame, episode titles, a 60-word series description, five platform tags, complete subtitle text, and one teaser sentence after Episodes 1 and 2.
"""

P7 = """### Prompt 7: The Nineteenth Floor

- **Format**: Multi-episode series (5 episodes)
- **Single-episode duration**: 4 minutes
- **Aspect ratio**: 16:9 (horizontal)
- **Characters**: Seven recurring characters. Mara Ellis, 34, a paramedic on leave; Theo Grant, 17, a student; Mrs. Alvarez, 65, the building's historian; Simon Pike, 45, the property manager.
- **Dialogue Norms**: Use restrained English, and no more than 500 spoken words per episode. Allow one recurring line—“Prediction is not permission”—which changes meaning.
- **Final Output Checklist**: Deliver five finished episodes, series and episode covers, titles, a spoiler-free synopsis, tags, complete subtitles, and teaser lines for Episodes 1–4.
"""


class TestParser(unittest.TestCase):
    def setUp(self):
        self.b = parse_prompts(P1 + "\n" + P7)

    def test_numbers_and_minutes(self):
        b1, b7 = self.b[1], self.b[7]
        self.assertEqual((b1.episodes, b1.seconds_per_episode, b1.aspect_ratio), (3, 60.0, "9:16"))
        self.assertEqual((b7.episodes, b7.seconds_per_episode, b7.aspect_ratio), (5, 240.0, "16:9"))

    def test_first_character_survives_any_count_sentence(self):
        self.assertEqual([c.first for c in self.b[1].characters],
                         ["maya", "victor", "lena", "evelyn"])
        self.assertEqual(self.b[7].characters[0].name, "Mara Ellis")

    def test_quoted_and_article_prefixed_final_characters_are_kept(self):
        text = """### Prompt 2: Example

- **Format**: Multi-episode series (3 episodes)
- **Characters**: Five recurring characters. Noah Reed, 24, a courier; Mrs. Bell, 68, a seamstress; Jay, 10, her grandson; Omar Singh, 36, a caretaker; and “M. Bell,” the unknown sender.
"""
        b = parse_prompts(text)[2]
        self.assertEqual([c.name for c in b.characters],
                         ["Noah Reed", "Mrs. Bell", "Jay", "Omar Singh", "M. Bell"])
        self.assertEqual(b.declared_character_count, 5)
        self.assertEqual(b.parse_warnings, [])

    def test_character_count_mismatch_is_reported(self):
        text = """### Prompt 2: Example

- **Format**: Multi-episode series (3 episodes)
- **Characters**: Three recurring characters. Noah Reed, 24, a courier; Mrs. Bell, 68, a seamstress.
"""
        b = parse_prompts(text)[2]
        self.assertEqual(b.declared_character_count, 3)
        self.assertEqual(len(b.characters), 2)
        self.assertTrue(b.parse_warnings)

    def test_honorific_yields_surname_token(self):
        alv = next(c for c in self.b[7].characters if "Alvarez" in c.name)
        self.assertEqual(alv.first, "alvarez")
        self.assertEqual(alv.tokens, ["alvarez"])

    def test_visual_design_attached(self):
        maya = self.b[1].characters[0]
        self.assertIn("silver mechanical pencil", maya.visual)

    def test_locations(self):
        self.assertEqual(self.b[1].locations[:2], ["an open-plan workspace", "glass conference room"])

    def test_dialogue_rules(self):
        b1 = self.b[1]
        self.assertEqual(b1.word_cap, 125)
        self.assertIs(b1.voiceover_allowed, False)
        self.assertEqual(b1.mandated_lines, ["You presented the numbers. I kept the receipts."])
        self.assertEqual(b1.line_placement, "reversal")
        b7 = self.b[7]
        self.assertEqual(b7.word_cap, 500)
        self.assertIsNone(b7.voiceover_allowed)
        self.assertEqual(b7.mandated_lines, ["Prediction is not permission"])
        self.assertIsNone(b7.line_placement)          # "recurring", not placed

    def test_timing_bands(self):
        rev = self.b[1].band("reversal")
        self.assertEqual((rev.start, rev.end), (39.0, 53.0))
        self.assertEqual(self.b[1].band("payoff").end, 60.0)

    def test_deliverables_distinct_keys_and_counts(self):
        d1 = {d.key: d.count for d in self.b[1].deliverables}
        self.assertEqual(d1["episodes"], 3)
        self.assertEqual(d1["description"], 60)
        self.assertEqual(d1["tags"], 5)
        self.assertIn("series_cover", d1)
        self.assertIn("teaser", d1)
        d7 = {d.key for d in self.b[7].deliverables}
        self.assertTrue({"series_cover", "episode_covers"} <= d7)
        self.assertEqual(len(self.b[7].deliverables), len(d7))     # no duplicates

    def test_prompt_number(self):
        self.assertEqual(prompt_number("prompt_07_the_nineteenth_floor"), 7)
        self.assertIsNone(prompt_number("case_x"))


# ------------------------------------------------------------------ bus ---
class FakeBus:
    def __init__(self, transcripts=None, av=None, images=(), sheets=None,
                 visual_refs=None, texts=(), silent=False, subtitles=None,
                 scene=None, clips=None, keyframes=None, segments=None, asr_segments=None):
        self._t, self._av, self._imgs = transcripts or {}, av or {}, list(images)
        self._sheets, self._vis, self._texts = sheets or {}, visual_refs, list(texts)
        self._silent, self.log = silent, []
        self._subs = subtitles or {}        # video -> (Path, cues)
        self._scene = scene or {}           # video -> scene_reference_check
        self._clips = clips or {}           # video -> [Path]
        self._kf = keyframes or {}          # video -> {clip index: Path}
        self._segs = segments or {}         # video -> [{start, end, frames}]
        self._asr_segments = asr_segments or {}
        self.ledger = Ledger()
        self.seeds, self.sheet_sources = {}, {}

    def delivered_videos(self):
        return [Path(k) for k in self._t] or [Path("ep1.mp4")]

    def seeded_visual(self, v, characters, brief=None, facts=None):
        sheets = self.character_sheets(characters)
        scene = self._scene.get(str(v))
        self.seeds[str(v)] = {"sheets": {k: str(p) for k, p in sheets.items()},
                              "sheet_sources": dict(self.sheet_sources),
                              "scene_references": ["kf"] * int(scene["n_shots"]) if scene else [],
                              "shots_json": None}
        if not sheets and not scene:
            return None
        rep = dict(self.visual(v, list(sheets.values())) or {})
        if scene:
            rep["scene_reference_check"] = scene
        return rep

    def subtitles_for(self, v):
        return self._subs.get(str(v))

    def episode_clips(self, v):
        return self._clips.get(str(v), [])

    def segments(self, v):
        return self._segs.get(str(v))

    def keyframes_for_clips(self, v):
        return self._kf.get(str(v), {})

    def transcript(self, v):
        t = self._t.get(str(v))
        if t is None:
            return None
        return {"silent": self._silent, "n_words": len(t.split()),
                "segments": self._asr_segments.get(str(v), [{"start": 45.0, "end": 48.0, "text": t}]), "text": t}

    def silent(self, v):
        return self._silent

    def av(self, v):
        return self._av.get(str(v))

    def produced_images(self):
        return self._imgs

    def files(self, exts, max_n=5000):
        return [p for p in self._imgs if p.suffix in exts]

    def text_artifacts(self, max_bytes=0):
        return self._texts

    def character_sheets(self, characters):
        out = {c.name: self._sheets[c.first] for c in characters if c.first in self._sheets}
        self.sheet_sources = {k: "filesystem" for k in out}
        return out

    def visual(self, v, reference_images=None):
        return self._vis


def _b1():
    return parse_prompts(P1)[1]


class TestExpectations(unittest.TestCase):
    LINE = "You presented the numbers. I kept the receipts."

    def test_word_cap_over_and_under(self):
        b = _b1()
        under = FakeBus({"ep1.mp4": "one two three four five six"})
        over = FakeBus({"ep1.mp4": " ".join(f"w{i}" for i in range(150))})
        self.assertIs(expectations.word_cap(b, under, under.delivered_videos()).verdict, Verdict.PASS)
        m = expectations.word_cap(b, over, over.delivered_videos())
        self.assertIs(m.verdict, Verdict.FAIL)
        self.assertAlmostEqual(m.value, 25 / 125)

    def test_silence_suppresses_dialogue_expectations(self):
        """Zero words on a silent track is not "under the cap"."""
        b = _b1()
        bus = FakeBus({"ep1.mp4": ""}, silent=True)
        self.assertIs(expectations.word_cap(b, bus, bus.delivered_videos()).verdict, Verdict.NOT_APPLICABLE)
        ms = {m.key: m for m in expectations.mandated_lines(b, bus, bus.delivered_videos(), [])}
        self.assertIs(ms["brief.mandated_line"].verdict, Verdict.NOT_APPLICABLE)

    def test_mandated_line_found_and_placed(self):
        b = _b1()
        bus = FakeBus({"ep1.mp4": "well. " + self.LINE + " indeed"})
        ms = {m.key: m for m in expectations.mandated_lines(b, bus, bus.delivered_videos(), [])}
        self.assertIs(ms["brief.mandated_line"].verdict, Verdict.PASS)
        self.assertIs(ms["brief.mandated_line_placement"].verdict, Verdict.PASS)   # at 45 s, band 39-53

    def test_mandated_line_tolerates_asr_slips(self):
        b = _b1()
        bus = FakeBus({"ep1.mp4": "you presented the numbers i kept receipts"})
        ms = {m.key: m for m in expectations.mandated_lines(b, bus, bus.delivered_videos(), [])}
        self.assertIs(ms["brief.mandated_line"].verdict, Verdict.PASS)

    def test_mandated_line_triangulation(self):
        """Planned (in the script file) but not delivered is execution failure;
        absent from both is the requirement being ignored."""
        b = _b1()
        bus = FakeBus({"ep1.mp4": "nothing relevant here"})
        planned = expectations.mandated_lines(b, bus, bus.delivered_videos(),
                                              [(Path("script.txt"), "MAYA: " + self.LINE)])[0]
        self.assertIs(planned.verdict, Verdict.FAIL)
        self.assertEqual(planned.detail["triangulation"], "planned_not_delivered")
        ignored = expectations.mandated_lines(b, bus, bus.delivered_videos(), [])[0]
        self.assertIn("被忽略", ignored.reason)

    def test_offscreen_dialogue_does_not_prove_narration(self):
        b = _b1()
        av = {"cleanliness": {"rms": 0.1}, "presence": {"windows": [
            {"start": 0, "end": 10, "sounds": [["Speech", 0.8]], "people": 2,
             "verdicts": [{"sound": "Speech", "verdict": "plausible"}]},
            {"start": 10, "end": 20, "sounds": [["Speech", 0.9]], "people": 0,
             "verdicts": [{"sound": "Speech", "verdict": "offscreen_voice"}]},
            {"start": 20, "end": 30, "sounds": [["Music", 0.5]], "people": 0,
             "verdicts": [{"sound": "Music", "verdict": "exempt"}]},
        ]}}
        bus = FakeBus({"ep1.mp4": "x"}, av={"ep1.mp4": av})
        m = expectations.no_voiceover(b, bus, bus.delivered_videos())
        self.assertIs(m.verdict, Verdict.ABSTAIN)
        self.assertIsNone(m.value)

    def test_no_voiceover_not_asked_means_no_expectation(self):
        b = parse_prompts(P7)[7]
        self.assertIs(expectations.no_voiceover(b, FakeBus({"ep1.mp4": "x"}), [Path("ep1.mp4")]).verdict,
                      Verdict.NOT_APPLICABLE)

    def test_deliverables_file_and_text_evidence(self):
        b = _b1()
        texts = [(Path("FINAL_DELIVERY.md"),
                  "Episode 1: The Slide\nDescription: a series about receipts\n"
                  "Tags: #office, #drama, #revenge, #finance, #series\nTeaser: next time...")]
        bus = FakeBus({"ep1.mp4": "x", "ep2.mp4": "x", "ep3.mp4": "x"},
                      images=[Path("out/ep1_cover.png"), Path("out/ep1.srt")], texts=texts)
        m = expectations.deliverables(b, bus, bus.delivered_videos(), texts)
        self.assertIs(m.verdict, Verdict.PASS, m.detail)

    def test_deliverables_in_markdown_headings(self):
        """claudecode's FINAL_DELIVERY.md: headings, items on the lines below."""
        b = _b1()
        md = ("## Series title options\n- Receipts\n- The Numbers\n\n## Episode titles\n"
              "| Episode | Title |\n| 1 | The Slide |\n\n## Series description\nMaya builds...\n\n"
              "## Platform tags\n#office #drama #revenge #finance #series\n\n"
              "## Next-episode teasers\n- Ep1: ...\n")
        texts = [(Path("FINAL_DELIVERY.md"), md)]
        bus = FakeBus({"ep1.mp4": "x", "ep2.mp4": "x", "ep3.mp4": "x"},
                      images=[Path("output/series_cover.png"), Path("output/ep1.srt")], texts=texts)
        m = expectations.deliverables(b, bus, bus.delivered_videos(), texts)
        self.assertIs(m.verdict, Verdict.PASS, m.detail)

    def test_deliverables_missing_are_named(self):
        b = _b1()
        bus = FakeBus({"ep1.mp4": "x"}, images=[], texts=[])
        m = expectations.deliverables(b, bus, bus.delivered_videos(), [])
        self.assertIs(m.verdict, Verdict.FAIL)
        self.assertTrue(m.detail["missing"])

    SHEETS = {"maya": Path("maya.png"), "victor": Path("victor.png"),
              "lena": Path("lena.png"), "evelyn": Path("evelyn.png")}

    @staticmethod
    def _vis(matches):
        return {"character_identity": {"reference_identity": {
            "n_references": 4, "n_matched": sum(matches),
            "per_reference": [{"best_similarity": 0.7 if m else 0.2, "matches": m}
                              for m in matches]}}}

    def test_characters_anchored_through_trajectory_sheets(self):
        b = _b1()
        bus = FakeBus({"ep1.mp4": "x"}, sheets=self.SHEETS,
                      visual_refs=self._vis([True, True, True, False]))
        m = expectations.characters_anchored(b, bus, bus.delivered_videos())
        self.assertIs(m.verdict, Verdict.FAIL)
        # Fraction mapping uses the actual cast size: three of four matched.
        self.assertAlmostEqual(m.value, 1 / 4)
        self.assertEqual(m.detail["n_matched"], 3)
        self.assertIn("Evelyn Shaw", m.reason)
        self.assertEqual(m.detail["n_characters_in_brief"], 4)
        self.assertTrue(all(v == "sheet" for v in m.detail["triangulation"].values()))

    def test_characters_are_anchored_at_series_level(self):
        """The CFO enters in episode 3. Unmatched in ep1 and ep2 is not a defect
        if some episode shows her face."""
        b = _b1()

        class EpBus(FakeBus):
            def visual(self, v, reference_images=None):
                return TestExpectations._vis([True, True, True, "ep3" in str(v)])
        bus = EpBus({"ep1.mp4": "x", "ep2.mp4": "x", "ep3.mp4": "x"}, sheets=self.SHEETS)
        m = expectations.characters_anchored(b, bus, bus.delivered_videos())
        self.assertIs(m.verdict, Verdict.PASS)
        self.assertEqual(m.detail["episodes_seen"]["Evelyn Shaw"], ["ep3.mp4"])
        self.assertEqual(len(m.detail["episodes_seen"]["Maya Chen"]), 3)

    def test_characters_abstain_without_sheets(self):
        b = _b1()
        m = expectations.characters_anchored(b, FakeBus({"ep1.mp4": "x"}), [Path("ep1.mp4")])
        self.assertIs(m.verdict, Verdict.ABSTAIN)
        self.assertIn("no_sheet", m.detail["triangulation"]["Maya Chen"])


class TestCoupling(unittest.TestCase):
    """The expectations that read all three channels at once: subtitles held
    against the transcript, delivered shots against their own conditioning
    keyframes, asset coverage, and the image-call triangulation behind an
    absent character sheet."""

    @staticmethod
    def _cues(*lines, start=1.0):
        return [{"start": start + 5 * i, "end": start + 5 * i + 3, "text": t}
                for i, t in enumerate(lines)]

    def test_subtitles_aligned_against_transcript(self):
        b = _b1()
        cues = self._cues("[APPENDIX — M. CHEN]", "You presented the numbers.",
                          "I kept the receipts.", "Victor. My office.")
        bus = FakeBus({"ep1.mp4": "you presented the numbers i kept the receipts victor my office"},
                      subtitles={"ep1.mp4": (Path("ep1.srt"), cues)},
                      asr_segments={"ep1.mp4": cues[1:]})
        m = expectations.subtitles_aligned(b, bus, bus.delivered_videos(), [])
        self.assertIs(m.verdict, Verdict.PASS)
        self.assertEqual(m.detail["n_cues"], 3)          # the bracketed cue is screen text
        self.assertIn("subtitles_vs_transcript", bus.ledger.rules)

    def test_subtitles_unheard_cues_are_a_rate(self):
        b = _b1()
        cues = self._cues("Ms. Chen. You are on the agenda.", "You presented the numbers.",
                          "I kept the receipts.", "Victor. My office.")
        bus = FakeBus({"ep1.mp4": "the bridge holds you presented the numbers "
                                  "i kept the receipts victor my office"},
                      subtitles={"ep1.mp4": (Path("ep1.srt"), cues)},
                      asr_segments={"ep1.mp4": [{**cues[0], "text": "the bridge holds"}, *cues[1:]]})
        m = expectations.subtitles_aligned(b, bus, bus.delivered_videos(), [])
        self.assertIs(m.verdict, Verdict.FAIL)
        # 11 matched words, 18 subtitle words and 14 spoken words:
        # both the unheard cue and the uncaptained speech count.
        self.assertAlmostEqual(m.value, 1 - 2 * 11 / (18 + 14))
        self.assertEqual(m.detail["unheard"], ["Ms. Chen. You are on the agenda."])

    def test_subtitles_silent_track_suppresses(self):
        b = _b1()
        bus = FakeBus({"ep1.mp4": ""}, silent=True,
                      subtitles={"ep1.mp4": (Path("ep1.srt"), self._cues("Hello there friend"))})
        self.assertIs(expectations.subtitles_aligned(b, bus, bus.delivered_videos(), []).verdict,
                      Verdict.NOT_APPLICABLE)

    def test_subtitles_demanded_but_none_delivered_voids(self):
        b = _b1()
        bus = FakeBus({"ep1.mp4": "x"})
        self.assertIs(expectations.subtitles_aligned(b, bus, bus.delivered_videos(), []).verdict,
                      Verdict.VOID)

    def test_subtitles_as_text_only_abstains_but_counts_as_delivered(self):
        b = _b1()
        md = "\n".join(f"| 00:{i:02d}.0 – 00:{i + 3:02d}.0 | line {i} |" for i in range(0, 12, 4))
        bus = FakeBus({"ep1.mp4": "x"})
        m = expectations.subtitles_aligned(b, bus, bus.delivered_videos(), [(Path("FINAL.md"), md)])
        self.assertIs(m.verdict, Verdict.ABSTAIN)
        d = expectations.deliverables(b, bus, bus.delivered_videos(), [(Path("FINAL.md"), md)])
        self.assertNotIn("complete subtitle text", d.detail["missing"])
        self.assertEqual(d.detail["owned_elsewhere"]["subtitles"], "brief.subtitles_aligned")

    def test_placeholder_subtitles_fail_when_dialogue_is_heard(self):
        """codex ep1.srt: six `[HOOK] The Woman Behind the Numbers` labels."""
        b = _b1()
        cues = self._cues("[HOOK] The Woman Behind the Numbers", "[CONTEXT] The Woman Behind the Numbers")
        heard = " ".join(f"w{i}" for i in range(40))          # forty distinct words
        bus = FakeBus({"ep1.mp4": heard}, subtitles={"ep1.mp4": (Path("ep1.srt"), cues)})
        m = expectations.subtitles_aligned(b, bus, bus.delivered_videos(), [])
        self.assertIs(m.verdict, Verdict.FAIL)
        self.assertEqual(m.value, 1.0)
        self.assertEqual(m.detail["placeholder_cues"], 2)
        # speaker tags in brackets are not labels: the words differ per cue
        tagged = self._cues("[MAYA] You presented the numbers.", "[VICTOR] My team, of course.")
        self.assertEqual(len(expectations._dialogue_cues(tagged)[0]), 2)
        # nothing heard either: a fake file over a music track must not rank
        # above delivering no file (VOID below) -- so VOID, not ABSTAIN
        quiet = FakeBus({"ep1.mp4": "hm"}, subtitles={"ep1.mp4": (Path("ep1.srt"), cues)})
        self.assertIs(expectations.subtitles_aligned(b, quiet, quiet.delivered_videos(), []).verdict,
                      Verdict.VOID)

    def test_whisper_hallucination_is_not_dialogue(self):
        """codex ep2: "Of course." seven times over music; ep3: 「請問」 x 24."""
        b = _b1()
        loop = " ".join(["of course."] * 7)
        bus = FakeBus({"ep1.mp4": loop})
        m = expectations.word_cap(b, bus, bus.delivered_videos())
        self.assertIs(m.verdict, Verdict.ABSTAIN)
        self.assertIn("幻觉", m.reason)
        # real subtitle lines over that track: written, never voiced
        cues = self._cues("You are on the agenda today", "The bridge holds two point four")
        bus2 = FakeBus({"ep1.mp4": loop}, subtitles={"ep1.mp4": (Path("ep1.srt"), cues)})
        m2 = expectations.subtitles_aligned(b, bus2, bus2.delivered_videos(), [])
        self.assertIs(m2.verdict, Verdict.FAIL)
        self.assertIn("写了没配音", m2.reason)
        ok, _ = expectations._dialogue_heard(
            {"n_words": 12, "text": "the bridge holds two point four the appendix is a formality victor",
             "segments": [{"text": "the bridge holds two point four"},
                          {"text": "the appendix is a formality"}, {"text": "victor"}]})
        self.assertTrue(ok)

    def test_placeholder_subtitle_file_is_not_a_delivered_subtitle(self):
        import tempfile
        b = _b1()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ep1.srt"
            p.write_text("1\n00:00:01,000 --> 00:00:03,000\n[HOOK] Title\n", encoding="utf-8")
            bus = FakeBus({"ep1.mp4": "x", "ep2.mp4": "x", "ep3.mp4": "x"}, images=[p])
            # the placeholder file is also among the text artifacts (it is text):
            # its own timestamps must not count as "subtitle text in a document"
            texts = [(p, p.read_text(encoding="utf-8"))]
            d = expectations.deliverables(b, bus, bus.delivered_videos(), texts)
            self.assertEqual(d.detail["owned_elsewhere"]["subtitles"], "brief.subtitles_aligned")
            self.assertFalse(expectations._subtitle_files_with_dialogue([p]))
            p.write_text("1\n00:00:01,000 --> 00:00:03,000\nYou are on the agenda\n", encoding="utf-8")
            d2 = expectations.deliverables(b, bus, bus.delivered_videos(), [])
            self.assertTrue(expectations._subtitle_files_with_dialogue([p]))
            self.assertNotIn("complete subtitle text", d2.detail["missing"])

    def test_live_track_without_speech_voids_the_word_cap(self):
        b = _b1()
        bus = FakeBus({"ep1.mp4": "la la"})          # two words: music, not dialogue
        m = expectations.word_cap(b, bus, bus.delivered_videos())
        self.assertIs(m.verdict, Verdict.ABSTAIN)
        self.assertIn("转不出台词", m.reason)

    def test_subtitles_not_demanded_is_not_applicable(self):
        b = BriefFacts(prompt_no=99)
        bus = FakeBus({"ep1.mp4": "x"})
        self.assertIs(expectations.subtitles_aligned(b, bus, bus.delivered_videos(), []).verdict,
                      Verdict.NOT_APPLICABLE)

    def test_scene_fidelity_from_conditioning_keyframes(self):
        b = _b1()
        scene = {"n_shots": 6, "n_shots_matching_own_ref": 5,
                 "per_shot": [{"shot": i, "matches_own_ref": i != 3} for i in range(6)]}
        bus = FakeBus({"ep1.mp4": "x"}, scene={"ep1.mp4": scene})
        m = expectations.scene_fidelity(b, bus, bus.delivered_videos())
        self.assertIs(m.verdict, Verdict.FAIL)
        self.assertAlmostEqual(m.value, 1 / 6)
        self.assertEqual(m.detail["unmatched_shots"], [3])

    def test_scene_fidelity_abstains_without_keyframes(self):
        b = _b1()
        m = expectations.scene_fidelity(b, FakeBus({"ep1.mp4": "x"}), [Path("ep1.mp4")])
        self.assertIs(m.verdict, Verdict.ABSTAIN)

    def test_asset_utilisation_counts_sheets_and_keyframes(self):
        b = _b1()
        clips = [Path(f"ep1_c0{i}.mp4") for i in range(1, 7)]
        kf = {i: Path(f"ep1_c0{i}.png") for i in range(1, 7)}
        bus = FakeBus({"ep1.mp4": "x"}, sheets=TestExpectations.SHEETS,
                      clips={"ep1.mp4": clips}, keyframes={"ep1.mp4": kf})
        m = expectations.asset_utilisation(b, bus, bus.delivered_videos(), None)
        self.assertIs(m.verdict, Verdict.PASS)
        self.assertEqual((m.detail["units"], m.detail["strong_edges"]), (10, 10))

    def test_asset_utilisation_partial_is_a_rate(self):
        b = _b1()
        clips = [Path(f"ep1_c0{i}.mp4") for i in range(1, 7)]
        bus = FakeBus({"ep1.mp4": "x"}, sheets={"maya": Path("maya.png")},
                      clips={"ep1.mp4": clips}, keyframes={"ep1.mp4": {1: Path("ep1_c01.png")}})
        m = expectations.asset_utilisation(b, bus, bus.delivered_videos(), None)
        self.assertIs(m.verdict, Verdict.FAIL)
        self.assertAlmostEqual(m.value, 1 - 2 / 10)

    def test_pixel_cuts_cannot_be_missing_reference_units(self):
        bus = FakeBus({'ep1.mp4': 'x'}, sheets=TestExpectations.SHEETS,
                      segments={'ep1.mp4': [{'start': i, 'end': i + 1} for i in range(153)]})
        def forbidden(_):
            raise AssertionError('Asset coverage must not infer source units from pixel cuts')
        bus.segments = forbidden
        m = expectations.asset_utilisation(_b1(), bus, bus.delivered_videos(), None)
        self.assertIs(m.verdict, Verdict.ABSTAIN)
        self.assertIsNone(m.value)
        self.assertEqual(m.detail['clips']['n'], 0)
        self.assertEqual(m.detail['characters']['with_sheet'], 4)

    def test_partial_reference_binding_stays_unknown_with_real_sheets(self):
        clips = [Path('ep1_c01.mp4'), Path('ep1_c02.mp4')]
        bus = FakeBus({'ep1.mp4': 'x'}, sheets=TestExpectations.SHEETS,
                      clips={'ep1.mp4': clips}, keyframes={'ep1.mp4': {1: Path('maya.png')}})
        bus._resolve_asset = lambda raw: None
        facts = TrajectoryFacts(conditioning=[ConditioningUse(
            target='clips across episodes', references=['maya.png'], kind='character')])
        m = expectations.asset_utilisation(_b1(), bus, bus.delivered_videos(), facts)
        self.assertIs(m.verdict, Verdict.ABSTAIN)
        self.assertIsNone(m.value)
        self.assertEqual(m.detail['unbound_conditioning_targets'], ['clips across episodes'])

    def test_explicit_no_references_remains_failure_without_clip_inventory(self):
        bus = FakeBus({'ep1.mp4': 'x'})
        facts = TrajectoryFacts(conditioning=[])
        m = expectations.asset_utilisation(_b1(), bus, bus.delivered_videos(), facts)
        self.assertIs(m.verdict, Verdict.FAIL)
        self.assertEqual(m.value, 1)

    def test_bound_partial_references_remain_a_measured_rate(self):
        clips = [Path('ep1_c01.mp4'), Path('ep1_c02.mp4')]
        bus = FakeBus({'ep1.mp4': 'x'}, sheets=TestExpectations.SHEETS,
                      clips={'ep1.mp4': clips}, keyframes={'ep1.mp4': {1: Path('maya.png')}})
        bus._resolve_asset = lambda raw: Path(raw).resolve()
        facts = TrajectoryFacts(conditioning=[ConditioningUse(
            target=str(clips[0]), references=['maya.png'], kind='character')])
        m = expectations.asset_utilisation(_b1(), bus, bus.delivered_videos(), facts)
        self.assertIs(m.verdict, Verdict.FAIL)
        self.assertAlmostEqual(m.value, 1 / 6)

    def test_complete_assets_do_not_depend_on_an_unresolved_extra_claim(self):
        clips = [Path('ep1_c01.mp4')]
        bus = FakeBus({'ep1.mp4': 'x'}, sheets=TestExpectations.SHEETS,
                      clips={'ep1.mp4': clips}, keyframes={'ep1.mp4': {1: Path('maya.png')}})
        facts = TrajectoryFacts(conditioning=[ConditioningUse(
            target='unspecified', references=['maya.png'], kind='character')])
        m = expectations.asset_utilisation(_b1(), bus, bus.delivered_videos(), facts)
        self.assertIs(m.verdict, Verdict.PASS)
        self.assertEqual(m.value, 0)

    def test_located_cover_reference_does_not_cancel_clip_coverage(self):
        clips = [Path('ep1_c01.mp4'), Path('ep1_c02.mp4')]
        bus = FakeBus({'ep1.mp4': 'x'}, sheets=TestExpectations.SHEETS,
                      clips={'ep1.mp4': clips}, keyframes={'ep1.mp4': {1: Path('maya.png')}})
        bus._resolve_asset = lambda raw: Path(raw).resolve()
        facts = TrajectoryFacts(conditioning=[ConditioningUse(
            target='series_cover.png', references=['maya.png'], kind='character')])
        m = expectations.asset_utilisation(_b1(), bus, bus.delivered_videos(), facts)
        self.assertIs(m.verdict, Verdict.FAIL)
        self.assertAlmostEqual(m.value, 1 / 6)

    def test_asset_utilisation_nothing_named_follows_the_trajectory(self):
        """Nothing on disk is named after a character or clip: abstain when the
        trajectory is silent (2.4 全无), FAIL when it says no references were used."""
        b = _b1()
        f = TrajectoryFacts()
        bus = FakeBus({"ep1.mp4": "x"}, clips={"ep1.mp4": [Path("ep1_c01.mp4")]})
        self.assertIs(expectations.asset_utilisation(b, bus, bus.delivered_videos(), f).verdict,
                      Verdict.ABSTAIN)
        f.conditioning = []
        m = expectations.asset_utilisation(b, bus, bus.delivered_videos(), f)
        self.assertIs(m.verdict, Verdict.FAIL)
        self.assertEqual(m.value, 1.0)

    def test_characters_without_sheets_read_the_image_calls(self):
        b = _b1()
        f = TrajectoryFacts()
        f.calls.image_gen = 0
        bus = FakeBus({"ep1.mp4": "x"})
        m = expectations.characters_anchored(b, bus, [Path("ep1.mp4")], f)
        self.assertIs(m.verdict, Verdict.ABSTAIN)
        self.assertEqual(m.detail["triangulation"]["Maya Chen"], "ignored")
        self.assertIn("image_triangulation", bus.ledger.rules)

    def test_silence_is_logged_as_a_flow(self):
        b = _b1()
        bus = FakeBus({"ep1.mp4": ""}, silent=True)
        expectations.word_cap(b, bus, bus.delivered_videos())
        e = next(e for e in bus.ledger.entries if e.rule == "penalty_prerequisite")
        self.assertEqual((e.dst, e.effect), ("brief.word_cap", "annotate"))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestContinuityContractExpectation(unittest.TestCase):
    """brief wardrobe -> the agent's own contract -> its own prompts.

    Both steps close inside the trajectory: no frame is decoded, so what they
    measure is the agent and nothing else.
    """

    def _brief(self):
        b = BriefFacts(prompt_no=1)
        b.characters = [
            Character(name="Maya Chen", first="maya",
                      visual="Maya wears restrained navy officewear and a "
                             "silver mechanical pencil behind one ear"),
            Character(name="Victor Hale", first="victor",
                      visual="Victor wears immaculate charcoal suits and a red tie"),
        ]
        return b

    def _facts(self, text=None, prompts=None):
        f = TrajectoryFacts()
        f.continuity.text = text
        f.requested.prompts = prompts
        f.continuity.source = "series_bible.md" if text else None
        return f

    def test_contract_carrying_the_wardrobe_passes(self):
        text = ("Maya: restrained navy officewear, silver mechanical pencil "
                "behind one ear. Victor: immaculate charcoal suits, red tie.")
        m = expectations.continuity_contract(self._brief(), self._facts(text))
        self.assertIs(m.verdict, Verdict.PASS)
        self.assertEqual(m.value, 0.0)

    def test_contract_without_wardrobe_fails_hard(self):
        """codex's CONTINUITY.md: episode structure, no character in it.
        Measured on the real run it lands at 0.96 while the other four sit at
        0.15-0.19, which is the spread this check exists to produce."""
        m = expectations.continuity_contract(
            self._brief(),
            self._facts("3 episodes x 60 seconds; aspect 9:16. Chronological."))
        self.assertIs(m.verdict, Verdict.FAIL)
        self.assertGreater(m.value, 0.8)

    def test_character_name_alone_is_not_a_spec(self):
        """A contract that says "Maya" and nothing about how she looks has
        carried nothing; the name tokens are dropped before matching."""
        m = expectations.continuity_contract(
            self._brief(), self._facts("Maya Chen. Victor Hale."))
        self.assertEqual(m.value, 1.0)

    def test_missing_contract_abstains(self):
        m = expectations.continuity_contract(self._brief(), self._facts())
        self.assertIs(m.verdict, Verdict.ABSTAIN)

    def test_brief_without_wardrobe_is_not_applicable(self):
        b = BriefFacts(prompt_no=1)
        b.characters = [Character(name="Maya Chen", first="maya")]
        self.assertIsNone(expectations.continuity_contract(b, self._facts("x")))

    def test_a_contract_nobody_quotes_is_an_unused_asset(self):
        """The text-side twin of reference_identity: the agent wrote the spec
        and then generated from prompts that do not mention it."""
        facts = self._facts(
            text="Maya: restrained navy officewear, silver mechanical pencil "
                 "behind one ear. Victor: immaculate charcoal suits, red tie.",
            prompts=["A woman stands in an office. Cinematic lighting, 9:16."])
        contract = expectations.continuity_contract(self._brief(), facts)
        used = expectations.specs_in_prompts(self._brief(), facts)
        self.assertIs(contract.verdict, Verdict.PASS)      # written
        self.assertIs(used.verdict, Verdict.FAIL)          # never quoted
        self.assertGreater(used.value, 0.7)

    def test_prompts_unrecoverable_abstains_rather_than_scoring_zero(self):
        """Three of five harnesses log prompts only inside trajectory exports
        that no deterministic rule can separate from the scaffolding's system
        prompt. Absent evidence must not read as absent effort."""
        m = expectations.specs_in_prompts(
            self._brief(), self._facts(text="navy officewear"))
        self.assertIs(m.verdict, Verdict.ABSTAIN)
