"""The evidence bus on a real (temporary) run directory, no modules needed:
clip/keyframe pairing, scratch preference, splice-derived shots, declared
conditioning resolved against a reorganised tree, subtitle mapping and
parsing. These are the joins that let one module's output configure another.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


from stage1_inputs.brief import BriefFacts
from stage1_inputs.brief import Character                        # noqa: E402
from stage3_evidence.collector import EvidenceBus
from stage3_evidence.collector import clip_index
from stage3_evidence.collector import episode_of
from stage3_evidence.collector import parse_subtitles
from stage1_inputs.schema import ConditioningUse
from stage1_inputs.schema import PlannedShot
from stage1_inputs.schema import TrajectoryFacts

FILES = [
    "output/ep1_final.mp4", "output/ep2_final.mp4", "output/ep3_final.mp4",
    "output/ep1.srt", "output/ep2.srt",
    "assets/clips/ep1_c01.mp4", "assets/clips/ep1_c02.mp4", "assets/clips/ep1_c03.mp4",
    "assets/work/ep1_c01_n.mp4", "assets/work/ep1_raw.mp4", "assets/work/ep3_raw.mp4",
    "assets/keyframes/ep1_c01.png", "assets/keyframes/ep1_c02.png", "assets/keyframes/ep1_c03.png",
    "assets/characters/maya.png", "assets/characters/victor.png",
    "assets/locations/lobby.png",
    "assets/clips/ep2_c01.mp4", "assets/keyframes/ep2_c01.png",
]
SRT = ("1\n00:00:01,000 --> 00:00:03,000\nHello there\n\n"
       "2\n00:00:04,000 --> 00:00:06,000\n[SIGN]\n")


class TestNames(unittest.TestCase):
    def test_episode_and_clip_numbers(self):
        self.assertEqual(episode_of(Path("out/ep3_final.mp4")), 3)
        self.assertEqual(episode_of(Path("episode_02/final.mp4")), 2)
        self.assertIsNone(episode_of(Path("series_cover.png")))
        self.assertEqual(clip_index(Path("ep1_c07.mp4")), 7)
        self.assertEqual(clip_index(Path("ep01_shot_03.mp4")), 3)
        self.assertEqual(clip_index(Path("clip_012.mp4")), 12)
        self.assertIsNone(clip_index(Path("ep1_raw.mp4")))
        self.assertIsNone(clip_index(Path("ep1_pass2.mp4")))
        # the e1s10 scheme: episode and scene in one token, no "ep"
        self.assertEqual((episode_of(Path("clip_e1s10.mp4")), clip_index(Path("clip_e1s10.mp4"))), (1, 10))
        self.assertEqual((episode_of(Path("seg_e2s3.mp4")), clip_index(Path("seg_e2s3.mp4"))), (2, 3))
        self.assertEqual((episode_of(Path("e1_s1_hook.mp4")), clip_index(Path("e1_s1_hook.mp4"))), (1, 1))
        self.assertIsNone(episode_of(Path("e1a.mp4")))          # no boundary after the number
        self.assertIsNone(episode_of(Path("cover_frame.png")))

    def test_short_episode_ids_match_delivery_and_evidence(self):
        from stage2_delivery.discover import EP_NUM
        for name, expected in [('STORY_E03.mp4', 3), ('E06.srt', 6)]:
            self.assertEqual(episode_of(Path(name)), expected)
            self.assertEqual(int(EP_NUM.search(name).group(1)), expected)
        for name in ('e1a.mp4', 'model_e123.mp4'):
            self.assertIsNone(episode_of(Path(name)))
            self.assertIsNone(EP_NUM.search(name))

    def test_parse_srt_vtt_ass(self):
        self.assertEqual(parse_subtitles(SRT, ".srt")[0],
                         {"start": 1.0, "end": 3.0, "text": "Hello there"})
        vtt = "WEBVTT\n\n00:01.500 --> 00:03.000\nHi <b>there</b>\n"
        self.assertEqual(parse_subtitles(vtt, ".vtt"), [{"start": 1.5, "end": 3.0, "text": "Hi there"}])
        ass = "[Events]\nDialogue: 0,0:00:01.50,0:00:03.00,Default,,0,0,0,,{\\i1}Hi\\Nthere\n"
        self.assertEqual(parse_subtitles(ass, ".ass"), [{"start": 1.5, "end": 3.0, "text": "Hi there"}])


class TestBusOnDisk(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "run"
        for rel in FILES:
            p = self.root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")
        (self.root / "output/ep1.srt").write_text(SRT, encoding="utf-8")
        self.v = {n: self.root / f"output/ep{n}_final.mp4" for n in (1, 2, 3)}
        vids = [SimpleNamespace(info=SimpleNamespace(path=self.v[n]), episode=n) for n in (1, 2, 3)]
        self.delivery = SimpleNamespace(videos=vids, found=True)
        self.run = SimpleNamespace(root=self.root, run_id="t", target=None)

    def tearDown(self):
        self.tmp.cleanup()

    def bus(self, facts=None):
        return EvidenceBus(self.run, self.delivery, Path(self.tmp.name) / "_out", facts=facts)

    def test_same_shots_across_buses_preserve_detector_cache_identity(self):
        from stage4_checks.dispatch import _cache_signature
        segments = [{'start': 0, 'end': 5}, {'start': 5, 'end': 10}]
        path = self.bus().shots_json(self.v[1], segments)
        os.utime(path, ns=(1_750_000_000_000_000_000,) * 2)
        options = {'shots_json': path}
        first = _cache_signature('physical', self.v[1], Path(sys.executable), options)
        self.bus().shots_json(self.v[1], segments)
        self.assertEqual(first, _cache_signature('physical', self.v[1], Path(sys.executable), options))
        self.bus().shots_json(self.v[1], [{'start': 0, 'end': 4}, {'start': 4, 'end': 10}])
        self.assertNotEqual(first, _cache_signature('physical', self.v[1], Path(sys.executable), options))

    def test_episode_clips_prefer_the_copy_outside_scratch(self):
        clips = self.bus().episode_clips(self.v[1])
        self.assertEqual([p.name for p in clips], ["ep1_c01.mp4", "ep1_c02.mp4", "ep1_c03.mp4"])
        self.assertEqual(clips[0].parent.name, "clips")

    def test_source_clips_seed_retiming_and_are_logged(self):
        b = self.bus()
        self.assertEqual(len(b.source_clips(self.v[1])), 3)
        self.assertEqual([p.name for p in b.source_clips(self.v[3])], ["ep3_raw.mp4"])   # master fallback
        self.assertEqual(sum(1 for e in b.ledger.entries if e.rule == "clips_seed_retiming"), 2)

    def test_keyframes_pair_by_stem_and_demand_completeness(self):
        b = self.bus()
        self.assertEqual(sorted(b.keyframes_for_clips(self.v[1])), [1, 2, 3])
        self.assertEqual([p.name for p in b.conditioning_images(self.v[1])],
                         ["ep1_c01.png", "ep1_c02.png", "ep1_c03.png"])
        (self.root / "assets/keyframes/ep1_c02.png").unlink()
        b2 = self.bus()
        self.assertEqual(len(b2.keyframes_for_clips(self.v[1])), 2)
        self.assertEqual(b2.conditioning_images(self.v[1]), [])     # a gap would shift every later shot

    def test_segments_from_splices_merge_slivers_into_shots(self):
        b = self.bus()
        b._memo[("artifact", str(self.v[1]), "")] = {
            "meta": {"fps": 24.0}, "splices": {"segment_frames": [240, 240, 4, 236]}}
        segs = b.segments(self.v[1])
        self.assertEqual([s["frames"] for s in segs], [240, 244, 236])
        shots = json.loads(b.shots_json(self.v[1]).read_text(encoding="utf-8"))["shots"]
        self.assertEqual(len(shots), 3)
        self.assertAlmostEqual(shots[1]["end"], (240 + 244) / 24, places=3)

    def test_shattered_clip_is_reconciled_to_the_keyframe_count(self):
        """claudecode ep1: the sixth clip came back as 17+4+4+25+4+186 frames."""
        b = self.bus()
        b._memo[("artifact", str(self.v[1]), "")] = {
            "meta": {"fps": 24.0},
            "splices": {"segment_frames": [240, 240, 240, 240, 240, 17, 4, 4, 25, 4, 186]}}
        raw = b.segments(self.v[1])
        self.assertEqual(len(raw), 7)                     # the 25-frame piece survives the sliver cut
        segs, notes = b.segments_for(self.v[1], 6)
        self.assertEqual(len(segs), 6)
        self.assertEqual(sum(s["frames"] for s in segs), 1440)
        self.assertEqual(len(notes), 1)
        # a genuine two-second shot is never merged away
        b2 = self.bus()
        b2._memo[("artifact", str(self.v[1]), "")] = {
            "meta": {"fps": 24.0}, "splices": {"segment_frames": [240, 240, 60, 240]}}
        segs2, notes2 = b2.segments_for(self.v[1], 3)
        self.assertEqual((len(segs2), notes2), (4, []))

    def test_declared_clip_lengths_decide_which_cuts_are_seams(self):
        """claudecode ep3: 183+57 frames where the fifth 10 s clip should be --
        a cut at 47.6 s that is at none of the declared seams. 2.4 s is too
        long for the size rule; the trajectory's structure settles it."""
        facts = SimpleNamespace(
            produced=None, conditioning=None,
            requested=SimpleNamespace(clip_durations=[10.0]),
            claims=SimpleNamespace(declared_clips_per_episode=[6, 6, 6]))
        b = self.bus(facts)
        b._memo[("artifact", str(self.v[3]), "")] = {
            "meta": {"fps": 24.0},
            "splices": {"segment_frames": [240, 240, 240, 240, 183, 57, 240]}}
        self.assertEqual(b.expected_seams(self.v[3], 6), [10.0, 20.0, 30.0, 40.0, 50.0])
        segs, notes = b.segments_for(self.v[3], 6)
        self.assertEqual([s["frames"] for s in segs], [240] * 6)
        self.assertEqual(len(notes), 1)
        self.assertIn("47.6s", notes[0])
        # without clip lengths on record the size rule applies, and 2.4 s stays
        b2 = self.bus()
        b2._memo[("artifact", str(self.v[3]), "")] = {
            "meta": {"fps": 24.0},
            "splices": {"segment_frames": [240, 240, 240, 240, 183, 57, 240]}}
        segs2, _ = b2.segments_for(self.v[3], 6)
        self.assertEqual(len(segs2), 7)

    def test_declared_conditioning_resolves_against_the_current_tree(self):
        """The language model reads paths as the log wrote them; the data has
        moved since. Match on the last two components."""
        facts = SimpleNamespace(produced=None, conditioning=[
            ConditioningUse(references=["C:\\old\\layout\\assets\\characters\\maya.png"])])
        b = self.bus(facts)
        chars = [Character(name="Maya Chen", first="maya"), Character(name="Victor Hale", first="victor")]
        sheets = b.character_sheets(chars)
        self.assertEqual(sheets["Maya Chen"].name, "maya.png")
        self.assertEqual(b.sheet_sources, {"Maya Chen": "conditioning", "Victor Hale": "filesystem"})

    def test_subtitles_map_to_the_episode_and_parse(self):
        b = self.bus()
        p, cues = b.subtitles_for(self.v[1])
        self.assertEqual(p.name, "ep1.srt")
        self.assertEqual(cues[0]["text"], "Hello there")
        self.assertIsNone(b.subtitles_for(self.v[3]))


if __name__ == "__main__":
    unittest.main()


# TestDeclaredLocations was deleted on 2026-09-22 with `visual.scene_grouping`
# and the `declared_locations` seed that only it read (SCORING_SYSTEM B.4f).


