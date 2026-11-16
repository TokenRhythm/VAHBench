"""Audit passes 1-3, pinned.

Pass 1  discovery: a file under a scratch tree is not a delivery however it is
        named, and a declaration that lists everything selects nothing.
Pass 2  scoring: assembly of one clip six times is not a merge; yield is
        meaningless once delivery exceeds generation; seams that show count.
Pass 3  closed loop: the agent's declared structure and its own *Ok verdicts
        are held against what the pixels and the audio say.
"""
from __future__ import annotations

import os
import json
import sys
import tempfile
import unittest
from pathlib import Path


import stage4_checks.reports as extract                                        # noqa: E402
from stage2_delivery.discover import Run
from stage2_delivery.discover import Target
from stage2_delivery.discover import locate_delivery          # noqa: E402
from shared.types import Measurement
from shared.types import Scale
from shared.types import Verdict              # noqa: E402
from stage2_delivery.probe import MediaInfo                                # noqa: E402
from stage1_inputs.schema import Claims
from stage1_inputs.schema import TrajectoryFacts      # noqa: E402
import stage4_checks.cross_checks as flows
from stage4_checks.cross_checks import Ledger
#                                # noqa: E402


class FakeCache:
    """ffprobe stand-in: duration by file name, everything else nominal."""

    def __init__(self, durations: dict[str, float]):
        self.d = durations

    def get(self, path):
        p = Path(path)
        return MediaInfo(path=p, ok=True, duration=self.d.get(p.name, 5.0),
                         width=1080, height=1920, has_audio=True)


def _run(root: Path, files: dict[str, float], declared: list[str] | None = None,
         episodes: int = 3) -> tuple[Run, FakeCache]:
    for rel in files:
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"\x00")
    run = Run(root=root, harness="h", case="c", model=None,
              target=Target(episodes=episodes, seconds_per_episode=60.0,
                            aspect_ratio="9:16"),
              declared_names=list(declared or []))
    return run, FakeCache({Path(k).name: v for k, v in files.items()})


class TestScratchTreeIsNotDelivery(unittest.TestCase):
    """vimax: two near-60 s files under working/scene_*/ were counted as
    episodes 1 and 2 because `final` matched and the run had declared them --
    along with 157 other files.

    They are still not episodes. They are material: measured on their own
    seconds and charged the assembly point, rather than either passing as a
    delivery or vanishing into "produced nothing".
    """

    def test_scratch_files_with_dump_declaration_are_not_episodes(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            run, cache = _run(root, {"working/scene_1/final_video.mp4": 48.3,
                                     "working/scene_5/final_video.mp4": 56.4},
                              declared=[f"v{i}.mp4" for i in range(157)]
                              + ["final_video.mp4", "final_video.mp4"])
            d = locate_delivery(run, cache)
        self.assertEqual(d.selection, "unassembled")
        self.assertFalse(d.assembled)
        self.assertTrue(any("工作目录" in x for x in d.disagreement))
        self.assertTrue(any("全量转储" in x for x in d.disagreement))

    def test_scratch_files_without_any_declaration_are_not_episodes(self):
        with tempfile.TemporaryDirectory() as t:
            run, cache = _run(Path(t), {"working/scene_1/final_video.mp4": 58.0})
            d = locate_delivery(run, cache)
        self.assertEqual(d.selection, "unassembled")
        self.assertFalse(d.assembled)

    def test_selective_declaration_can_vouch_for_a_scratch_file(self):
        """A precise declaration is a statement; honour it."""
        with tempfile.TemporaryDirectory() as t:
            run, cache = _run(Path(t), {"working/final_video.mp4": 59.0},
                              declared=["final_video.mp4"])
            d = locate_delivery(run, cache)
        self.assertTrue(d.found)
        self.assertEqual(d.selection, "declared")

    def test_dump_declaration_earns_no_bonus_over_content(self):
        """Same duration, one under output/ (undeclared), one under working/
        (declared in a dump): output/ must win."""
        with tempfile.TemporaryDirectory() as t:
            run, cache = _run(Path(t), {"output/ep1.mp4": 60.0,
                                        "working/x.mp4": 60.0},
                              declared=[f"v{i}.mp4" for i in range(20)] + ["x.mp4"],
                              episodes=1)
            d = locate_delivery(run, cache)
        self.assertTrue(d.found)
        self.assertEqual(d.videos[0].info.path.name, "ep1.mp4")
        self.assertEqual(d.selection, "content")

    def test_archived_episode_manifest_survives_harness_inventory(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            run, cache = _run(root, {'artifacts/renders/ep1_concat.mp4': 60.0,
                                    'artifacts/renders/ep1_final.mp4': 60.04},
                              declared=[f'v{i}.mp4' for i in range(20)], episodes=1)
            (root / 'delivery_package.json').write_text(json.dumps({
                'episodes': [{'file': 'renders/ep1_final.mp4'}]}), encoding='utf8')
            delivery = locate_delivery(run, cache)
            self.assertEqual(delivery.videos[0].info.path.name, 'ep1_final.mp4')
            self.assertEqual(delivery.selection, 'declared')
            self.assertFalse(any('未被声明' in message for message in delivery.disagreement))

    def test_manifest_cannot_promote_short_or_missing_export(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            run, cache = _run(root, {'renders/ep1_concat.mp4': 60.0,
                                    'renders/ep1_final.mp4': 5.0}, episodes=1)
            (root / 'delivery_package.json').write_text(json.dumps({
                'episodes': [{'file': 'renders/ep1_final.mp4'}]}), encoding='utf8')
            self.assertEqual(locate_delivery(run, cache).videos[0].info.path.name, 'ep1_concat.mp4')
            (root / 'renders/ep1_final.mp4').unlink()
            self.assertEqual(locate_delivery(run, cache).videos[0].info.path.name, 'ep1_concat.mp4')

    def test_conflicting_manifests_do_not_choose_one_arbitrarily(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            run, cache = _run(root, {'renders/ep1_a.mp4': 60.0,
                                    'renders/ep1_b.mp4': 59.0}, episodes=1)
            for name, file in [('delivery_package.json', 'ep1_a.mp4'), ('manifest.json', 'ep1_b.mp4')]:
                (root / name).write_text(json.dumps({'episodes': [{'file': 'renders/' + file}]}), encoding='utf8')
            self.assertEqual(locate_delivery(run, cache).selection, 'content')

    def test_off_duration_episodes_are_not_missing_deliveries(self):
        with tempfile.TemporaryDirectory() as t:
            run, cache = _run(Path(t), {'output/ep1.mp4': 60.0,
                                        'output/ep2.mp4': 95.0,
                                        'output/ep3.mp4': 12.0})
            delivery = locate_delivery(run, cache)
            self.assertEqual([c.episode for c in delivery.videos], [1, 2, 3])
            self.assertTrue(delivery.assembled)

    def test_all_off_duration_episode_exports_are_still_assembled(self):
        with tempfile.TemporaryDirectory() as t:
            run, cache = _run(Path(t), {'output/episode_1.mp4': 95.0,
                                        'output/episode_2.mp4': 12.0}, episodes=2)
            delivery = locate_delivery(run, cache)
            self.assertEqual(len(delivery.videos), 2)
            self.assertTrue(delivery.assembled)

    def test_episode_tag_on_source_clips_does_not_make_them_exports(self):
        with tempfile.TemporaryDirectory() as t:
            run, cache = _run(Path(t), {'clips/ep1.mp4': 10.0,
                                        'output/ep2_shot_1.mp4': 10.0,
                                        'output/ep3_raw.mp4': 10.0})
            self.assertFalse(locate_delivery(run, cache).assembled)

    def test_selectively_listed_short_sources_are_not_episode_exports(self):
        with tempfile.TemporaryDirectory() as t:
            run, cache = _run(Path(t), {'working/scene_1/shots/shot_a.mp4': 8.0,
                                        'working/scene_1/shots/shot_b.mp4': 9.0,
                                        'working/scene_1/shots/shot_c.mp4': 10.0},
                              declared=['shot_a.mp4', 'shot_b.mp4', 'shot_c.mp4'])
            self.assertFalse(locate_delivery(run, cache).assembled)

    def test_episode_export_supersedes_equal_length_assembly_base(self):
        with tempfile.TemporaryDirectory() as t:
            run, cache = _run(Path(t), {'build/base_ep1.mp4': 60.0,
                                        'final/episode1_story.mp4': 60.0}, episodes=1)
            delivery = locate_delivery(run, cache)
            self.assertEqual(delivery.videos[0].info.path.name, 'episode1_story.mp4')
            self.assertTrue(any('中间文件' in x for x in delivery.disagreement))

    def test_duration_fit_cannot_promote_base_over_off_spec_export(self):
        with tempfile.TemporaryDirectory() as t:
            run, cache = _run(Path(t), {'build/base_ep1.mp4': 60.0,
                                        'final/episode1_story.mp4': 35.0}, episodes=1)
            self.assertEqual(locate_delivery(run, cache).videos[0].info.path.name,
                             'episode1_story.mp4')

    def test_base_is_not_removed_without_same_episode_export(self):
        with tempfile.TemporaryDirectory() as t:
            run, cache = _run(Path(t), {'build/base_ep1.mp4': 60.0,
                                        'final/episode2_story.mp4': 60.0}, episodes=2)
            self.assertEqual([c.episode for c in locate_delivery(run, cache).videos], [1, 2])

    def test_selective_declaration_can_deliver_a_build_file(self):
        with tempfile.TemporaryDirectory() as t:
            run, cache = _run(Path(t), {'build/base_ep1.mp4': 60.0,
                                        'final/episode1_story.mp4': 60.0},
                              declared=['base_ep1.mp4'], episodes=1)
            self.assertEqual(locate_delivery(run, cache).videos[0].info.path.name,
                             'base_ep1.mp4')

    def test_short_e_episode_ids_do_not_double_count_silent_intermediates(self):
        with tempfile.TemporaryDirectory() as t:
            files = {f'build/render/ep{i:02d}/video_silent.mp4': 60.0 for i in range(1, 4)}
            files.update({f'episodes/STORY_E{i:02d}.mp4': 60.0 for i in range(1, 4)})
            run, cache = _run(Path(t), files)
            delivery = locate_delivery(run, cache)
            self.assertEqual([c.episode for c in delivery.videos], [1, 2, 3])
            self.assertTrue(all(c.info.path.name.startswith('STORY_E') for c in delivery.videos))

    def test_short_e_episode_export_is_admitted_when_off_spec(self):
        with tempfile.TemporaryDirectory() as t:
            run, cache = _run(Path(t), {'episodes/STORY_E01.mp4': 35.0}, episodes=1)
            delivery = locate_delivery(run, cache)
            self.assertTrue(delivery.assembled)
            self.assertEqual(delivery.videos[0].episode, 1)

    def test_short_e_source_clips_are_not_promoted_to_exports(self):
        with tempfile.TemporaryDirectory() as t:
            run, cache = _run(Path(t), {'clips/E01.mp4': 10.0}, episodes=1)
            self.assertFalse(locate_delivery(run, cache).assembled)


class TestJoinArtifacts(unittest.TestCase):
    """artifact_checks reported seam glitches all along; nothing read them."""

    def _rep(self, n_ja, n_splices):
        return {"meta": {"n_frames": 1440},
                "splices": {"n_splices": n_splices, "segment_frames": [240] * 6},
                "findings": {"join_artifacts": {"present": bool(n_ja),
                                                "n_join_artifacts": n_ja}}}

    def test_seams_that_show_fail_on_the_statistical_band(self):
        """Per-seam rate, so the statistical band: one bad seam in six is a
        flaw, not the same score as six bad seams."""
        m = {x.key: x for x in extract.from_artifact(self._rep(2, 5))}["artifact.join_artifacts"]
        self.assertIs(m.verdict, Verdict.FAIL)
        self.assertEqual(m.scale, Scale.RATE)
        self.assertAlmostEqual(m.value, 0.4)

    def test_one_bad_seam_in_six_is_half_credit_not_zero(self):
        from stage5_scoring import DEFAULT_CUT_POINTS
        from stage5_scoring import check_score
        m = {x.key: x for x in extract.from_artifact(self._rep(1, 6))}["artifact.join_artifacts"]
        # 1/6 = 0.167 sits between clean 0.05 and suspect 0.20 on the
        # statistical band: partial credit, strictly between 0 and 1.
        s = check_score(m, DEFAULT_CUT_POINTS)
        self.assertGreater(s, 0.0)
        self.assertLess(s, 1.0)

    def test_locator_notes_without_direct_file_evidence_abstain(self):
        # Locator inventories mix harness and agent channels and are not proof.
        for notes in (["声明是全量转储"], ["声明了不存在的视频文件：a.mp4"]):
            d = {x.key: x for x in extract.from_delivery(
                3, _T(), 180.0, _facts(267.0, declared_deliverables=["a.mp4"]),
                disagreement=notes)}
            self.assertIs(d["claims.deliverables_consistent"].verdict, Verdict.ABSTAIN)

    def test_clean_seams_pass(self):
        m = {x.key: x for x in extract.from_artifact(self._rep(0, 5))}["artifact.join_artifacts"]
        self.assertIs(m.verdict, Verdict.PASS)

    def test_no_joins_means_nothing_to_judge(self):
        m = {x.key: x for x in extract.from_artifact(self._rep(0, 0))}["artifact.join_artifacts"]
        self.assertIs(m.verdict, Verdict.ABSTAIN)


class _T:
    episodes, aspect_ratio, resolution = 3, "9:16", None


def _facts(generated=None, source="per_request", **claims):
    from stage1_inputs.schema import RequestedParams
    return TrajectoryFacts(claims=Claims(**claims),
                           requested=RequestedParams(generated_seconds=generated,
                                                     generated_seconds_source=source))


class TestPaddingLandsOnProvenanceAlone(unittest.TestCase):
    """2.2 效率与 `provenance.generation_yield` 于 2026-09-22 删除（键与抽取器
    一并删，不是转成诊断项）。「交付多于生成」只由 duration_provenance 说一次，
    并在 1.1 里按生成秒数封顶；「生成多于交付」不再由任何地方计分。"""

    def test_delivery_exceeding_generation_is_a_provenance_verdict(self):
        """codex: 30 s generated, 180 s delivered."""
        d = {x.key: x for x in extract.from_delivery(3, _T(), 180.0, _facts(30.0))}
        self.assertIs(d["provenance.duration_provenance"].verdict, Verdict.ABSTAIN)
        self.assertNotIn("provenance.generation_yield", d)

    def test_trimming_is_not_charged_anywhere(self):
        """267 s shot, 180 s delivered: that is an edit, not waste."""
        d = {x.key: x for x in extract.from_delivery(3, _T(), 180.0, _facts(267.0))}
        self.assertIs(d["provenance.duration_provenance"].verdict, Verdict.ABSTAIN)
        self.assertNotIn("provenance.generation_yield", d)




class TestCrossChannel(unittest.TestCase):
    # `near_duplicates_void_assembly` (codex's six-times-concatenated clip
    # collecting P_merge credit) went with `provenance.segments_recovered` on
    # 2026-09-22: the loop it caught is charged in C's seconds ledger now.

    def test_self_check_contradicted_by_measurement(self):
        """The agent wrote audioOk=true; the track is silent."""
        ms = {"spec.audio_track": extract._m("spec.audio_track", Verdict.FAIL,
                                             reason="音轨存在但全程静音（rms=0）"),
              "claims.deliverables_consistent": extract._m(
                  "claims.deliverables_consistent", Verdict.PASS)}
        facts = TrajectoryFacts(claims=Claims(self_checks={"audioOk": True,
                                                            "durationOk": True}))
        flows.apply(ms, facts, Ledger())
        m = ms["claims.deliverables_consistent"]
        self.assertIs(m.verdict, Verdict.FAIL)
        self.assertIn("自述 audioOk=true", m.reason)
        self.assertIn("静音", m.reason)

    def test_self_check_that_holds_is_not_penalised(self):
        ms = {"spec.audio_track": extract._m("spec.audio_track", Verdict.PASS),
              "claims.deliverables_consistent": extract._m(
                  "claims.deliverables_consistent", Verdict.PASS)}
        flows.apply(ms, TrajectoryFacts(claims=Claims(self_checks={"audioOk": True})), Ledger())
        self.assertIs(ms["claims.deliverables_consistent"].verdict, Verdict.PASS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
