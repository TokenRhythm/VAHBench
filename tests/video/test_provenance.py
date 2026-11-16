"""Generated footage vs delivery: regressions from the local corpus audit."""

from shared.paths import REPO_ROOT
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO = REPO_ROOT
sys.path[:0] = [str(REPO), str(REPO / 'detectors' / 'spec_checks')]
from stage3_evidence.collector import EvidenceBus
from stage1_inputs.extractors import _from_trace_jsonl
from stage1_inputs.schema import TrajectoryFacts
from stage5_scoring import EpisodeLedger
from stage5_scoring import DEFAULT_CUT_POINTS
from stage5_scoring import score_completeness
from stage5_scoring import ledger_from_reports
from stage5_scoring import score_case


class TestSourceSeconds(unittest.TestCase):
    def score(self, ledgers, **kwargs):
        return score_completeness(ledgers, len(ledgers), DEFAULT_CUT_POINTS, **kwargs)

    def test_partial_post_durations_cannot_cap_real_delivery(self):
        rows = [{'url': 'https://x/videos', 'method': 'POST', 'status': 202, 'duration': 10},
                {'url': 'https://x/videos', 'method': 'POST', 'status': 202}]
        with patch.object(Path, 'is_file', return_value=True), patch.object(
                Path, 'read_text', return_value='\n'.join(map(json.dumps, rows))):
            facts = TrajectoryFacts()
            _from_trace_jsonl(Path('fixture'), facts)
        self.assertEqual(facts.requested.generated_seconds_source, 'partial_requests')
        result = self.score([EpisodeLedger('a', target_s=20, delivered_s=20)],
                            generated_s=facts.requested.generated_seconds,
                            generated_source=facts.requested.generated_seconds_source)
        self.assertEqual(result.score, 1.0)
        self.assertFalse(result.generation['bounds_genuine_seconds'])

    def test_accepted_or_unknown_requests_are_not_output_caps(self):
        for status in (202, None):
            row = dict(url='https://x/videos', method='POST', status=status, duration=10)
            with patch.object(Path, 'is_file', return_value=True), patch.object(
                    Path, 'read_text', return_value=json.dumps(row)):
                facts = TrajectoryFacts()
                _from_trace_jsonl(Path('fixture'), facts)
            result = self.score([EpisodeLedger('a', target_s=60, delivered_s=60)],
                                generated_s=facts.requested.generated_seconds,
                                generated_source=facts.requested.generated_seconds_source)
            self.assertEqual(result.score, 1.0)
            self.assertFalse(result.generation['applied'])


    def test_legacy_container_retiming_is_not_charged_twice(self):
        spec = dict(meta=dict(duration=82.79), checks=dict(retiming=dict(
            ok=False, source_duration=60.04, final_duration=82.79)))
        led = ledger_from_reports('a', 1, 60, 82.79, spec,
                                  dict(meta=dict(duration=60.04), findings={}))
        self.assertAlmostEqual(led.deductions_s, 22.75)

    def test_slowdown_and_frozen_frames_share_a_clock(self):
        # 20 clean + 20 frozen source seconds, uniformly slowed to 60 s.
        spec = dict(meta=dict(duration=60, video_duration=60), checks=dict(
            retiming=dict(ok=False, source_duration=40, final_duration=60, correspondence_verified=True)))
        artifact = dict(meta=dict(duration=60), findings=dict(
            frozen_runs=dict(segments=[dict(start_s=30, end_s=60)])))
        led = ledger_from_reports('a', 1, 60, 60, spec, artifact)
        self.assertAlmostEqual(led.deductions_s, 40)
        self.assertAlmostEqual(self.score([led]).genuine_s, 20)

    def test_frozen_tail_is_anchored_to_picture_not_audio_end(self):
        artifact = dict(meta=dict(duration=60), findings=dict(
            tail_freeze=dict(present=True, held_seconds=30)))
        led = ledger_from_reports('a', 1, 60, 82.79, artifact=artifact)
        self.assertAlmostEqual(led.deductions_s, 52.79)
        self.assertEqual(self.score([led]).genuine_s, 30)

    def test_large_tolerance_never_exempts_zero_content(self):
        led = EpisodeLedger('a', target_s=60, delivered_s=60, frozen_s=60)
        self.assertEqual(self.score([led], tolerance_s=60).score, 0)

    def test_small_padding_cannot_hide_in_duration_pass_band(self):
        led = EpisodeLedger('a', target_s=60, delivered_s=60, frozen_s=1)
        self.assertLess(self.score([led]).score, 1)
        clean = EpisodeLedger('a', target_s=60, delivered_s=59)
        self.assertEqual(self.score([clean]).score, 1)

    def test_missing_detectors_are_explicit_upper_bound(self):
        led = ledger_from_reports('a', 1, 60, 60)
        result = self.score([led])
        self.assertTrue(result.is_upper_bound)
        self.assertIn('timeline_repeats_unmeasured', result.evidence_gaps)
        self.assertIn('source_correspondence_unverified', result.evidence_gaps)

    def test_untimed_episode_prevents_total_score(self):
        ledgers = [EpisodeLedger('a', target_s=60, delivered_s=60),
                   EpisodeLedger('b', target_s=60)]
        result = score_case({}, ledgers, n_required=2, delivered=True)
        self.assertIsNone(result.C)
        self.assertIsNone(result.S)

    def test_measured_zero_is_preserved_under_scoring_floor(self):
        result = score_case({}, [EpisodeLedger('a', target_s=60, delivered_s=60,
                                               frozen_s=60)], n_required=1, delivered=True)
        self.assertEqual(result.completeness.score, 0)
        self.assertEqual(result.C, .05)
        self.assertEqual(result.S, 5)
        self.assertIsNone(result.Q)
        self.assertTrue(result.aggregation['completeness_floor']['applied'])




class TestConcatSources(unittest.TestCase):
    def bus(self, text_by_path):
        root = Path('/audit_run').resolve()
        video = root / 'output/ep1_final.mp4'
        bus = EvidenceBus(SimpleNamespace(root=root, run_id='fixture'),
                          SimpleNamespace(found=True, videos=[SimpleNamespace(
                              info=SimpleNamespace(path=video), episode=1)]), root)
        bus.files = lambda exts: list(text_by_path)
        return root, video, bus

    def test_relocated_repeated_clip_list_keeps_uses(self):
        root, video, bus = self.bus({})
        manifest = root / 'clips/ep1_concat_final.txt'
        bus.files = lambda exts: [manifest]
        clip = root / 'assets/clips/ep1_skill_10s.mp4'
        bus.produced_clips = lambda: [clip]
        text = "file 'C:/old/data/assets/clips/ep1_skill_10s.mp4'\n" * 6
        with patch.object(Path, 'read_text', return_value=text):
            self.assertEqual(bus.concat_sources(video), [clip] * 6)

    def test_ambiguous_or_trimmed_manifest_abstains(self):
        root, video, bus = self.bus({})
        manifest = root / 'ep1/concat.txt'
        bus.files = lambda exts: [manifest]
        a, b = root / 'a/clip.mp4', root / 'b/clip.mp4'
        bus.produced_clips = lambda: [a, b]
        with patch.object(Path, 'read_text', return_value="file 'clip.mp4'"):
            self.assertEqual(bus.concat_sources(video), [])
        bus.produced_clips = lambda: [a]
        with patch.object(Path, 'read_text', return_value="file 'clip.mp4'\ninpoint 1"):
            self.assertEqual(bus.concat_sources(video), [])

    def test_repeated_source_comparison_is_not_a_manifest_only_cap(self):
        meta = dict(path='source.mp4', video_duration=10)
        spec = dict(meta=dict(duration=60, video_duration=60), source_meta=[meta] * 6,
                    source_clips=['source.mp4'] * 6, checks=dict(retiming=dict(
                        ok=True, source_duration=60, final_duration=60,
                        frame_count_preserved=False)))
        led = ledger_from_reports('a', 1, 60, 60, spec)
        result = score_completeness([led], 1, DEFAULT_CUT_POINTS)
        comparison = result.episodes[0].source_comparison
        self.assertEqual(comparison['unique_source_video_s'], 10)
        self.assertEqual(comparison['n_source_uses'], 6)
        self.assertEqual(comparison['picture_minus_unique_source_s'], 50)
        self.assertTrue(result.is_upper_bound)
        self.assertEqual(result.genuine_s, 60)  # needs independent pixel evidence


if __name__ == '__main__':
    unittest.main()
