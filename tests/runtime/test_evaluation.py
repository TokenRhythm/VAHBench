"""Production-path failures found during real-data end-to-end validation."""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import stage4_checks.dispatch as runners
from shared.settings import Settings
from stage3_evidence.collector import EvidenceBus
from stage2_delivery.discover import Candidate
from stage2_delivery.discover import Delivery
from stage2_delivery.discover import Run
from stage2_delivery.discover import Target
from stage2_delivery.probe import MediaInfo
from stage2_delivery.probe import ProbeCache
from stage1_inputs.preprocess import InputPreprocessor
from stage1_inputs.schema import TrajectoryFacts
import stage4_checks.reports as adapters
from run import _finalize_evaluation_status
from run import evaluate_run


class TestSettings(unittest.TestCase):
    def test_output_budget_uses_yaml_and_environment_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / 'eval.yaml'
            cfg.write_text('lm:\n  max_tokens: 12345\n', encoding='utf8')
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(Settings.load(cfg).lm.max_tokens, 12345)
                os.environ['VAB_EVAL_LM_MAX_TOKENS'] = '23456'
                self.assertEqual(Settings.load(cfg).lm.max_tokens, 23456)


class TestCredentials(unittest.TestCase):
    def test_credentials_redacted_in_text_and_nested_json_with_valid_audit(self):
        secret = 'sk-or-v1-' + 'a' * 40
        for filename, raw in [('trace.jsonl', json.dumps({'content': secret}) + '\n'),
                              ('run.json', json.dumps({'result': {'note': secret}})),
                              ('script.py', 'key = "' + secret + '"\n')]:
            with self.subTest(filename=filename):
                cleaner = InputPreprocessor()
                part = cleaner.prepare(filename, raw)
                audit = cleaner.finalize([part])
                self.assertNotIn(secret, part)
                self.assertNotIn(secret, json.dumps(audit))
                self.assertEqual(audit['rule_hits']['P007'], 1)
                self.assertEqual(audit['validation']['status'], 'passed')


class TestEvaluationStatus(unittest.TestCase):
    def test_absent_audio_is_not_clean_audio_and_does_not_invoke_asr(self):
        report = {'audio_stream': {'present': False}, 'cleanliness': None,
                  'presence': None, 'mouth': None}
        measurements = {m.key: m for m in adapters.from_av(report)}
        self.assertEqual(measurements['spec.audio_track'].verdict.value, 'FAIL')
        self.assertEqual(measurements['av.clipping'].verdict.value, 'NOT_APPLICABLE')
        self.assertEqual(measurements['av.speaker_consistency'].verdict.value, 'NOT_APPLICABLE')
        root = Path('not_read_by_this_test')
        bus = EvidenceBus(Run(root, 'h', 'case', 'm', Target()), Delivery(), root)
        with patch.object(bus, 'av', return_value=report), \
             patch('stage3_evidence.collector.subprocess.run') as process:
            transcript = bus.transcript(root / 'video.mp4')
        process.assert_not_called()
        self.assertEqual(transcript['model'], 'skipped:no_audio')
        self.assertFalse(transcript['audio_present'])
        self.assertEqual(transcript['n_words'], 0)

    def test_same_named_video_measurements_keep_their_own_plan_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos = [root / 'scene1/final_video.mp4', root / 'scene2/final_video.mp4']
            run = Run(root, 'h', 'case', 'm', Target(episodes=2, seconds_per_episode=60))
            delivery = Delivery(videos=[Candidate(MediaInfo(p, True, duration=60)) for p in videos],
                                selection='content')
            report = {'meta': {'duration': 60}, 'checks': {'duration': {'ok': True}}}
            with patch('run.gather_facts', return_value=(TrajectoryFacts(), None)), \
                 patch.object(EvidenceBus, 'spec', return_value=report), \
                 patch.object(EvidenceBus, 'status', return_value='ok'), \
                 patch.object(EvidenceBus, 'assembly_evidence', return_value={}), \
                 patch.object(EvidenceBus, 'plan_spans', side_effect=lambda p: [{'start': 0, 'end': 60, 'text': p.parent.name}]), \
                 patch('run.flows.apply') as apply_flows:
                result = evaluate_run(run, delivery, root / 'out', {}, modules=('spec',), device='cpu', lm_opts={})
            plans = apply_flows.call_args.kwargs['plan_spans']
            self.assertEqual(set(plans), {str(p) for p in videos})
            for episode, video in zip(result['episode_measurements'], videos):
                origin = episode['measurements']['spec.duration']['detail']['episode']
                self.assertEqual(plans[origin][0]['text'], video.parent.name)

    def test_discovery_never_invokes_the_scoring_algorithm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = Run(root, 'h', 'case', 'm', Target())
            with patch('run.gather_facts', return_value=(TrajectoryFacts(), None)), \
                 patch('run.score_case') as score:
                report = evaluate_run(run, Delivery(), root, {}, modules=(), device='cpu',
                                      lm_opts={'discover_only': True})
            score.assert_not_called()
            self.assertEqual(report['evaluation_status'], 'discovery_only')
            self.assertNotIn('quality', report)

    def test_failed_extraction_cannot_publish_a_score(self):
        facts = TrajectoryFacts()
        facts.lm_audit = {'chunks_total': 3, 'chunks_valid': 2, 'input_complete': True}
        report = {'S': 99, 'C': 1, 'sufficient_evidence': True}
        _finalize_evaluation_status(report, facts)
        self.assertIsNone(report['S'])
        self.assertFalse(report['sufficient_evidence'])
        self.assertEqual(report['evaluation_status'], 'extraction_failed')

    def test_honest_unknown_fields_after_success_are_not_a_transport_failure(self):
        facts = TrajectoryFacts()
        facts.lm_audit = {'chunks_total': 3, 'chunks_valid': 3, 'input_complete': True}
        report = {'S': 42}
        _finalize_evaluation_status(report, facts)
        self.assertEqual(report['S'], 42)
        self.assertEqual(report['evaluation_status'], 'completed')

    def test_failed_detector_keeps_observed_total_but_marks_execution_gap(self):
        report = {'S': 99, 'C': 1, 'sufficient_evidence': True,
                  'modules': {'av': {'ep1.mp4': 'ok', 'ep2.mp4': 'audio decode failed'}}}
        _finalize_evaluation_status(report, TrajectoryFacts())
        self.assertEqual(report['S'], 99)
        self.assertTrue(report['score_is_partial'])
        self.assertEqual(report['evaluation_status'], 'incomplete_evidence')
        self.assertEqual(report['execution_failures'][0]['video'], 'ep2.mp4')

    def test_failed_asr_keeps_observed_total_and_failure_record(self):
        report = {'S': 99, 'C': 1, 'execution_failures': [
            {'module': 'asr', 'video': 'ep2.mp4', 'reason': 'transcription unavailable'}]}
        _finalize_evaluation_status(report, TrajectoryFacts())
        self.assertEqual(report['S'], 99)
        self.assertTrue(report['score_is_partial'])

    def test_measured_zero_completeness_remains_zero_despite_failed_audio(self):
        report = {'S': 0, 'C': 0, 'modules': {'av': {'ep1.mp4': 'audio decode failed'}}}
        _finalize_evaluation_status(report, TrajectoryFacts())
        self.assertEqual(report['S'], 0)
        self.assertTrue(report['execution_failures'])

    def test_discovery_has_no_top_level_scores(self):
        report = dict(S=0, C=0, Q=.5, P=.5)
        _finalize_evaluation_status(report, TrajectoryFacts(), discover_only=True)
        self.assertTrue(all(report[k] is None for k in ('S', 'C', 'Q', 'P')))


class TestInputCache(unittest.TestCase):
    def test_silence_cache_refreshes_media_and_parameters_and_rejects_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / 'video.mp4'
            video.write_bytes(b'original')
            run = Run(root, 'control', 'silence', 'local', Target())
            bus = EvidenceBus(run, Delivery(), root / 'out')
            silent_log = 'silence_start: 0\nsilence_end: 10 | silence_duration: 10'
            with patch('stage3_evidence.collector._ffmpeg', return_value='ffmpeg'), \
                 patch('stage3_evidence.collector.subprocess.run', side_effect=[
                     SimpleNamespace(returncode=0, stderr=''),
                     SimpleNamespace(returncode=0, stderr=silent_log),
                     SimpleNamespace(returncode=0, stderr=''),
                     SimpleNamespace(returncode=1, stderr='decode error')]) as execute:
                self.assertEqual(bus.silence_runs(video), [])
                self.assertEqual(bus.silence_runs(video), [])
                video.write_bytes(b'replaced by silence')
                self.assertEqual(bus.silence_runs(video)[0]['duration'], 10)
                self.assertEqual(bus.silence_runs(video, min_s=20), [])
                # Failed decode must be unknown, never cached as clean audio.
                self.assertIsNone(bus.silence_runs(video, noise_db=-30))
                fresh = EvidenceBus(run, Delivery(), root / 'out')
                self.assertEqual(fresh.silence_runs(video)[0]['duration'], 10)
                self.assertEqual(execute.call_count, 4)

    def test_probe_cache_refreshes_same_size_same_second_media_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / 'video.mp4'
            cache_file = Path(tmp) / 'cache.json'
            video.write_bytes(b'old')
            stamp = 1_790_000_000_100_000_000
            os.utime(video, ns=(stamp, stamp))
            with patch('stage2_delivery.probe.probe', side_effect=[
                    MediaInfo(video, True, duration=10),
                    MediaInfo(video, True, duration=5)]) as probe_media:
                cache = ProbeCache(cache_file)
                self.assertEqual(cache.get(video).duration, 10)
                cache.save()
                cache = ProbeCache(cache_file)
                self.assertEqual(cache.get(video).duration, 10)
                video.write_bytes(b'new')
                os.utime(video, ns=(stamp + 100_000_000, stamp + 100_000_000))
                self.assertEqual(cache.get(video).duration, 5)
                self.assertEqual(probe_media.call_count, 2)

    def test_report_write_and_cache_hit_survive_windows_long_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "detector_fixture" / "run.py"
            script.parent.mkdir()
            script.write_text("# isolated cache test entrypoint\n")
            deep = root
            while len(str(deep)) < 285:
                deep /= 'nested_audit_directory'

            def compute(cmd, **kw):
                pending = Path(cmd[cmd.index('--out') + 1])
                self.assertLessEqual(len(pending.name), 26)
                pending.write_text('{"valid":true}', encoding='utf8')
                return SimpleNamespace(returncode=0, stderr='', stdout='')

            try:
                with patch.dict(runners.SCRIPTS, {'spec': script}), \
                     patch.object(runners, '_interpreter', return_value=os.sys.executable), \
                     patch.object(runners.subprocess, 'run', side_effect=compute) as execute:
                    first = runners.run_module('spec', root / 'video.mp4', deep)
                    second = runners.run_module('spec', root / 'video.mp4', deep)
                    self.assertTrue(first.ok, first.error)
                    self.assertTrue(second.ok, second.error)
                    self.assertEqual(execute.call_count, 1)
            finally:
                if os.name == 'nt':
                    shutil.rmtree(Path('\\\\?\\' + str(root)), ignore_errors=True)

    def test_same_named_videos_keep_independent_evidence_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = root / 'scene1/final_video.mp4', root / 'scene2/final_video.mp4'
            delivery = Delivery(videos=[Candidate(MediaInfo(p, True)) for p in (a, b)])
            run = Run(root, 'h', 'case', 'm', Target())
            bus = EvidenceBus(run, delivery, root / 'out')
            self.assertNotEqual(bus._vdir(a), bus._vdir(b))

    def test_changed_media_arguments_references_and_failed_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "detector_fixture" / "run.py"
            script.parent.mkdir()
            script.write_text("# isolated cache test entrypoint\n")
            video, ref = root / 'ep.mp4', root / 'ref.png'
            video.write_bytes(b'original')
            ref.write_bytes(b'reference')

            def compute(cmd, **kw):
                Path(cmd[cmd.index('--out') + 1]).write_text('{"valid":true}', encoding='utf8')
                return SimpleNamespace(returncode=0, stderr='', stdout='')

            with patch.dict(runners.SCRIPTS, {'visual': script}), \
                 patch.object(runners, '_interpreter', return_value=os.sys.executable), \
                 patch.object(runners.subprocess, 'run', side_effect=compute) as execute:
                def run(**options):
                    return runners.run_module('visual', video, root / 'out', reference_images=[ref], **options)
                self.assertTrue(run().ok)
                self.assertEqual(run().seconds, 0)
                self.assertEqual(execute.call_count, 1)
                video.write_bytes(b'changed video')
                self.assertTrue(run().ok)
                ref.write_bytes(b'changed reference')
                self.assertTrue(run().ok)
                self.assertTrue(run(device='cpu').ok)
                self.assertEqual(execute.call_count, 4)
                execute.side_effect = lambda *a, **kw: SimpleNamespace(returncode=1, stderr='broken', stdout='')
                video.write_bytes(b'another changed video')
                result = run()
                self.assertFalse(result.ok)
                self.assertIn('broken', result.error)
                self.assertTrue((root / 'out/visual_r2.json').is_file())


if __name__ == '__main__':
    unittest.main()
