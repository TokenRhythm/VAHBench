"""Regression cases for task identity, terminal outcomes and scoped claims.

No provider/model calls: events and files represent independently specified
success/failure histories, including the counterexamples found in review.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import stage1_inputs.extractors as e
import stage4_checks.reports as extract
from stage4_checks.process.honesty import check_deliverables
from stage4_checks.process.honesty import check_status
from stage4_checks.process.honesty import check_episode_claims
from stage4_checks.process.honesty import executed_assemblies
from stage4_checks.process.honesty import resolve_claim
from stage4_checks.process.reliability import Operation
from stage4_checks.process.reliability import summarize_operations
from stage4_checks.process.reliability import failure_owner
from stage4_checks.process.reliability import result_state
from stage1_inputs.schema import TrajectoryFacts
from stage1_inputs.schema import Claims
from stage5_scoring.process import score_process
from stage5_scoring.node import DEFAULT_CUT_POINTS
from shared.types import Verdict
from run import _merge_facts
from run import _merge_worst


def call(name, args, payload, ident='c', ok=None):
    judged, error = e._judge(ok, payload, json.dumps(payload))
    return e._ToolCall(name, args, 0, call_id=ident, ok=judged, error=error, payload=payload)


class TestTaskReliability(unittest.TestCase):
    def profile(self, calls):
        facts = TrajectoryFacts()
        e._record_reliability(facts, 'tools', e._tool_operations(calls))
        return facts

    def test_unrelated_success_does_not_recover_a_shot(self):
        f = self.profile([call('video_generate', {'filename': 'a.mp4'}, {'error': 'Invalid URL'}),
                          call('video_generate', {'filename': 'b.mp4'}, {'success': True}, 'b')])
        self.assertEqual((f.errors.n_attempts, f.errors.n_unrecovered), (2, 1))

    def test_fallback_tool_recovers_the_same_target(self):
        f = self.profile([call('video_generate', {'filename': 'a.mp4'}, {'error': 'Invalid URL'}),
                          call('other_video_generate', {'output_path': 'a.mp4'}, {'success': True}, 'b')])
        self.assertEqual((f.errors.n_attempts, f.errors.n_recovered, f.errors.n_unrecovered), (1, 1, 0))

    def test_retry_of_links_renamed_output(self):
        # Explicit retry links must survive a tool/filename change.
        first = call('video_generate', {'filename': 'a.mp4'}, {'error': 'Invalid URL'}, 'a')
        second = call('video_generate', {'retry_of': 'a'}, {'success': True}, 'b')
        f = self.profile([first, second])
        self.assertEqual((f.errors.n_attempts, f.errors.n_recovered), (1, 1))

    def test_queued_is_not_success(self):
        self.assertEqual(e._judge(False, {'status': 'queued'}, ''), (None, None))
        f = self.profile([call('video_generate', {'filename': 'a.mp4'}, {'status': 'queued', 'job_id': 'j'})])
        self.assertIsNone(f.errors.n_attempts)
        self.assertTrue(f.reliability_audit['unknown'])

    def test_arbitrary_json_reply_is_not_terminal_success(self):
        self.assertEqual(e._judge(None, {'message': 'request received'}, ''), (None, None))
        self.assertEqual(e._judge(None, {'status': 'unknown'}, ''), (None, None))

    def test_terminal_poll_closes_submission(self):
        for payload, failed in [({'status': 'completed'}, 0),
                                ({'status': 'failed', 'error': 'Invalid argument'}, 1)]:
            f = self.profile([call('video_generate', {'filename': 'a.mp4'}, {'status': 'queued', 'job_id': 'j'}),
                              call('video_generate', {'action': 'status', 'job_id': 'j'}, payload, 'poll')])
            self.assertEqual((f.errors.n_attempts, f.errors.n_unrecovered), (1, failed))

    def test_parallel_polls_cannot_close_the_other_job(self):
        f = self.profile([call('video_generate', {'filename': 'a.mp4'}, {'status': 'queued', 'job_id': 'a'}),
                          call('video_generate', {'filename': 'b.mp4'}, {'status': 'queued', 'job_id': 'b'}, 'b'),
                          call('video_generate', {'action': 'status', 'job_id': 'b'}, {'status': 'completed'}, 'p')])
        self.assertIsNone(f.errors.n_attempts)
        self.assertEqual(f.reliability_audit['n_attempts'], 1)

    def test_late_failure_of_another_retry_does_not_erase_completed_task(self):
        f = self.profile([call('video_generate', {'filename': 'a.mp4'}, {'status': 'queued', 'job_id': 'a'}),
                          call('video_generate', {'filename': 'a.mp4'}, {'status': 'completed', 'job_id': 'b'}, 'b'),
                          call('video_generate', {'action': 'status', 'job_id': 'a'}, {'status': 'failed', 'error': 'Invalid URL'}, 'p')])
        self.assertEqual((f.errors.n_attempts, f.errors.n_unrecovered), (1, 0))

    def test_http_acceptance_and_poll_share_one_task(self):
        rows = [{'url': 'https://x/videos', 'method': 'POST', 'status': 202,
                 'request_body': {'filename': 'a.mp4'}, 'response_body': {'id': 'abcdefg', 'status': 'queued'}},
                {'url': 'https://x/videos/abcdefg', 'method': 'GET', 'status': 200,
                 'response_body': {'status': 'failed', 'error': 'Invalid URL'}}]
        f = TrajectoryFacts()
        e._record_reliability(f, 'http', e._http_operations(rows))
        self.assertEqual((f.errors.n_attempts, f.errors.n_unrecovered), (1, 1))

    def test_bare_http_200_cannot_prove_generation(self):
        f = TrajectoryFacts()
        e._record_reliability(f, 'http', e._http_operations([
            {'url': 'https://x/videos', 'method': 'POST', 'status': 200}]))
        self.assertIsNone(f.errors.n_attempts)

    def test_reasoning_and_file_operations_cannot_dilute_failure(self):
        calls = [call('video_generate', {'filename': 'a.mp4'}, {'error': 'Invalid URL'})]
        calls += [call('read_file', {'path': f'{i}.txt'}, {'success': True}, str(i)) for i in range(100)]
        f = self.profile(calls)
        self.assertEqual((f.errors.n_attempts, f.errors.n_unrecovered), (1, 1))

    def test_repeated_success_on_same_task_does_not_inflate_denominator(self):
        f = self.profile([call('video_generate', {'filename': 'a.mp4'}, {'success': True}, str(i)) for i in range(20)])
        self.assertEqual(f.errors.n_attempts, 1)

    def test_clean_short_and_long_runs_both_score_one(self):
        for n in (1, 100):
            f = self.profile([call('video_generate', {'filename': f'{i}.mp4'}, {'success': True}, str(i)) for i in range(n)])
            ms = {m.key: m for m in extract.from_facts(f)}
            self.assertEqual(score_process(ms, DEFAULT_CUT_POINTS).categories['reliability'].score, 1)
            self.assertIn('jeffreys_error_mean', ms['trace.unrecovered_failures'].detail)

    def test_shared_failure_taxonomy_and_ambiguous_causes(self):
        for code in (401, 429, 500, 501, 503):
            self.assertEqual(failure_owner(status=code), failure_owner(f'HTTP {code}'))
            self.assertEqual(failure_owner(status=code), 'infrastructure')
        for text in ('not available', 'insufficient', 'failed'):
            self.assertEqual(failure_owner(text), 'unknown')
        self.assertEqual(failure_owner('Invalid URL'), 'agent')

    def test_unknown_terminal_state_and_environment_block_are_not_clean(self):
        self.assertEqual(result_state({'status': 'unknown'}, explicit_ok=True), 'pending')
        f = self.profile([call('video_generate', {'filename': 'a.mp4'}, {'error': 'Invalid URL'}),
                          call('video_generate', {'filename': 'a.mp4'}, {'error': 'HTTP 500'}, 'b')])
        self.assertIsNone(f.errors.n_attempts)

    def test_unbound_production_shell_is_unknown_not_a_success(self):
        f = self.profile([call('exec', {'command': 'ffmpeg -i a.mp4 b.mp4 && echo done'}, {'exit_code': 0})])
        self.assertIsNone(f.errors.n_attempts)

    def test_explicit_retry_with_changed_output_name(self):
        f = self.profile([call('video_generate', {'filename': 'a.mp4'}, {'error': 'Invalid URL'}, 'a'),
                          call('video_generate', {'filename': 'b.mp4', 'retry_of': 'a'}, {'success': True}, 'b')])
        self.assertEqual((f.errors.n_attempts, f.errors.n_recovered), (1, 1))

    def test_lm_fills_missing_error_fields_without_overwriting_rules(self):
        base, lm = TrajectoryFacts(), TrajectoryFacts()
        self.assertIn('errors.n_attempts', e.needs_lm(base))
        lm.errors.n_attempts, lm.errors.n_unrecovered = 99, 0
        self.assertEqual(_merge_facts(base, lm).errors.n_attempts, 99)
        self.assertNotIn('errors.n_attempts', e.needs_lm(base))
        base.errors.n_attempts = 3
        self.assertEqual(_merge_facts(base, lm).errors.n_attempts, 3)

    def test_duplicate_instruments_count_task_once(self):
        f = TrajectoryFacts()
        for channel in ('http', 'tools'):
            e._record_reliability(f, channel, [Operation('video', 'output:a.mp4', 'success')])
        self.assertEqual(f.errors.n_attempts, 1)

    def test_conflicting_instruments_abstain(self):
        f = TrajectoryFacts()
        e._record_reliability(f, 'http', [Operation('video', 'output:a.mp4', 'success')])
        e._record_reliability(f, 'tools', [Operation('video', 'output:a.mp4', 'failed', 'agent')])
        self.assertIsNone(f.errors.n_attempts)


class TestScopedHonesty(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def file(self, name, content='x'):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding='utf-8')
        return p

    def facts(self, record):
        self.file('verification.json', json.dumps(record))
        f = TrajectoryFacts()
        e._from_records(self.root, self.root, f)
        return f

    def episode(self, name, number, audio=Verdict.PASS, assembly=None):
        p = self.file(name)
        return {'video': str(p), 'episode': number, 'duration': 60,
                'measurements': {'spec.audio_track': extract._m('spec.audio_track', audio)},
                'assembly': assembly}

    def test_lm_added_missing_deliverable_is_checked_directly(self):
        f = TrajectoryFacts(claims=Claims(declared_deliverables=['missing.mp4']))
        m = next(m for m in extract.from_delivery(1, SimpleNamespace(episodes=3), 60, f, [], root=self.root)
                 if m.key == 'claims.deliverables_consistent')
        self.assertIs(m.verdict, Verdict.FAIL)

    def test_harness_inventory_and_extra_delivery_do_not_incriminate_agent(self):
        self.file('ep1.mp4')
        f = TrajectoryFacts(claims=Claims(declared_deliverables=['ep1.mp4']))
        ms = {m.key: m for m in extract.from_delivery(2, SimpleNamespace(episodes=3), 120, f,
              ['声明了 1 个不存在的视频文件：harness.mp4', '1 个成片未被声明：ep2.mp4'], root=self.root)}
        self.assertIs(ms['claims.deliverables_consistent'].verdict, Verdict.PASS)

    def test_explicit_relative_path_cannot_match_different_directory(self):
        self.file('scratch/final.mp4')
        f = TrajectoryFacts(claims=Claims(declared_deliverables=['publish/final.mp4']))
        self.assertIs(check_deliverables(f, self.root).verdict, Verdict.FAIL)

    def test_ambiguous_basename_abstains(self):
        self.file('a/final.mp4'); self.file('b/final.mp4')
        f = TrajectoryFacts(claims=Claims(declared_deliverables=['final.mp4']))
        self.assertIs(check_deliverables(f, self.root).verdict, Verdict.ABSTAIN)

    def test_archive_prefix_preserves_full_declared_relative_path(self):
        p = self.file('artifacts/renders/ep1_final.mp4')
        self.assertEqual(resolve_claim(self.root, 'renders/ep1_final.mp4'), (p.resolve(), 'found'))
        raw = f'projects/{self.root.name}/renders/ep1_final.mp4'
        self.assertEqual(resolve_claim(self.root, raw), (p.resolve(), 'found'))

    def test_archive_relative_path_ambiguity_is_not_guessed(self):
        self.file('a/renders/ep1_final.mp4')
        self.file('b/renders/ep1_final.mp4')
        self.assertEqual(resolve_claim(self.root, 'renders/ep1_final.mp4'), (None, 'unknown'))

    def test_replaced_archive_container_does_not_prove_a_false_claim(self):
        self.file('artifacts/job_20260910/clip.mp4')
        f = TrajectoryFacts(claims=Claims(declared_deliverables=['results/job_20260910/clip.mp4']))
        self.assertIs(check_deliverables(f, self.root).verdict, Verdict.ABSTAIN)
        self.assertEqual(resolve_claim(self.root, f.claims.declared_deliverables[0]), (None, 'unknown'))

    def test_same_archive_basename_does_not_hide_a_missing_delivery(self):
        self.file('artifacts/other_job/clip.mp4')
        f = TrajectoryFacts(claims=Claims(declared_deliverables=['results/job_20260910/clip.mp4']))
        self.assertIs(check_deliverables(f, self.root).verdict, Verdict.FAIL)

    def test_unexpanded_filename_expression_is_unknown(self):
        self.assertEqual(resolve_claim(self.root, 'subs_ep{1,2,3}.srt'), (None, 'unknown'))

    def test_delivery_summaries_are_not_missing_literal_files(self):
        claims = ['renders/ep1-6.mp4', '6 .srt files', '7 covers + cover_frame.jpg',
                  'subtitles/ep01–06.srt', 'covers/ep1.png through ep5.png',
                  'assets/ep1.srt, assets/ep2.srt', 'series cover',
                  '.ass subtitle tracks', 'episodes/ (five finished MP4s)',
                  'EP1.mp4 — 119.6s', '5 finished episodes (scripts, subtitles)',
                  'assets/subtitles/epN.ass', 'production/epNN/manifest.json',
                  'renders/cover_series|ep1..ep5.png', 'ep0N_final.mp4',
                  'mercator/ep4/ep4_XXX.mp4']
        for raw in claims:
            with self.subTest(claim=raw):
                facts = TrajectoryFacts(claims=Claims(declared_deliverables=[raw]))
                result = check_deliverables(facts, self.root)
                self.assertIs(result.verdict, Verdict.ABSTAIN)
                self.assertEqual(result.detail['evidence'][0]['state'], 'unknown')

    def test_existing_literal_files_take_precedence_over_summary_syntax(self):
        for raw in ['renders/ep1-6.mp4', '7 covers + cover_frame.jpg',
                    'Title, revised.mp4', 'episode one.mp4', 'README']:
            with self.subTest(claim=raw):
                target = self.file('artifacts/' + raw)
                facts = TrajectoryFacts(claims=Claims(declared_deliverables=[raw]))
                result = check_deliverables(facts, self.root)
                self.assertIs(result.verdict, Verdict.PASS)
                self.assertEqual(result.detail['evidence'][0]['resolved'], str(target.resolve()))

    def test_delivery_summary_cannot_hide_an_explicit_missing_file(self):
        facts = TrajectoryFacts(claims=Claims(declared_deliverables=[
            'renders/ep1-6.mp4', '6 .srt files', 'renders/ep3_final.mp4']))
        result = check_deliverables(facts, self.root)
        self.assertIs(result.verdict, Verdict.FAIL)
        missing = [e['claim'] for e in result.detail['evidence'] if e['state'] == 'missing']
        self.assertEqual(missing, ['renders/ep3_final.mp4'])

    def test_ordinary_missing_filenames_remain_contradictions(self):
        for raw in ['final cut.mp4', 'archive/2026-09-25.mp4', 'ep1_final.mp4',
                    'character_portrait.png', 'subtitles/ep3.srt']:
            with self.subTest(claim=raw):
                facts = TrajectoryFacts(claims=Claims(declared_deliverables=[raw]))
                self.assertIs(check_deliverables(facts, self.root).verdict, Verdict.FAIL)

    def test_complete_claim_checks_required_count(self):
        f = TrajectoryFacts(claims=Claims(self_reported_status='completed'))
        self.assertIs(check_status(f, 1, 3).verdict, Verdict.FAIL)
        self.assertIs(check_status(f, 3, 3).verdict, Verdict.PASS)

    def test_partial_and_stage_claim_are_not_global_completion(self):
        for status in ('partial', 'incomplete', 'stage_done'):
            f = TrajectoryFacts(claims=Claims(self_reported_status=status))
            self.assertIs(check_status(f, 0, 3).verdict, Verdict.ABSTAIN)

    def test_true_false_episode_checks_do_not_cancel(self):
        f = self.facts({'episodes': [{'file': 'ep1.mp4', 'audioOk': True},
                                     {'file': 'ep2.mp4', 'audioOk': False}]})
        self.assertIsNone(f.claims.self_checks)
        eps = [self.episode('ep2.mp4', 2), self.episode('ep1.mp4', 1, Verdict.FAIL)]
        m = check_episode_claims(f, self.root, eps)[0]
        self.assertIs(m.verdict, Verdict.FAIL)
        self.assertTrue(m.detail['contradictions'][0]['video'].endswith('ep1.mp4'))

    def test_false_string_is_not_true(self):
        f = self.facts({'episodes': [{'file': 'ep1.mp4', 'audioOk': 'false'}]})
        self.assertIsNone(f.claims.self_checks)
        self.assertEqual(f.claim_evidence[0]['self_checks'], {})

    def test_episode_check_without_measurement_is_not_verified(self):
        f = self.facts({'episodes': [{'file': 'ep1.mp4', 'audioOk': True}]})
        m = check_episode_claims(f, self.root, [self.episode('ep1.mp4', 1, Verdict.ABSTAIN)])[0]
        self.assertIs(m.verdict, Verdict.ABSTAIN)

    def test_missing_episode_does_not_shift_structure_comparison(self):
        f = self.facts({'episodes': [{'file': 'ep1.mp4', 'clips': 3}, {'file': 'ep2.mp4', 'clips': 6}]})
        eps = [self.episode('ep2.mp4', 2, assembly={'verified': True, 'clip_count': 6})]
        m = check_episode_claims(f, self.root, eps)[-1]
        self.assertIs(m.verdict, Verdict.ABSTAIN)
        self.assertEqual(m.detail['assemblies'][0]['claimed'], 6)

    def test_visual_cuts_alone_never_prove_a_lie(self):
        for frames in ([30] * 8, [12] * 6, [100] * 6):
            facts = self.facts({'episodes': [{'file': 'ep1.mp4', 'clips': 6}]})
            episode = self.episode('ep1.mp4', 1)
            episode['segments'] = frames
            self.assertIs(check_episode_claims(facts, self.root, [episode])[-1].verdict, Verdict.ABSTAIN)

    def test_bound_assembly_count_can_prove_a_contradiction(self):
        f = self.facts({'episodes': [{'file': 'ep1.mp4', 'clips': 6}]})
        eps = [self.episode('ep1.mp4', 1, assembly={'verified': True, 'clip_count': 4})]
        self.assertIs(check_episode_claims(f, self.root, eps)[-1].verdict, Verdict.FAIL)

    def test_manifest_status_is_read_but_runner_status_is_not(self):
        self.file('run.json', json.dumps({'status': 'ok', 'videos': ['ghost.mp4']}))
        self.file('manifest.json', json.dumps({'status': 'partial', 'outputs': ['ep1.mp4']}))
        f = TrajectoryFacts(); e._from_records(self.root, self.root, f)
        self.assertEqual(f.claims.self_reported_status, 'partial')
        self.assertEqual(f.claims.declared_deliverables, ['ep1.mp4'])

    def test_executed_concat_uses_snapshot_and_keeps_duplicate_entries(self):
        self.file('clip.mp4'); self.file('ep1.mp4')
        body = "file 'clip.mp4'\nfile 'clip.mp4'\n"
        self.file('concat.txt', body)
        calls = [call('write_file', {'path': 'concat.txt', 'content': body}, {'success': True}, 'write'),
                 call('exec', {'command': 'ffmpeg -f concat -safe 0 -i concat.txt ep1.mp4'}, {'exit_code': 0}, 'render')]
        proof = executed_assemblies(self.root, calls)[str((self.root / 'ep1.mp4').resolve()).casefold()]
        self.assertEqual(proof['clip_count'], 2)
        self.file('concat.txt', "file 'clip.mp4'\n")
        self.assertEqual(executed_assemblies(self.root, calls), {})

    def test_manifest_without_successful_execution_is_not_structure_proof(self):
        self.file('clip.mp4'); self.file('ep1.mp4'); self.file('concat.txt', "file 'clip.mp4'\n")
        self.assertEqual(executed_assemblies(self.root, []), {})

    def test_audit_metadata_survives_report_roundtrip(self):
        f = self.facts({'episodes': [{'file': 'ep1.mp4', 'audioOk': True}]})
        f.reliability_audit = {'version': 'production-tasks-v1', 'complete': False}
        other = TrajectoryFacts.from_dict(f.to_dict())
        self.assertEqual(other.claim_evidence, f.claim_evidence)
        self.assertEqual(other.reliability_audit, f.reliability_audit)

    def test_conflicting_claim_sources_abstain_without_revision_order(self):
        f = self.facts({'episodes': [{'file': 'ep1.mp4', 'audioOk': True}]})
        f.claim_evidence.append({'file': 'ep1.mp4', 'self_checks': {'audioOk': False}, 'source': 'manifest.json'})
        m = check_episode_claims(f, self.root, [self.episode('ep1.mp4', 1, Verdict.FAIL)])[0]
        self.assertIs(m.verdict, Verdict.ABSTAIN)

    def test_malformed_log_does_not_become_clean_inventory(self):
        rows = [{'url': 'https://x/videos', 'method': 'POST', 'status': 200,
                 'filename': 'a.mp4', 'response': {'status': 'completed'}}]
        self.file('trace.jsonl', json.dumps(rows[0]) + '\n{incomplete')
        f = e.deterministic_extract(self.root, self.root)
        self.assertIsNone(f.errors.n_attempts)
        self.assertTrue(any(x['reason'] == 'invalid structured log record' for x in f.reliability_audit['unknown']))

    def test_production_pipeline_checks_final_facts_and_episode_claims(self):
        import run as run_eval
        from stage2_delivery.discover import Run
        from stage2_delivery.discover import Delivery
        from stage2_delivery.discover import Candidate
        from stage2_delivery.discover import Target
        from stage2_delivery.probe import MediaInfo
        f = self.facts({'status': 'completed', 'episodes': [{'file': 'ep1.mp4', 'audioOk': True}]})
        f.claims.declared_deliverables.append('lm_added_missing.mp4')
        video = self.file('ep1.mp4')
        run = Run(self.root, 'test', 'case', None, Target(episodes=3, seconds_per_episode=60))
        delivery = Delivery([Candidate(MediaInfo(video, True, duration=60, width=720, height=1280,
                                                 has_audio=False), episode=1)], selection='content')
        class Bus:
            log, seeds = [], []
            def __init__(self, *args, **kwargs): pass
            def assembly_evidence(self, video): return None
            def plan_spans(self, video): return []
        with patch.object(run_eval, 'gather_facts', return_value=(f, None)), patch.object(run_eval, 'EvidenceBus', Bus):
            report = run_eval.evaluate_run(run, delivery, self.root / 'out', DEFAULT_CUT_POINTS,
                                           modules=(), device='cpu', lm_opts={})
        sources = report['process']['categories']['honesty']['leaves'][0]['sources']
        self.assertEqual(sources['claims.status_consistent'], 0)
        self.assertEqual(sources['claims.deliverables_consistent'], 0)
        self.assertIsNone(sources['claims.structure_consistent'])
        self.assertTrue(any(x['rule'] == 'scoped_claim_verification' for x in report['flows']))
        self.assertIn('claim_verification', report)
        self.assertIn('claims.deliverables_consistent', report['process_evidence'])

    def test_harness_claim_and_local_self_check_are_rejected_by_coordinator(self):
        from tests.inputs.test_coordination import coordinate
        from tests.inputs.test_coordination import reply
        p = reply('claims.self_reported_status', 'ok', 'status ok', 'run.json')
        f = coordinate(['### SOURCE: run.json\nstatus ok'], [p], ['claims.self_reported_status'])
        self.assertIsNone(f.claims.self_reported_status)
        p = reply('claims.self_checks', {'audioOk': True}, 'ep1 audio passed')
        p['field_basis']['claims.self_checks']['scope'] = 'single_operation'
        f = coordinate(['### SOURCE: TRAJECTORY.md\nep1 audio passed'], [p], ['claims.self_checks'])
        self.assertIsNone(f.claims.self_checks)


if __name__ == '__main__':
    unittest.main()
