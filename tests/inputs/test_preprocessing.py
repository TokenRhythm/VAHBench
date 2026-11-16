"""Input cleanup must reduce noise without losing checklist evidence."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from stage1_inputs.preprocess import InputPreprocessor
import stage1_inputs.extractors as e


class TestPreparation(unittest.TestCase):
    def test_unrecognised_progress_payload_is_retained(self):
        row = {'type': 'heartbeat', 'new_payload': {'final_file': 'ep1.mp4'}}
        part = InputPreprocessor().prepare('stdout.log', json.dumps(row))
        self.assertEqual(json.loads(part.partition('\n')[2]), row)

    def test_structured_stdout_drops_progress_but_keeps_errors_and_declarations(self):
        rows = [
            {'type': 'system', 'subtype': 'thinking_tokens', 'count': 7},
            {'type': 'system', 'subtype': 'api_retry', 'error': 'quota exceeded'},
            {'type': 'system', 'subtype': 'thinking_tokens', 'error': 'unexpected'},
            {'type': 'assistant', 'content': 'Delivered ep1.mp4; audioOk=false'},
            {'type': 'tool_result', 'id': 'a', 'result': {'status': 'queued'}},
            {'type': 'tool_result', 'id': 'a', 'result': {'status': 'failed'}},
            {'type': 'unrecognised', 'content': 'keep me'},
        ]
        cleaner = InputPreprocessor()
        part = cleaner.prepare('stdout.log', '\n'.join(map(json.dumps, rows)))
        self.assertEqual([json.loads(s) for s in part.splitlines()[1:]], rows[1:])
        self.assertEqual(cleaner.audit['files'][0]['removed_progress'], [1])
        self.assertEqual(cleaner.audit['files'][0]['output_lines'], list(range(2, 8)))

    def test_binary_payload_is_replaced_but_media_and_request_identity_survive(self):
        blob = 'QUJD' * 1000
        row = {'id': 'shot-2', 'image_url': 'https://example.test/reference.png',
               'prompt': 'Maya in navy', 'duration': 10,
               'image': {'type': 'base64', 'media_type': 'image/png', 'data': blob},
               'file': {'base64': blob}, 'note': 'data:image/png;base64,' + blob}
        cleaner = InputPreprocessor()
        part = cleaner.prepare('events.jsonl', json.dumps(row))
        cleaned = json.loads(part.partition('\n')[2])
        self.assertNotIn(blob, part)
        for key in ('id', 'image_url', 'prompt', 'duration'):
            self.assertEqual(cleaned[key], row[key])
        self.assertEqual(cleaned['image']['media_type'], 'image/png')
        self.assertIn('sha256=', cleaned['image']['data'])
        self.assertEqual(len(cleaner.audit['files'][0]['media']), 3)

    def test_dedup_keeps_all_source_aliases_and_never_collapses_repeated_calls(self):
        cleaner = InputPreprocessor()
        body = json.dumps({'id': 'a', 'duration': 10}) + '\n'
        body += json.dumps({'id': 'b', 'duration': 10}) + '\n'
        body += json.dumps({'id': 'b', 'duration': 10}) + '\n'
        first = cleaner.prepare('events.jsonl', body)
        second = cleaner.prepare('backup/events.jsonl', body)
        self.assertEqual(len(first.splitlines()[1:]), 3)
        self.assertIn('IDENTICAL SOURCE: events.jsonl', second)
        self.assertEqual(cleaner.audit['duplicate_sources'], {'backup/events.jsonl': 'events.jsonl'})
        # Similar but not identical documents are retained, including corrections.
        third = cleaner.prepare('updated.jsonl', body + '{"status":"failed"}\n')
        self.assertIn('"status":"failed"', third)

    def test_prose_malformed_records_and_provider_history_remain(self):
        body = '# Plan\nMaya wears navy.\n{"broken":\nfinal: only ep1 delivered\n'
        cleaner = InputPreprocessor()
        self.assertEqual(cleaner.prepare('stdout.log', body).partition('\n')[2], body)
        history = {'type': 'llm_call', 'input': {'messages': [{'content': 'only copy of contract'}]}}
        cleaned = cleaner.prepare('events.jsonl', json.dumps(history)).partition('\n')[2]
        self.assertEqual(json.loads(cleaned)['input']['messages'], history['input']['messages'])

    def test_pretty_json_preserves_non_media_values(self):
        row = {'episodes': [{'file': 'ep1.mp4', 'audioOk': False}],
               'preview': {'type': 'base64', 'data': 'QUJD' * 300}}
        cleaner = InputPreprocessor()
        cleaned = json.loads(cleaner.prepare('manifest.json', json.dumps(row, indent=2)).partition('\n')[2])
        self.assertEqual(cleaned['episodes'], row['episodes'])
        self.assertIn('OMITTED', cleaned['preview']['data'])

    def test_provider_replay_keeps_unique_history_and_new_results(self):
        first = {'role': 'user', 'content': 'contract only recorded in provider history'}
        last = {'role': 'tool', 'content': 'job-1 failed; use a different reference'}
        rows = [{'type': 'llm_call', 'id': 'call-1', 'input': {'messages': [first]}},
                {'type': 'llm_call', 'id': 'call-2', 'input': {'messages': [first, last]},
                 'output': 'retry planned'},
                {'type': 'message', 'id': 'tool-2', 'content': last['content']}]
        cleaner = InputPreprocessor()
        cleaned = [json.loads(s) for s in cleaner.prepare('events.jsonl',
                   '\n'.join(map(json.dumps, rows))).splitlines()[1:]]
        self.assertEqual(cleaned[0]['input']['messages'], rows[0]['input']['messages'])
        self.assertEqual(cleaned[1]['input']['messages'][1], last)
        ref = cleaned[1]['input']['messages'][0]['_replayed_input_message']
        self.assertEqual(ref, cleaned[0]['input']['_input_history_ids']['0'])
        self.assertEqual(cleaner.audit['files'][0]['replayed_messages'][0]['first']['location'],
                         'line:1.input.messages[0]')
        self.assertEqual(cleaned[1]['output'], 'retry planned')
        self.assertEqual(cleaned[2], rows[2])
        self.assertEqual(len(cleaner.audit['files'][0]['replayed_messages']), 1)

    def test_media_uri_never_consumes_the_next_prose_line(self):
        cleaner = InputPreprocessor()
        body = 'data:image/png;base64,QUJD\nIMPORTANT final result failed\n'
        result = cleaner.prepare('notes.md', body)
        self.assertIn('\nIMPORTANT final result failed\n', result)

    def test_default_extraction_reads_beyond_eight_chunks(self):
        parts = [f'### SOURCE: {i}.md\nchunk {i + 1}\n' + 'x' * 40 for i in range(10)]
        calls = []
        def reply(prompt):
            calls.append(prompt)
            return json.dumps({'statements': [{'field': 'claims.deviations',
                'stage': 'delivery', 'subject': None, 'value': [f'chunk {len(calls)}'],
                'evidence': [{'source': f'{len(calls) - 1}.md',
                              'quote': f'chunk {len(calls)}'}]}]})
        with patch.object(e, '_lm_parts', return_value=parts):
            f = e.lm_extract(Path('.'), model='test', revision='test', call_fn=reply,
                             max_chars=80, focus_fields=['claims.deviations'])
        self.assertEqual(len(calls), 10)
        self.assertIn('chunk 10', f.claims.deviations)
        self.assertTrue(f.lm_audit['input_complete'])


if __name__ == '__main__':
    unittest.main()
