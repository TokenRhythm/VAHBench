"""Contracts of the specification detector, on whichever one is installed.

With the paper's original `detectors/spec_checks` present these run against
it; otherwise against the bundled reference reimplementation. The inputs are
synthetic metadata or mocked probe results, so no media, model weight or
corpus is involved and nothing here is optional: a missing implementation is
an import error, not a skip.
"""
import os
import sys
import unittest
from unittest.mock import patch
from shared.paths import REPO_ROOT
from stage5_scoring import ledger_from_reports, score_completeness, DEFAULT_CUT_POINTS


class DetectorContracts(unittest.TestCase):
    def score(self, ledgers, **kwargs):
        return score_completeness(ledgers, len(ledgers), DEFAULT_CUT_POINTS, **kwargs)

    @classmethod
    def setUpClass(cls):
        from stage4_checks.dispatch import detector_dir
        sys.path.insert(0, str(detector_dir("spec")))
        global check_retiming, check_video, probe
        from speccheck.timing import check_retiming
        from speccheck.report import check_video
        from speccheck.probe import probe

    def test_same_count_is_not_same_footage(self):
        rt = check_retiming(dict(n_frames=100, video_duration=10),
                            [dict(n_frames=100, video_duration=5)])
        self.assertTrue(rt['undetermined'])
        self.assertFalse(rt['correspondence_verified'])
        # Stale metadata-only reports cannot sneak a deduction past the bus.
        spec = dict(meta=dict(video_duration=10), checks=dict(retiming=dict(
            ok=False, source_duration=5, final_duration=10)))
        self.assertEqual(ledger_from_reports('a', 1, 10, 10, spec).retime_excess_s, 0)


    def test_audio_overhang_is_not_retiming(self):
        final = dict(n_frames=1441, fps=24, duration=82.788, video_duration=60.041667)
        source = dict(n_frames=1441, fps=24, duration=60.1, video_duration=60.041667)
        rt = check_retiming(final, [source])
        self.assertTrue(rt['ok'])
        led = ledger_from_reports('a', 1, 60, 82.788,
                                  dict(meta=final, checks=dict(retiming=rt)))
        self.assertAlmostEqual(led.no_picture_s, 22.746333)
        self.assertEqual(led.retime_excess_s, 0)
        self.assertAlmostEqual(self.score([led]).genuine_s, 60.04)


    def test_missing_source_probe_does_not_look_like_complete_inventory(self):
        meta = dict(path='a', n_frames=240, fps=24, video_duration=10,
                    duration=10, width=720, height=1280)
        with patch('speccheck.report.probe', side_effect=[meta, meta, FileNotFoundError()]):
            report = check_video('a', source_clips=['s1', 's2'])
        self.assertTrue(report['checks']['retiming']['undetermined'])


    def test_probe_does_not_invent_frames_from_audio_duration(self):
        data = dict(format=dict(duration='80'), streams=[dict(codec_type='video',
                    avg_frame_rate='24/1', width=720, height=1280)])
        with patch('speccheck.probe.exists', return_value=True), patch(
                'speccheck.probe._ffprobe', return_value=data):
            meta = probe('fixture')
        self.assertEqual(meta['n_frames'], 0)
        self.assertIsNone(meta['video_duration'])

