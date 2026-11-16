"""Accuracy and ordering regressions, independent of model outputs."""

from shared.paths import REPO_ROOT
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(REPO_ROOT / 'detectors/spec_checks'))
from shared.types import Measurement
from shared.types import Scale
from shared.types import Verdict
from stage5_scoring import check_score
from stage5_scoring import DEFAULT_CUT_POINTS
from stage5_scoring import ledger_from_reports
from stage3_evidence.collector import EvidenceBus


class RankingRepairs(unittest.TestCase):
    def test_defect_counts_keep_their_order_above_twenty_percent(self):
        for key in ('seam.cut_continuity', 'physical.penetration'):
            for verdict in (Verdict.PASS, Verdict.FAIL):
                got = [check_score(Measurement(key, verdict, Scale.RATE, rate), DEFAULT_CUT_POINTS)
                       for rate in (0, .2, .5, 1)]
                self.assertEqual(got, [1, .8, .5, 0])


    def test_directory_scoped_clips_do_not_collide(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            finals = []
            for scene in ('scene_0', 'scene_1'):
                folder = root / scene
                shot = folder / 'shots/0'
                shot.mkdir(parents=True)
                for file in (folder / 'final_video.mp4', shot / 'video.mp4', shot / 'first_frame.png'):
                    file.touch()
                (shot / 'shot_description.json').write_text('{"idx": 0}')
                (shot / 'first_frame_selector_output.json').write_text(json.dumps({
                    'text_prompt': 'a specific frame',
                    'reference_image_path_and_text_pairs': [['character.png', 'reference']]}))
                finals.append(folder / 'final_video.mp4')
            bus = EvidenceBus(NS(root=root, run_id='test'),
                              NS(found=True, videos=[NS(info=NS(path=p), episode=None) for p in finals]),
                              root / 'out')
            a, b = [bus.keyframes_for_clips(p)[1] for p in finals]
            self.assertNotEqual(a, b)
            self.assertEqual(a, root / 'scene_0/shots/0/first_frame.png')
            # Existing/reused first frames need no additional selector call.
            (root / 'scene_0/shots/0/first_frame_selector_output.json').unlink()
            (root / 'scene_0/shots/0/shot_description.json').write_text(
                '{"idx": 0, "ff_desc": "Continue the previous shot first frame"}')
            self.assertEqual(bus.keyframes_for_clips(finals[0])[1], a)

    def test_ambiguous_name_does_not_credit_arbitrary_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ('a', 'b', 'clips'):
                (root / folder).mkdir()
            for file in ('a/ep1_c01.png', 'b/ep1_c01.png', 'clips/ep1_c01.mp4', 'ep1.mp4'):
                (root / file).touch()
            final = root / 'ep1.mp4'
            bus = EvidenceBus(NS(root=root, run_id='test'),
                              NS(found=True, videos=[NS(info=NS(path=final), episode=1)]), root / 'out')
            self.assertEqual(bus.keyframes_for_clips(final), {})


if __name__ == '__main__':
    unittest.main()
