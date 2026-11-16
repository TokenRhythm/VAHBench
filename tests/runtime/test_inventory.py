"""Material accounting: the numbers behind "delivered, just never assembled".

Every case here is a shape that actually occurred in `data/` and got the count
wrong at some point:

  * vimax        scene assemblies beside the shots that compose them
  * openclaw     the whole renders/ tree copied one directory deeper
  * univa        one render saved seven times while polling
  * videoclaw    episodes beside the segments they were cut from

The failure these guard against is silent: an inflated total turns a run that
produced 64% of the brief into one that reports 96% and passes.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


from stage2_delivery.discover import Candidate
from stage2_delivery.discover import EP_NUM
from stage2_delivery.discover import _distinct
from stage2_delivery.discover import _material_seconds
from stage2_delivery.probe import MediaInfo  # noqa: E402

ROOT = Path("C:/run") if os.name == "nt" else Path("/run")


def clip(rel: str, duration: float, size: int = 0) -> Candidate:
    """A candidate at `rel` under ROOT, with the episode index the locator reads."""
    p = ROOT / rel
    c = Candidate(info=MediaInfo(path=p, ok=True, duration=duration,
                                 size_bytes=size or int(duration * 1_000_000)))
    m = EP_NUM.search(p.name) or EP_NUM.search(p.parent.name)
    if m:
        c.episode = int(m.group(1))
    return c


class TestMaterial(unittest.TestCase):

    def test_assembly_replaces_the_shots_below_it(self):
        """vimax: scene_1/final_video.mp4 stands for its ten shots."""
        cands = [clip("working/scene_1/final_video.mp4", 48.0)]
        cands += [clip(f"working/scene_1/shots/{i}/video.mp4", 8.0) for i in range(10)]
        self.assertEqual(_material_seconds(ROOT, cands), 48.0)

    def test_scenes_without_episode_numbers_add_up_per_directory(self):
        """Every scene is named final_video.mp4; they are still distinct scenes."""
        cands = [clip("working/scene_1/final_video.mp4", 48.0),
                 clip("working/scene_2/final_video.mp4", 112.0),
                 clip("working/scene_3/final_video.mp4", 40.0)]
        self.assertEqual(_material_seconds(ROOT, cands), 200.0)

    def test_duplicated_tree_collapses_by_episode_index(self):
        """openclaw copied renders/ one level deeper; ep1 is ep1 either way."""
        one = [clip(f"artifacts/renders/ep{i}.mp4", 192.0) for i in (1, 2, 3)]
        two = [clip(f"artifacts/case/renders/ep{i}.mp4", 192.0) for i in (1, 2, 3)]
        self.assertEqual(_material_seconds(ROOT, one + two), 576.0)

    def test_segments_do_not_add_to_the_episodes_cut_from_them(self):
        """videoclaw: 27 segments beside output/ep1..ep3."""
        eps = [clip(f"video/out/x_ep{i}.mp4", d)
               for i, d in ((1, 71.0), (2, 84.0), (3, 101.0))]
        segs = [clip(f"video/seg_01_{i:02d}.mp4", 12.0) for i in range(27)]
        self.assertEqual(_material_seconds(ROOT, eps + segs), 256.0)

    def test_clips_only_when_nothing_was_assembled(self):
        """univa's timestamped filenames match no assembly hint."""
        cands = [clip(f"artifacts/2026091100{i}/09110{i}.mp4", 5.0) for i in range(4)]
        self.assertEqual(_material_seconds(ROOT, cands), 20.0)

    def test_distinct_drops_byte_identical_repeats(self):
        """One render polled and saved seven times is one piece of material."""
        same = [clip(f"artifacts/t{i}/0911{i}.mp4", 50.6, size=27047031)
                for i in range(7)]
        other = clip("artifacts/t9/other.mp4", 50.6, size=12345678)
        kept = _distinct(same + [other])
        self.assertEqual(len(kept), 2)

    def test_a_short_assembly_set_is_still_the_material(self):
        """vimax prompt_09: four scenes totalling 234 s against a 300 s episode.

        A floor of "assemblies must cover one episode" was tried here and sent
        the total back to the double-counted shots. Below one episode is a
        finding about the run, not a reason to count its sources again.
        """
        cands = [clip(f"working/scene_{i}/final_video.mp4", d)
                 for i, d in enumerate((8.0, 80.0, 72.0, 72.0))]
        cands += [clip(f"working/scene_1/shots/{i}/video.mp4", 8.0) for i in range(60)]
        self.assertEqual(_material_seconds(ROOT, cands), 232.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
