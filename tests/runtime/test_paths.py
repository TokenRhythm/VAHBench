"""The probe must not report a readable file as absent because its path is long.

Windows' stat()/is_file() answer "no such file" past roughly 240 characters
unless the path carries the \\\\?\\ prefix, while ffprobe and cv2 open the same
file without complaint. The guard that trusted is_file() made twenty files in
`data/` invisible -- among them videoclaw's episodes, which cost three runs
their delivery verdict, and two of codex's prompt_05 episodes, which were
simply missing from its score.

Nothing here needs a real video: the failure was in the existence check, which
fires before any decoding.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


from stage2_delivery.probe import _stat
from stage2_delivery.probe import probe  # noqa: E402


def _ext(p: Path) -> Path:
    """The form Windows will accept for creating a path this deep."""
    return Path("\\\\?\\" + os.path.abspath(str(p))) if os.name == "nt" else p


class TestLongPath(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        # Pad to comfortably past the limit; each level is 40 characters.
        deep = self.tmp
        while len(str(deep)) < 300:
            deep = deep / ("d" * 39)
        _ext(deep).mkdir(parents=True, exist_ok=True)
        self.deep_file = deep / "episode_01.mp4"
        _ext(self.deep_file).write_bytes(b"not really a video, but it exists")
        self.short_file = self.tmp / "short.mp4"
        self.short_file.write_bytes(b"also exists")

    def tearDown(self):
        shutil.rmtree(_ext(self.tmp), ignore_errors=True)

    def test_the_premise_still_holds(self):
        """If pathlib ever learns to do this itself, the fallback is dead code."""
        self.assertGreater(len(str(self.deep_file)), 260)
        if os.name == "nt":
            self.assertFalse(self.deep_file.is_file(),
                             "plain is_file() unexpectedly succeeded -- long paths "
                             "may now be enabled, so re-check whether _stat is needed")

    def test_stat_sees_the_deep_file(self):
        st = _stat(self.deep_file)
        self.assertIsNotNone(st)
        self.assertEqual(st.st_size, 33)

    def test_stat_still_says_no_to_a_file_that_is_absent(self):
        self.assertIsNone(_stat(self.deep_file.parent / "nope.mp4"))
        self.assertIsNone(_stat(self.tmp / "nope.mp4"))

    def test_probe_gets_past_the_existence_guard(self):
        """Not decodable, so not ok -- but the reason must not be 'not a file'."""
        info = probe(self.deep_file)
        self.assertFalse(info.ok)
        self.assertNotEqual(info.error, "not a file")

    def test_short_paths_are_unaffected(self):
        self.assertIsNotNone(_stat(self.short_file))
        self.assertEqual(probe(self.tmp / "missing.mp4").error, "not a file")


if __name__ == "__main__":
    unittest.main(verbosity=2)
