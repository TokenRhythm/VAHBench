"""The reference detectors against real FFmpeg, on encoded files with known content.

These are the only tests that exercise FFmpeg's own decoding (frame
pass-through, scaling, range conversion, codec noise). They need `ffmpeg` and
`ffprobe` on PATH, which the evaluator requires anyway. Where the tools are
absent the tests are skipped with that reason -- they are an environment
prerequisite, not an optional feature, and CI installs them.

The clips are the same synthetic scenes the array-level tests use, encoded
here, so a disagreement between the two layers points at the decoding.
"""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

import stage4_checks.dispatch as runners
from stage5_scoring import ledger_from_reports
from tests.reference import synth

synth.use_reference("spec", "artifact", "visual", "av", "seam")
import artifacts  # noqa: E402
import run_audio_check as av  # noqa: E402
from speccheck.report import check_video  # noqa: E402
from vahref import features as ft  # noqa: E402
from vahref import media  # noqa: E402
from vcheck.repeats import scan_video  # noqa: E402

FFMPEG, FFPROBE = shutil.which("ffmpeg"), shutil.which("ffprobe")
FPS = 24


def _encoder() -> list[str]:
    listed = subprocess.run([FFMPEG, "-hide_banner", "-encoders"], capture_output=True,
                            encoding="utf-8", errors="replace").stdout
    if "libx264" in listed:
        return ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "12", "-pix_fmt", "yuv420p"]
    return ["-c:v", "mpeg4", "-q:v", "2", "-g", "300", "-pix_fmt", "yuv420p"]


def encode(path: Path, frames: np.ndarray, fps: float = FPS, audio=None, sr: int = 48000,
           extra: list[str] | None = None) -> Path:
    """Write grey frames (n, 64, 64) as a 256x256 video, optionally with PCM audio."""
    big = np.kron(frames, np.ones((1, 4, 4), dtype=np.uint8))
    raw = path.with_suffix(".gray")
    raw.write_bytes(big.tobytes())
    cmd = [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", "256x256",
           "-r", str(fps), "-i", str(raw)]
    if audio is not None:
        pcm = path.with_suffix(".pcm")
        pcm.write_bytes(np.asarray(audio, dtype="<i2").tobytes())
        cmd += ["-f", "s16le", "-ar", str(sr), "-ac", str(audio.shape[1]), "-i", str(pcm),
                "-c:a", "pcm_s16le"]
    cmd += [*_encoder(), *(extra or []), str(path)]
    subprocess.run(cmd, check=True, capture_output=True)
    raw.unlink()
    return path


def tone(seconds, amplitude=0.4, sr=48000):
    t = np.arange(int(sr * seconds)) / sr
    x = np.clip(amplitude * np.sin(2 * np.pi * 220 * t), -1, 1)
    return np.round(np.repeat(x[:, None], 2, axis=1) * 32767).astype(np.int16)


@unittest.skipUnless(FFMPEG and FFPROBE, "ffmpeg and ffprobe are required on PATH")
class RealMedia(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_every_encoded_frame_is_decoded_once(self):
        path = encode(self.tmp / "plain.mp4", synth.scene(120, 1))
        meta = media.probe(path)
        self.assertEqual(meta["n_frames"], 120)
        self.assertAlmostEqual(meta["video_duration"], 5.0, delta=0.05)
        self.assertEqual(ft.from_video(path).n, 120)

    def test_a_container_without_a_frame_count_is_counted(self):
        path = encode(self.tmp / "plain.mkv", synth.scene(96, 2), audio=tone(4.0))
        meta = media.probe(path)
        self.assertEqual(meta["n_frames"], 96)
        self.assertTrue(meta["has_audio"])

    def test_clean_footage_survives_encoding_without_a_finding(self):
        rep = artifacts.analyse(encode(self.tmp / "clean.mp4", synth.scene(240, 3)))
        for key in ("black_frames", "solid_frames", "frozen_runs", "tail_freeze"):
            self.assertFalse(rep["findings"][key]["present"], key)
        self.assertEqual(rep["splices"]["n_splices"], 0)

    def test_held_black_and_cut_are_found_after_encoding(self):
        a, b = synth.scene(96, 4), synth.scene(96, 5)
        frames = np.concatenate([a, np.repeat(a[-1:], 72, axis=0),         # 3 s held
                                 np.zeros((48, 64, 64), np.uint8), b])     # 2 s black, then a new scene
        rep = artifacts.analyse(encode(self.tmp / "defects.mp4", frames))
        f = rep["findings"]
        # The held picture is frames 96-167 (4 s to 7 s). The black stretch
        # after it is identical frames too and forms a held run of its own;
        # the ledger takes the union, so it is charged once.
        held = [s for s in f["frozen_runs"]["segments"] if s["end_s"] <= 7.2]
        self.assertAlmostEqual(sum(s["end_s"] - s["start_s"] for s in held), 3.0, delta=0.3)
        self.assertAlmostEqual(f["black_frames"]["n_frames"], 48, delta=3)
        self.assertGreaterEqual(rep["splices"]["n_splices"], 1)
        duration = rep["meta"]["duration"]
        led = ledger_from_reports("v", 1, duration, duration, artifact=rep)
        self.assertAlmostEqual(led.deductions_s, 5.0, delta=0.5)

    def test_a_loop_is_found_and_distinct_footage_is_not(self):
        clip = synth.scene(144, 6)
        looped = scan_video(encode(self.tmp / "loop.mp4", np.concatenate([clip, clip])))
        self.assertEqual(looped["abstained"], "")
        self.assertGreaterEqual(looped["duplicate_seconds"], 4.0)
        plain = scan_video(encode(self.tmp / "noloop.mp4",
                                  np.concatenate([clip, synth.scene(144, 7)])))
        self.assertEqual(plain["duplicate_seconds"], 0.0)

    def test_a_frame_preserving_slowdown_is_verified_against_its_source(self):
        clip = synth.scene(240, 8)
        source = encode(self.tmp / "source.mp4", clip)
        slow = encode(self.tmp / "slow.mp4", clip, fps=12)
        other = encode(self.tmp / "other.mp4", synth.scene(240, 9))
        same = check_video(str(slow), source_clips=[str(source)])["checks"]["retiming"]
        self.assertFalse(same["ok"])
        self.assertTrue(same["correspondence_verified"])
        self.assertAlmostEqual(same["final_duration"] - same["source_duration"], 10.0, delta=0.2)
        different = check_video(str(slow), source_clips=[str(other)])["checks"]["retiming"]
        self.assertTrue(different["undetermined"])
        self.assertFalse(different["correspondence_verified"])

    def test_audio_outlasting_the_picture_is_a_no_picture_interval(self):
        path = encode(self.tmp / "overhang.mkv", synth.scene(120, 10), audio=tone(9.0))
        rep = check_video(str(path))
        meta = rep["meta"]
        self.assertAlmostEqual(meta["video_duration"], 5.0, delta=0.1)
        led = ledger_from_reports("v", 1, 9.0, meta["duration"], rep)
        self.assertAlmostEqual(led.no_picture_s, 4.0, delta=0.2)

    def test_clipping_and_silence_and_absence(self):
        frames = synth.scene(48, 11)
        clean = av.check(encode(self.tmp / "clean.mkv", frames, audio=tone(2.0)))
        self.assertEqual(clean["cleanliness"]["n_clipped_runs"], 0)
        loud = av.check(encode(self.tmp / "loud.mkv", frames, audio=tone(2.0, amplitude=3.0)))
        self.assertGreater(loud["cleanliness"]["n_clipped_runs"], 50)
        silent = av.check(encode(self.tmp / "silent.mkv", frames,
                                 audio=np.zeros((96000, 2), np.int16)))
        self.assertEqual(silent["cleanliness"]["rms"], 0.0)
        absent = av.check(encode(self.tmp / "mute.mp4", frames))
        self.assertFalse(absent["audio_stream"]["present"])
        self.assertTrue(av.audio_stream_present(self.tmp / "clean.mkv"))
        self.assertEqual(len(av.extract_audio(self.tmp / "clean.mkv")), av.SR * 2)
        with self.assertRaises(RuntimeError):
            av.extract_audio(self.tmp / "mute.mp4")

    def test_modules_run_through_the_dispatcher(self):
        frames = np.concatenate([synth.scene(96, 12), synth.scene(96, 13)])
        video = encode(self.tmp / "dispatch.mkv", frames, audio=tone(8.0))
        runners._probe_cache.clear()
        for module, kw in (("spec", {}), ("artifact", {}), ("visual", {"device": "cpu"}),
                           ("av", {"device": "cpu"}), ("seam", {"seam_times": [4.0]})):
            with self.subTest(module=module):
                if runners.resolve_script(module)[1] != "reference":
                    self.skipTest("an original detector is installed for this module")
                got = runners.run_module(module, video, self.tmp / "out", **kw)
                self.assertTrue(got.ok, got.error)
                self.assertEqual(got.origin, "reference")
                self.assertFalse(got.report["implementation"]["original_detector"])


if __name__ == "__main__":
    unittest.main()
