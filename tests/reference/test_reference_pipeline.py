"""Dispatch -> subprocess -> report -> ledger -> score, with the reference
detectors run as real child processes.

FFmpeg itself is replaced by the stand-ins in fake_media.py so the chain can
be exercised on any POSIX machine. That covers the wiring this repository
owns. It does not cover FFmpeg's decoding; see test_reference_media.py.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import stage4_checks.dispatch as runners
import shared.preflight as preflight
from run import evaluate_run
from shared.settings import ALL_MODULES
from stage1_inputs.schema import TrajectoryFacts
from stage2_delivery.discover import Candidate
from stage2_delivery.discover import Delivery
from stage2_delivery.discover import Run
from stage2_delivery.discover import Target
from stage2_delivery.probe import probe as evaluator_probe
from stage3_evidence.collector import EvidenceBus
from stage5_scoring import DEFAULT_CUT_POINTS
from tests.reference import fake_media
from tests.reference import synth

FPS = synth.FPS


def tone(seconds, amplitude=0.4, sr=48000):
    t = np.arange(int(sr * seconds)) / sr
    x = amplitude * np.sin(2 * np.pi * 220 * t)
    return np.round(np.repeat(x[:, None], 2, axis=1) * 32767).astype(np.int16)


@unittest.skipIf(os.name == "nt", "stand-in executables are POSIX scripts; "
                 "the real-FFmpeg tests cover Windows")
class ReferencePipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        tools = fake_media.install(self.tmp / "bin")
        env = patch.dict(os.environ, {"PATH": fake_media.path_with(tools)})
        env.start()
        self.addCleanup(env.stop)
        # The evaluator's probe resolved ffprobe at import time.
        probe_path = patch("stage2_delivery.probe.FFPROBE", str(tools / "ffprobe"))
        probe_path.start()
        self.addCleanup(probe_path.stop)
        runners._probe_cache.clear()
        self.addCleanup(runners._probe_cache.clear)
        self.root = self.tmp / "data" / "h" / "m" / "prompt_01_case"

    def delivery(self, **videos):
        cands = []
        for i, (name, spec) in enumerate(sorted(videos.items()), 1):
            path = fake_media.write_video(self.root / "output" / f"{name}.mp4", **spec)
            cands.append(Candidate(evaluator_probe(path), episode=i))
        return Delivery(videos=cands, selection="content")

    def bus(self, delivery, seconds=20.0):
        run = Run(self.root, "h", "prompt_01_case", "m",
                  Target(episodes=len(delivery.videos), seconds_per_episode=seconds))
        return run, EvidenceBus(run, delivery, self.tmp / "out", facts=TrajectoryFacts())

    # ------------------------------------------------------------ dispatch --
    def test_preflight_names_the_implementation_and_what_is_missing(self):
        pf = preflight.check(self.tmp, want_modules=ALL_MODULES, want_lm=False)
        self.assertEqual(pf.module_origins["physical"], "missing")
        self.assertEqual({m for m, o in pf.module_origins.items() if o == "reference"},
                         {"spec", "artifact", "visual", "av", "seam"})
        self.assertEqual(set(pf.runnable_modules), {"spec", "artifact", "visual", "av", "seam"})
        self.assertIn("参考重实现", pf.render())

    def test_each_reference_module_runs_as_a_process_and_stamps_its_report(self):
        delivery = self.delivery(ep1=dict(frames=np.concatenate([synth.scene(120, 1), synth.scene(120, 2)]),
                                          audio=tone(10.0)))
        _, bus = self.bus(delivery, seconds=10.0)
        video = delivery.videos[0].info.path
        for name in ("spec", "artifact", "visual", "av", "seam"):
            report = getattr(bus, name)(video)
            self.assertIsNotNone(report, (name, bus.errors))
            self.assertEqual(bus.status(name, video), "ok")
            self.assertFalse(report["implementation"]["original_detector"], name)
        files = sorted(p.name for p in (self.tmp / "out").rglob("*_ref.json"))
        self.assertEqual(len(files), 5, files)             # never the originals' cache names
        self.assertEqual(bus.artifact(video)["splices"]["segment_frames"], [120, 120])
        self.assertEqual(bus.seam(video)["summary"]["n_seams"], 1)
        self.assertNotIn("character_identity", bus.visual(video))

    def test_old_ffmpeg_without_fps_mode_falls_back(self):
        delivery = self.delivery(ep1=dict(frames=synth.scene(96, 3)))
        _, bus = self.bus(delivery, seconds=4.0)
        with patch.dict(os.environ, {"FAKE_FFMPEG_OLD": "1"}):
            report = bus.artifact(delivery.videos[0].info.path)
        self.assertEqual(report["meta"]["n_frames"], 96)

    def test_a_stream_that_logs_heavily_while_decoding_does_not_hang(self):
        delivery = self.delivery(ep1=dict(frames=synth.scene(240, 3)))
        _, bus = self.bus(delivery, seconds=10.0)
        with patch.dict(os.environ, {"FAKE_FFMPEG_NOISY": "1"}), \
             patch.dict(runners.DEFAULT_TIMEOUTS, {"artifact": 60}):
            report = bus.artifact(delivery.videos[0].info.path)
        self.assertIsNotNone(report, bus.errors)
        self.assertEqual(report["meta"]["n_frames"], 240)

    def test_a_decode_that_dies_half_way_is_a_failure_not_a_shorter_film(self):
        delivery = self.delivery(ep1=dict(frames=synth.scene(240, 3)))
        video = delivery.videos[0].info.path
        with patch.dict(os.environ, {"FAKE_FFMPEG_PARTIAL": "1"}):
            for module in ("artifact", "visual"):
                got = runners.run_module(module, video, self.tmp / "out_partial", device="cpu") \
                    if module == "visual" else runners.run_module(module, video, self.tmp / "out_partial")
                if module == "artifact":
                    self.assertFalse(got.ok)
                else:                       # the repeat scan reports its own abstention
                    self.assertTrue(got.report["degeneracy"]["repeats"]["abstained"])

    def test_a_file_name_with_a_colon_is_read_as_a_file(self):
        path = fake_media.write_video(self.root / "output" / "ep1 12:30.mp4", synth.scene(48, 3))
        got = runners.run_module("artifact", path, self.tmp / "out")
        self.assertTrue(got.ok, got.error)

    def test_undecodable_file_fails_the_module_not_the_run(self):
        bad = self.root / "output" / "ep1.mp4"
        bad.parent.mkdir(parents=True)
        bad.write_bytes(b"not a video")
        run = runners.run_module("artifact", bad, self.tmp / "out")
        self.assertFalse(run.ok)
        self.assertIn("rc=2", run.error)

    def test_original_entrypoint_wins_and_reference_can_be_disabled(self):
        script, origin = runners.resolve_script("artifact")
        self.assertEqual(origin, "reference")
        self.assertEqual(runners.resolve_script("physical"), (None, "missing"))
        with patch.dict(os.environ, {"VAB_EVAL_REFERENCE_DETECTORS": "0"}):
            self.assertEqual(runners.resolve_script("artifact"), (None, "missing"))
        with patch.object(runners, "USE_REFERENCE", False):
            self.assertEqual(runners.resolve_script("artifact"), (None, "missing"))
        fixture = self.tmp / "detectors" / "artifact_checks" / "run_artifact_check.py"
        fixture.parent.mkdir(parents=True)
        fixture.write_text("# an installed original\n")
        with patch.dict(runners.SCRIPTS, {"artifact": fixture}):
            self.assertEqual(runners.resolve_script("artifact"), (fixture, "original"))
            self.assertEqual(runners.detector_dir("artifact"), fixture.parent)
            self.assertEqual(runners.PROBES["artifact"], "import cv2, numpy")

    # ------------------------------------------------------- whole episode --
    def score(self, delivery, seconds=20.0):
        run, _ = self.bus(delivery, seconds)
        with patch("run.gather_facts", return_value=(TrajectoryFacts(), None)):
            return evaluate_run(run, delivery, self.tmp / "out", dict(DEFAULT_CUT_POINTS),
                                modules=ALL_MODULES, device="cpu", lm_opts={})

    def test_padded_delivery_loses_completeness_through_the_whole_chain(self):
        clip = synth.scene(240, 4)
        genuine = self.score(self.delivery(ep1=dict(frames=synth.scene(480, 4), audio=tone(20.0))))
        self.assertEqual(genuine["C"], 1.0)
        shutil.rmtree(self.tmp / "out")
        shutil.rmtree(self.root)
        padded = np.concatenate([clip[:120], np.repeat(clip[120:121], 120, axis=0),   # 5 s held
                                 clip[120:], clip[:120]])                             # 5 s shown again
        result = self.score(self.delivery(ep1=dict(frames=padded, audio=tone(20.0))))
        episode = result["completeness"]["episodes"][0]
        self.assertAlmostEqual(episode["deductions"]["frozen_s"], 5.0, delta=0.2)
        self.assertAlmostEqual(episode["deductions"]["duplicate_s"], 5.0, delta=0.6)
        self.assertAlmostEqual(episode["genuine_s"], 10.0, delta=0.8)
        self.assertLess(result["C"], 0.5)
        self.assertNotIn("artifact_unmeasured", episode["evidence_gaps"])
        self.assertNotIn("timeline_repeats_unmeasured", episode["evidence_gaps"])

    def test_picture_shorter_than_container_is_deducted(self):
        result = self.score(self.delivery(ep1=dict(frames=synth.scene(240, 5), audio=tone(20.0),
                                                   container_extra=10.0)))
        episode = result["completeness"]["episodes"][0]
        self.assertAlmostEqual(episode["delivered_s"], 20.0, delta=0.01)
        self.assertAlmostEqual(episode["deductions"]["no_picture_s"], 10.0, delta=0.05)
        self.assertAlmostEqual(episode["genuine_s"], 10.0, delta=0.05)

    def test_measured_and_unmeasured_quality_sources_are_kept_apart(self):
        frames = np.concatenate([synth.scene(160, 6), synth.scene(160, 7), synth.scene(160, 8)])
        result = self.score(self.delivery(ep1=dict(frames=frames, audio=tone(20.0))))
        ms = result["measurements"]
        self.assertEqual(ms["seam.cut_continuity"]["verdict"], "PASS")
        self.assertEqual(ms["av.clipping"]["verdict"], "PASS")
        self.assertEqual(ms["av.long_silence"]["verdict"], "PASS")
        # Not reimplemented: these abstain, they do not pass.
        self.assertEqual(ms["visual.identity_consistency"]["verdict"], "ABSTAIN")
        self.assertEqual(ms["av.dynamic_range"]["verdict"], "ABSTAIN")
        # No implementation at all: physical contact stays unmeasured.
        self.assertEqual(ms["physical.penetration"]["verdict"], "ABSTAIN")
        consistency = result["quality"]["dimensions"]["consistency"]
        self.assertEqual(consistency["leaves"][1]["sources"]["seam.cut_continuity"], 1.0)
        self.assertIsNone(consistency["leaves"][1]["sources"]["visual.identity_consistency"])
        self.assertIn("visual.identity_consistency", consistency["missing_sources"])
        self.assertTrue(result["score_is_partial"])

    def test_a_silent_track_fails_the_audio_track(self):
        silent = np.zeros((48000 * 20, 2), dtype=np.int16)
        result = self.score(self.delivery(ep1=dict(frames=synth.scene(480, 9), audio=silent)))
        self.assertEqual(result["measurements"]["spec.audio_track"]["verdict"], "FAIL")


class WithoutAnyModule(unittest.TestCase):
    def test_container_overhang_is_deducted_from_probe_metadata_alone(self):
        from stage5_scoring import ledger_from_reports
        led = ledger_from_reports("v", 1, 60.0, 82.79, probe_picture_s=60.04)
        self.assertAlmostEqual(led.no_picture_s, 22.75, delta=0.01)
        self.assertEqual(led.sources["no_picture_s"], "ffprobe")
        self.assertEqual(ledger_from_reports("v", 1, 60.0, 60.5, probe_picture_s=60.0).no_picture_s, 0)
        self.assertEqual(ledger_from_reports("v", 1, 60.0, 82.79).no_picture_s, 0)

    def test_a_module_report_still_takes_precedence_over_the_probe(self):
        from stage5_scoring import ledger_from_reports
        spec = {"meta": {"duration": 82.79, "video_duration": 70.0}}
        led = ledger_from_reports("v", 1, 60.0, 82.79, spec, probe_picture_s=60.04)
        self.assertAlmostEqual(led.no_picture_s, 12.79, delta=0.01)
        self.assertEqual(led.sources["no_picture_s"], "video_stream")

    def test_unknown_cut_structure_is_unmeasured_not_inapplicable(self):
        root = Path("unused")
        delivery = Delivery(videos=[], selection="content")
        bus = EvidenceBus(Run(root, "h", "c", "m", Target()), delivery, root)
        video = root / "ep1.mp4"
        with patch.object(bus, "artifact", return_value=None), \
             patch.object(bus, "conditioning_images", return_value=[]), \
             patch.object(runners, "run_module") as launched:
            self.assertIsNone(bus.seam(video))
        launched.assert_not_called()
        self.assertIn("切点结构未知", bus.status("seam", video))


if __name__ == "__main__":
    unittest.main()
