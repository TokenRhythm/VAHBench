"""Source timing: a stretch is a candidate until the pictures prove it."""
import unittest
from unittest.mock import patch

import numpy as np

from stage5_scoring import ledger_from_reports
from tests.reference import synth

synth.use_reference("spec")
from speccheck import report as spec_report  # noqa: E402
from speccheck import timing  # noqa: E402
from speccheck.probe import probe  # noqa: E402
from vahref import media  # noqa: E402


def thumbs(frames):
    return synth.features(frames).thumbs


def meta(n, fps=24.0, **extra):
    return {"path": "x", "n_frames": n, "fps": fps, "video_duration": n / fps,
            "duration": n / fps, "width": 720, "height": 1280, "has_audio": True,
            "rotation_metadata": 0, **extra}


class TestMetadataLayer(unittest.TestCase):
    def test_no_sources_is_unknown(self):
        rt = timing.check_retiming(meta(240), [])
        self.assertTrue(rt["undetermined"])
        self.assertFalse(rt["correspondence_verified"])

    def test_a_trim_is_not_a_slowdown(self):
        rt = timing.check_retiming(meta(200), [meta(240)])
        self.assertTrue(rt["ok"])
        self.assertFalse(rt["undetermined"])

    def test_missing_source_frame_count_is_unknown_not_clean(self):
        rt = timing.check_retiming(meta(480), [dict(meta(240), n_frames=0)])
        self.assertTrue(rt["undetermined"])

    def test_a_longer_picture_is_only_a_candidate_from_metadata(self):
        rt = timing.check_retiming(meta(240, fps=12.0), [meta(240)])
        self.assertTrue(rt["undetermined"])
        self.assertTrue(rt["frame_count_preserved"])
        self.assertAlmostEqual(rt["stretch"], 2.0)


class TestPictureLayer(unittest.TestCase):
    def test_every_frame_kept_at_half_rate_is_verified(self):
        clip = synth.scene(240, 1)
        got = timing.verify_correspondence(thumbs(clip), thumbs(clip))
        self.assertTrue(got["verified"], got)

    def test_duplicated_frames_are_verified(self):
        clip = synth.scene(240, 2)
        slow = np.repeat(clip, 2, axis=0)
        self.assertTrue(timing.verify_correspondence(thumbs(slow), thumbs(clip))["verified"])

    def test_other_footage_with_the_same_frame_count_is_not_verified(self):
        got = timing.verify_correspondence(thumbs(synth.scene(240, 3)), thumbs(synth.scene(240, 4)))
        self.assertFalse(got["verified"])

    def test_static_content_cannot_establish_correspondence(self):
        got = timing.verify_correspondence(thumbs(synth.still(240, 5)), thumbs(synth.still(120, 5)))
        self.assertFalse(got["verified"])
        self.assertIn("static", got["reason"])

    def test_sources_at_their_own_speed_plus_a_held_ending_are_not_a_slowdown(self):
        # Slow content: neighbouring frames look alike, so a 10 % stretch
        # would also "match". Pairing by frame number shows it is not one.
        for clip in (synth.smooth_drift(240, 6, speed=0.1), synth.scene(240, 6, speed=0.2)):
            held = np.concatenate([clip, np.repeat(clip[-1:], 24, axis=0)])
            got = timing.verify_correspondence(thumbs(held), thumbs(clip))
            self.assertFalse(got["verified"], got)

    def test_extra_footage_beside_the_sources_is_not_a_slowdown(self):
        clip = synth.scene(240, 6)
        longer = np.concatenate([clip, synth.scene(240, 7)])
        self.assertFalse(timing.verify_correspondence(thumbs(longer), thumbs(clip))["verified"])


class TestReport(unittest.TestCase):
    def run_check(self, final_frames, source_frames, final_fps, source_fps=24.0, **kw):
        metas = {"final.mp4": meta(len(final_frames), final_fps),
                 "src.mp4": meta(len(source_frames), source_fps)}
        frames = {"final.mp4": final_frames, "src.mp4": source_frames}
        with patch.object(spec_report, "probe", side_effect=lambda p: metas[str(p)]), \
             patch.object(spec_report, "_thumbs", side_effect=lambda p: thumbs(frames[str(p)])):
            return spec_report.check_video("final.mp4", source_clips=["src.mp4"], **kw)

    def test_verified_slowdown_is_deducted_from_genuine_seconds(self):
        clip = synth.scene(240, 8)                       # 10 s of footage
        rep = self.run_check(clip, clip, final_fps=12.0)  # shown over 20 s
        rt = rep["checks"]["retiming"]
        self.assertFalse(rt["ok"])
        self.assertTrue(rt["correspondence_verified"])
        self.assertTrue(rt["frame_count_preserved"])
        led = ledger_from_reports("final.mp4", 1, 20.0, 20.0, rep)
        self.assertAlmostEqual(led.retime_excess_s, 10.0, delta=0.05)
        self.assertEqual([m.verdict.value for m in __import__("stage4_checks.reports", fromlist=["x"])
                          .from_spec(rep) if m.key == "spec.retiming"], ["VOID"])

    def test_same_count_different_footage_is_not_deducted(self):
        rep = self.run_check(synth.scene(240, 9), synth.scene(240, 10), final_fps=12.0)
        rt = rep["checks"]["retiming"]
        self.assertTrue(rt["undetermined"])
        self.assertFalse(rt["correspondence_verified"])
        self.assertEqual(ledger_from_reports("final.mp4", 1, 20.0, 20.0, rep).retime_excess_s, 0)

    def test_format_predicates(self):
        clip = synth.scene(48, 11)
        rep = self.run_check(clip, clip, final_fps=24.0, target_duration=2.0,
                             target_resolution="720x1280", target_aspect="9:16")
        checks = rep["checks"]
        for name in ("duration", "resolution", "aspect", "audio", "rotation_metadata"):
            self.assertTrue(checks[name]["ok"], name)
            self.assertFalse(checks[name].get("undetermined"), name)
        bad = self.run_check(clip, clip, final_fps=24.0, target_duration=60.0,
                             target_resolution="1080x1920", target_aspect="16:9")
        self.assertEqual([bad["checks"][k]["ok"] for k in ("duration", "resolution", "aspect")],
                         [False, False, False])
        self.assertFalse(rep["implementation"]["original_detector"])

    def test_unreadable_picture_leaves_the_candidate_unverified(self):
        clip = synth.scene(240, 12)
        metas = {"final.mp4": meta(240, 12.0), "src.mp4": meta(240)}
        with patch.object(spec_report, "probe", side_effect=lambda p: metas[str(p)]), \
             patch.object(spec_report, "_thumbs", side_effect=media.MediaError("decode")):
            rep = spec_report.check_video("final.mp4", source_clips=["src.mp4"])
        self.assertTrue(rep["checks"]["retiming"]["undetermined"])
        del clip


class TestFrameClock(unittest.TestCase):
    def test_complete_decode_uses_the_streams_own_duration(self):
        self.assertAlmostEqual(media.frame_clock({"fps": 24.0, "video_duration": 10.0,
                                                  "n_frames": 239}, 239), 23.9)

    def test_partial_or_padded_decode_keeps_the_nominal_rate(self):
        m = {"fps": 24.0, "video_duration": 10.0, "n_frames": 240}
        self.assertEqual(media.frame_clock(m, 100), 24.0)
        self.assertEqual(media.frame_clock(m, 480), 24.0)
        self.assertEqual(media.frame_clock({"fps": None, "video_duration": None}, 10), 24.0)


class TestProbe(unittest.TestCase):
    def data(self, **video):
        return {"format": {"duration": "82.79"},
                "streams": [{"codec_type": "video", "avg_frame_rate": "24/1",
                             "width": 720, "height": 1280, **video},
                            {"codec_type": "audio", "duration": "82.79",
                             "sample_rate": "44100", "channels": 2}]}

    def test_picture_duration_is_the_streams_own(self):
        with patch("speccheck.probe.exists", return_value=True), \
             patch("speccheck.probe._ffprobe", return_value=self.data(nb_frames="1441", duration="60.041667")):
            m = probe("f")
        self.assertEqual(m["n_frames"], 1441)
        self.assertAlmostEqual(m["video_duration"], 60.041667)
        self.assertAlmostEqual(m["duration"], 82.79)
        self.assertTrue(m["has_audio"])

    def test_missing_frame_count_is_counted_from_packets(self):
        counted = self.data(nb_read_packets="240")
        with patch("speccheck.probe.exists", return_value=True), \
             patch("speccheck.probe._ffprobe", side_effect=[self.data(), counted]):
            m = probe("f")
        self.assertEqual((m["n_frames"], m["frame_count_source"]), (240, "counted_packets"))
        self.assertAlmostEqual(m["video_duration"], 10.0)

    def test_rotation_metadata_swaps_the_displayed_size(self):
        rotated = self.data(nb_frames="24", side_data_list=[{"rotation": -90}])
        with patch("speccheck.probe.exists", return_value=True), \
             patch("speccheck.probe._ffprobe", return_value=rotated):
            m = probe("f")
        self.assertEqual((m["width"], m["height"], m["rotation_metadata"]), (1280, 720, 270))

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            probe("/no/such/file.mp4")


if __name__ == "__main__":
    unittest.main()
