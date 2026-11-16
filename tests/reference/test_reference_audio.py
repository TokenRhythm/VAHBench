"""Track level and clipping; what is not measured must abstain, not pass."""
import unittest

import numpy as np

import stage4_checks.reports as adapters
from shared.types import Verdict
from tests.reference import synth

synth.use_reference("av")
import run_audio_check as av  # noqa: E402

SR = 48000


def tone(seconds=2.0, amplitude=0.5, freq=220.0, channels=2):
    t = np.arange(int(SR * seconds)) / SR
    x = amplitude * np.sin(2 * np.pi * freq * t)
    return np.repeat(x[:, None], channels, axis=1)


def pcm(x):
    return np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)


def report(samples):
    return {"audio_stream": {"present": True}, "cleanliness": av.cleanliness(samples, SR),
            "not_measured": av.NOT_MEASURED}


def by_key(rep):
    return {m.key: m for m in adapters.from_av(rep)}


class TestClipping(unittest.TestCase):
    def test_a_clean_tone_has_no_clipped_run(self):
        clean = av.cleanliness(pcm(tone()), SR)
        self.assertEqual(clean["clipped_runs"], [])
        self.assertAlmostEqual(clean["peak"], 0.5, places=3)
        self.assertAlmostEqual(clean["rms"], 0.5 / np.sqrt(2), places=3)
        self.assertIs(by_key(report(pcm(tone())))["av.clipping"].verdict, Verdict.PASS)

    def test_a_full_scale_peak_without_a_flat_top_is_not_clipping(self):
        self.assertEqual(av.cleanliness(pcm(tone(amplitude=1.0, freq=4000.0)), SR)["clipped_runs"], [])

    def test_clean_low_tones_are_not_clipping(self):
        # The crest of a low tone repeats one value for a few samples.
        for amplitude, freq in ((0.5, 50.0), (0.26, 80.0), (1.0, 50.0), (0.9, 30.0)):
            clean = av.cleanliness(pcm(tone(amplitude=amplitude, freq=freq)), SR)
            self.assertEqual(clean["n_clipped_runs"], 0, (amplitude, freq))

    def test_overdriven_audio_is_clipping(self):
        clean = av.cleanliness(pcm(np.clip(tone(amplitude=2.0), -1, 1)), SR)
        self.assertGreater(clean["n_clipped_runs"], 100)
        self.assertEqual({r["kind"] for r in clean["clipped_runs"]}, {"full_scale"})
        m = by_key(report(pcm(np.clip(tone(amplitude=2.0), -1, 1))))["av.clipping"]
        self.assertIs(m.verdict, Verdict.FAIL)

    def test_clipped_then_turned_down_is_still_clipping(self):
        clipped = np.clip(tone(amplitude=2.0), -1, 1) * 0.5
        clean = av.cleanliness(pcm(clipped), SR)
        self.assertGreater(clean["n_clipped_runs"], 100)
        self.assertEqual({r["kind"] for r in clean["clipped_runs"]}, {"flat_top"})

    def test_one_clipped_channel_is_enough(self):
        x = tone()
        x[:, 1] = np.clip(tone(amplitude=3.0)[:, 1], -1, 1)
        runs = av.cleanliness(pcm(x), SR)["clipped_runs"]
        self.assertEqual({r["channel"] for r in runs}, {1})


class TestWhatIsNotMeasured(unittest.TestCase):
    def test_dynamic_range_abstains_instead_of_passing(self):
        d = by_key(report(pcm(tone())))
        self.assertIs(d["av.dynamic_range"].verdict, Verdict.ABSTAIN)
        self.assertIs(d["av.voice_face_match"].verdict, Verdict.ABSTAIN)
        self.assertIs(d["av.speaker_consistency"].verdict, Verdict.ABSTAIN)
        clean = av.cleanliness(pcm(tone()), SR)
        self.assertIsNone(clean["dynamic_range_db"])
        self.assertIsNone(clean["pause_fraction"])


class TestSilence(unittest.TestCase):
    def test_a_silent_stream_fails_the_audio_track(self):
        silent = np.zeros((SR, 2), dtype=np.int16)
        clean = av.cleanliness(silent, SR)
        self.assertEqual((clean["rms"], clean["peak"], clean["signal_db"]), (0.0, 0.0, -200.0))
        d = by_key(report(silent))
        self.assertIs(d["spec.audio_track"].verdict, Verdict.FAIL)
        self.assertIs(d["av.clipping"].verdict, Verdict.NOT_APPLICABLE)

    def test_channels_in_opposite_phase_are_not_silence(self):
        x = tone(amplitude=0.3)
        x[:, 1] = -x[:, 0]
        clean = av.cleanliness(pcm(x), SR)
        self.assertAlmostEqual(clean["rms"], 0.3 / np.sqrt(2), places=3)
        self.assertNotIn("spec.audio_track", by_key(report(pcm(x))))

    def test_an_absent_stream_fails_the_audio_track(self):
        d = by_key({"audio_stream": {"present": False}, "not_measured": av.NOT_MEASURED})
        self.assertIs(d["spec.audio_track"].verdict, Verdict.FAIL)


if __name__ == "__main__":
    unittest.main()
