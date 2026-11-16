"""Extractors against the real module schemas.

Every fixture here is copied from an actual module report produced by
`data/openmontage/prompt_01/.../renders/ep1_e1.mp4`, not invented. That matters:
the first two drafts of these extractors guessed the field names and were both
wrong -- spec_checks reports `{ok, undetermined, detail}` and not `status`, and
artifact_checks puts phenomena under `findings` with a precomputed `fraction`,
not under `phenomena`. Both mistakes scored a clean video as a total failure
while every module reported "ok", which is the worst possible failure mode for
a benchmark: confident, silent, and wrong.

So the schemas are pinned here. If a module renames a field, these fail.
"""
from __future__ import annotations

import os
import sys
import unittest


import stage4_checks.reports as extract                                       # noqa: E402
from shared.types import Scale
from shared.types import Verdict                          # noqa: E402

# --- verbatim from spec_checks -------------------------------------------
SPEC = {
    "meta": {"fps": 24.0, "n_frames": 1440, "duration": 60.0,
             "width": 1080, "height": 1920, "has_audio": True,
             "audio_duration": 60.0, "rotation_metadata": 0},
    "source_clips": [],
    "checks": {
        "frame_rate": {"ok": True, "fps": 24.0, "detail": ""},
        "retiming": {"ok": True, "undetermined": True,
                     "detail": "no source clips found to compare against"},
        "duration": {"ok": True, "duration": 60.0, "target": 60.0, "detail": ""},
        "resolution": {"ok": True, "undetermined": True, "detail": "no target given"},
        "aspect": {"ok": True, "detail": ""},
        "audio": {"ok": True, "detail": ""},
        "rotation_metadata": {"ok": True, "detail": ""},
    },
}

# --- verbatim from artifact_checks ---------------------------------------
ARTIFACT = {
    "meta": {"fps": 24.0, "n_frames": 1440, "duration": 60.0},
    "splices": {"n_splices": 10, "z_cut": 25.0, "z_p99": 11.5,
                "subtitle_band_excluded": True,
                "segment_frames": [72, 120, 120, 120],
                "splices": [{"frame": 72, "t_s": 3.0, "z": 40.2}]},
    "findings": {
        "black_frames": {"present": False, "n_frames": 0, "fraction": 0.0},
        "solid_frames": {"present": False, "n_frames": 0, "fraction": 0.0},
        "frozen_runs": {"present": False, "n_frames": 0, "fraction": 0.0},
        "flicker": {"present": False, "n_events": 0},
        "tail_freeze": {"present": False, "held_frames": 0, "held_seconds": 0.0},
        "join_artifacts": {"present": False, "n_join_artifacts": 0},
        "burned_in_subtitles": {"present": True, "n_events": 16},
    },
    "n_defects": 0, "clean": True,
}


def by_key(ms):
    return {m.key: m for m in ms}


class TestSpec(unittest.TestCase):
    def test_undetermined_is_abstain_even_though_ok_is_true(self):
        """spec_checks ships undetermined checks with ok=True. Reading `ok`
        alone promotes 'no source clips to compare against' into a clean
        retiming pass -- exactly the evidence the module refuses to fabricate."""
        d = by_key(extract.from_spec(SPEC))
        self.assertIs(d["spec.retiming"].verdict, Verdict.ABSTAIN)
        self.assertIs(d["spec.resolution"].verdict, Verdict.ABSTAIN)
        self.assertIn("no source clips", d["spec.retiming"].reason)

    def test_clean_video_passes_everything_measurable(self):
        d = by_key(extract.from_spec(SPEC))
        for k in ("spec.duration", "spec.aspect", "spec.audio_track"):
            self.assertIs(d[k].verdict, Verdict.PASS, k)

    def test_retiming_failure_is_void_not_fail(self):
        bad = {**SPEC, "checks": {**SPEC["checks"],
               "retiming": {"ok": False, "undetermined": False,
                            "detail": "1060 frames delivered as 59.97s"}}}
        self.assertIs(by_key(extract.from_spec(bad))["spec.retiming"].verdict,
                      Verdict.VOID)

    def test_empty_report_yields_nothing(self):
        self.assertEqual(extract.from_spec(None), [])


class TestArtifact(unittest.TestCase):
    def test_findings_not_phenomena(self):
        d = by_key(extract.from_artifact(ARTIFACT))
        self.assertIn("artifact.black_frames", d)
        self.assertIs(d["artifact.black_frames"].verdict, Verdict.PASS)
        self.assertEqual(d["artifact.black_frames"].scale, Scale.HARD_RATE)

    def test_precomputed_fraction_is_used_directly(self):
        rep = {**ARTIFACT, "findings": {**ARTIFACT["findings"],
               "frozen_runs": {"present": True, "n_frames": 144, "fraction": 0.1}}}
        m = by_key(extract.from_artifact(rep))["artifact.frozen_runs"]
        self.assertIs(m.verdict, Verdict.FAIL)
        self.assertAlmostEqual(m.value, 0.1)

    def test_tail_freeze_is_void(self):
        rep = {**ARTIFACT, "findings": {**ARTIFACT["findings"],
               "tail_freeze": {"present": True, "held_frames": 96,
                               "held_seconds": 4.0}}}
        self.assertIs(by_key(extract.from_artifact(rep))["artifact.tail_freeze"].verdict,
                      Verdict.VOID)


class TestSilentAudio(unittest.TestCase):
    """All three openmontage episodes: rms=0, peak=0, signal_db=-200, and they
    were passing `spec.audio_track` (a stream exists) and O5 (zero clipping --
    on silence). Zero defects on nothing is not a clean result."""

    SILENT = {"cleanliness": {"rms": 0.0, "peak": 0.0, "signal_db": -200.0,
                              "clipped_runs": [], "dynamic_range_db": 0.0,
                              "pause_fraction": 0.0, "n_speaker_clusters": 0,
                              "n_speech_windows": 0}}

    def test_silent_track_fails_audio_and_suppresses_every_audio_number(self):
        d = by_key(extract.from_av(self.SILENT))
        self.assertIs(d["spec.audio_track"].verdict, Verdict.FAIL)
        for k in ("av.clipping", "av.dynamic_range", "av.voice_face_match", "av.long_silence"):
            self.assertIs(d[k].verdict, Verdict.NOT_APPLICABLE, k)
            self.assertIn("静音", d[k].reason)
        # Voice consistency (1.2.2.d) is the exception: no voices is not
        # evidence of inconsistent voices, and silence is already charged in
        # 1.2.1.a and 1.2.3.b/c.
        self.assertIs(d["av.speaker_consistency"].verdict, Verdict.NOT_APPLICABLE)

    def test_live_track_is_untouched(self):
        live = {"cleanliness": {**self.SILENT["cleanliness"], "rms": 0.12,
                                "peak": 0.9, "signal_db": -18.0}}
        d = by_key(extract.from_av(live))
        self.assertNotIn("spec.audio_track", d)
        self.assertIs(d["av.clipping"].verdict, Verdict.PASS)


class _T:
    def __init__(self, episodes=3, aspect_ratio="9:16", resolution=None):
        self.episodes, self.aspect_ratio, self.resolution = episodes, aspect_ratio, resolution


def _facts(status=None, declared=None, generated=None, source=None):
    from stage1_inputs.schema import Claims
    from stage1_inputs.schema import RequestedParams
    from stage1_inputs.schema import TrajectoryFacts
    if generated is not None and source is None:
        source = "per_request"
    return TrajectoryFacts(
        claims=Claims(self_reported_status=status, declared_deliverables=declared),
        requested=RequestedParams(generated_seconds=generated,
                                  generated_seconds_source=source))


class TestClosedLoop(unittest.TestCase):
    """Trajectory x delivery. Every contradiction between what the run said,
    did and shipped is a scored finding, not a note."""

    def test_missing_episodes_fail(self):
        d = by_key(extract.from_delivery(1, _T(3), 71.0, _facts()))
        m = d["o2.episode_count"]
        self.assertIs(m.verdict, Verdict.FAIL)
        self.assertAlmostEqual(m.value, 2 / 3)

    def test_full_delivery_passes(self):
        d = by_key(extract.from_delivery(3, _T(3), 180.0, _facts()))
        self.assertIs(d["o2.episode_count"].verdict, Verdict.PASS)

    def test_no_delivery_voids_content_checks_rather_than_abstaining(self):
        """univa: zero episodes across five briefs, and it was invisible."""
        d = by_key(extract.from_delivery(0, _T(3), None, _facts(status="ok")))
        for k in ("spec.duration", "spec.audio_track", "artifact.tail_freeze"):
            self.assertIs(d[k].verdict, Verdict.VOID, k)
        self.assertIs(d["o2.episode_count"].verdict, Verdict.FAIL)
        self.assertEqual(d["o2.episode_count"].value, 1.0)

    def test_said_ok_delivered_nothing(self):
        d = by_key(extract.from_delivery(0, _T(3), None, _facts(status="ok")))
        self.assertIs(d["claims.status_consistent"].verdict, Verdict.FAIL)
        self.assertIn("0/3", d["claims.status_consistent"].reason)

    def test_said_failed_delivered_two_is_not_a_lie(self):
        """Under-claiming is not dishonesty: an agent that reported failure and
        delivered anyway assessed itself pessimistically. Only over-claiming is
        charged (2026-09-22; the branch that failed this was 5 of the 12 honesty
        FAILs over `data/`, every one of them a harness's return code)."""
        d = by_key(extract.from_delivery(2, _T(3), 105.0, _facts(status="failed")))
        self.assertIs(d["claims.status_consistent"].verdict, Verdict.PASS)

    def test_consistent_self_report_passes(self):
        d = by_key(extract.from_delivery(3, _T(3), 180.0, _facts(status="ok")))
        self.assertIs(d["claims.status_consistent"].verdict, Verdict.PASS)

    def test_partial_does_not_specify_a_delivered_video(self):
        for n, seconds in ((1, 60.0), (0, 0.0)):
            d = by_key(extract.from_delivery(n, _T(3), seconds, _facts(status="partial")))
            self.assertIs(d["claims.status_consistent"].verdict, Verdict.ABSTAIN)

    def test_unknown_status_abstains(self):
        d = by_key(extract.from_delivery(3, _T(3), 180.0, _facts(status="weird")))
        self.assertIs(d["claims.status_consistent"].verdict, Verdict.ABSTAIN)

    def test_unused_generation_is_no_longer_charged(self):
        """videoclaw: 510 s generated, 71 s delivered. That used to score 0.14
        as waste; with 2.2 gone nothing reads generated-minus-delivered."""
        d = by_key(extract.from_delivery(1, _T(3), 71.0, _facts(generated=510.0)))
        self.assertNotIn("provenance.generation_yield", d)
        self.assertIs(d["provenance.duration_provenance"].verdict, Verdict.ABSTAIN)

    def test_request_duration_does_not_prove_successful_output(self):
        """Even a complete POST inventory is not a measured output inventory."""
        d = by_key(extract.from_delivery(3, _T(3), 180.0,
                                         _facts(generated=44.0, source="per_request")))
        m = d["provenance.duration_provenance"]
        self.assertIs(m.verdict, Verdict.ABSTAIN)
        self.assertEqual(m.detail["reported_seconds"], 44.0)

    def test_derived_generated_figure_abstains(self):
        """Unverified totals do not decide authenticity."""
        d = by_key(extract.from_delivery(3, _T(3), 180.0,
                                         _facts(generated=30.0, source="derived")))
        m = d["provenance.duration_provenance"]
        self.assertIs(m.verdict, Verdict.ABSTAIN)
        self.assertEqual(m.detail["source"], "derived")


class TestCorroboration(unittest.TestCase):
    """Trajectory x video: a derived finding is upgraded only when an unrelated
    channel says the same thing."""

    def _dp(self):
        return extract._m("provenance.duration_provenance", Verdict.FAIL,
                          reason="交付 180s 超过生成 30s（6.00×）：内容被拉伸、循环或填充"
                                 "（生成时长为derived，待像素侧印证）", source="derived")

    def test_upgraded_when_pixels_show_duplicates(self):
        from stage4_checks.cross_checks import pixels_corroborate_duration
        from stage4_checks.cross_checks import Ledger
        ms = {"provenance.duration_provenance": self._dp(),
              "visual.near_duplicate_shots": extract._m(
                  "visual.near_duplicate_shots", Verdict.FAIL, Scale.RATE, 2.5,
                  reason="15 组近重复镜头")}
        pixels_corroborate_duration(ms, Ledger())
        m = ms["provenance.duration_provenance"]
        self.assertIs(m.verdict, Verdict.VOID)
        self.assertIn("像素侧印证", m.reason)
        self.assertEqual(m.detail["corroborated_by"], ["visual.near_duplicate_shots"])

    def test_left_alone_without_pixel_agreement(self):
        from stage4_checks.cross_checks import pixels_corroborate_duration
        from stage4_checks.cross_checks import Ledger
        ms = {"provenance.duration_provenance": self._dp(),
              "visual.near_duplicate_shots": extract._m(
                  "visual.near_duplicate_shots", Verdict.PASS, Scale.RATE, 0.0)}
        pixels_corroborate_duration(ms, Ledger())
        self.assertIs(ms["provenance.duration_provenance"].verdict, Verdict.FAIL)

    def test_exact_void_is_not_touched(self):
        from stage4_checks.cross_checks import pixels_corroborate_duration
        from stage4_checks.cross_checks import Ledger
        exact = extract._m("provenance.duration_provenance", Verdict.VOID,
                           reason="x", source="per_request")
        ms = {"provenance.duration_provenance": exact}
        pixels_corroborate_duration(ms, Ledger())
        self.assertIs(ms["provenance.duration_provenance"], exact)

    def test_matching_requested_and_delivered_seconds_is_not_proof(self):
        """claudecode: 18 x 10 s generated, 3 x 60 s delivered."""
        d = by_key(extract.from_delivery(3, _T(3), 180.0, _facts(generated=180.0)))
        self.assertIs(d["provenance.duration_provenance"].verdict, Verdict.ABSTAIN)

    def test_unknown_generated_seconds_abstains(self):
        d = by_key(extract.from_delivery(3, _T(3), 180.0, _facts()))
        self.assertIs(d["provenance.duration_provenance"].verdict, Verdict.ABSTAIN)


class TestNoiseNormalisation(unittest.TestCase):
    def test_reference_episode_ratio(self):
        """ep1 of the reference case: inter p10 0.353, intra mean 0.901."""
        r = extract.ratio_from_similarity(0.3534, 0.9014)
        self.assertAlmostEqual(r, (1 - 0.3534) / (1 - 0.9014), places=6)
        self.assertGreater(r, 6.0)

    def test_perfect_intra_means_floor_not_estimable(self):
        """A noiseless floor would manufacture an enormous ratio out of
        rounding. Abstain instead -- same rule as av_consistency's
        floor_not_estimable."""
        self.assertIsNone(extract.ratio_from_similarity(0.4, 1.0))
        self.assertIsNone(extract.ratio_from_similarity(0.4, 0.9999))

    def test_inter_equal_to_intra_is_exactly_one(self):
        self.assertAlmostEqual(extract.ratio_from_similarity(0.9, 0.9), 1.0)

    def test_missing_input_abstains(self):
        self.assertIsNone(extract.ratio_from_similarity(None, 0.9))
        self.assertIsNone(extract.ratio_from_similarity(0.9, None))


class TestVisualGating(unittest.TestCase):
    IDENT = {"per_identity_consistency": {"comparison_scope": "cross_shot", "worst": 0.3534, "intra_shot": 0.9014}}

    def test_degeneracy_does_not_replace_face_measurement(self):
        rep = {
            "n_shots": 4,
            "character_identity": self.IDENT,
            "degeneracy": {"verdict": {"degenerate": True,
                                       "reason": "6 个近重复镜头"}},
        }
        d = by_key(extract.from_visual(rep))
        self.assertIs(d["visual.identity_consistency"].verdict, Verdict.FAIL)
        self.assertEqual(d["visual.identity_consistency"].scale, Scale.RATIO)

    def test_healthy_video_uses_the_ratio(self):
        rep = {
            "n_shots": 4,
            "character_identity": self.IDENT,
            "degeneracy": {"verdict": {"degenerate": False}},
        }
        m = by_key(extract.from_visual(rep))["visual.identity_consistency"]
        self.assertIs(m.verdict, Verdict.FAIL)
        self.assertEqual(m.scale, Scale.RATIO)
        self.assertGreater(m.value, 6.0)

    def test_whole_frame_ratios_are_no_longer_emitted(self):
        rep = {"n_shots": 4,
               "subject": {"inter_shot": {"p10": 0.3}, "intra_shot": {"mean": 0.9}},
               "background": {"within_location": {"p10": 0.3}, "intra_shot": {"mean": 0.9}},
               "degeneracy": {"verdict": {"degenerate": False}}}
        d = by_key(extract.from_visual(rep))
        self.assertNotIn("visual.subject_consistency", d)
        self.assertNotIn("visual.background_consistency", d)


class TestDig(unittest.TestCase):
    def test_first_resolving_path_wins(self):
        d = {"b": {"c": 2}}
        self.assertEqual(extract.dig(d, "a.x", "b.c"), 2)

    def test_missing_returns_default(self):
        self.assertIsNone(extract.dig({}, "a.b.c"))
        self.assertEqual(extract.dig({}, "a", default=7), 7)

    def test_does_not_crash_on_non_dict_midpath(self):
        self.assertIsNone(extract.dig({"a": 5}, "a.b"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
