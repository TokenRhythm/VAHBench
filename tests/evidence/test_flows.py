"""flows.py: the cross-channel rules, the ledger, and the checklist consumer
registry -- plus the extract.py changes those rules exposed on real reports
(fade-out frames scored as defects, a module verdict overruled by its own raw
fraction, a boolean dynamic-range cliff, a rounding artefact abstaining the
yield check, "delivered" not counted as a success status).
"""
from __future__ import annotations

import os
import sys
import unittest


import stage4_checks.reports as extract
import stage4_checks.cross_checks as flows                                      # noqa: E402
from stage2_delivery.discover import Target                                     # noqa: E402
from stage4_checks.reports import _m                                          # noqa: E402
from stage4_checks.cross_checks import Ledger                                        # noqa: E402
from stage5_scoring.registry import ALL_CHECKS
from shared.types import Scale
from shared.types import Verdict                    # noqa: E402
from stage1_inputs.schema import CONSUMERS
from stage1_inputs.schema import ProducedArtifact
from stage1_inputs.schema import TrajectoryFacts
from stage1_inputs.schema import _flatten


def _facts(**kw) -> TrajectoryFacts:
    f = TrajectoryFacts()
    f.calls.audio_gen = kw.get("audio_gen")
    f.produced = kw.get("produced")
    f.conditioning = kw.get("conditioning")
    f.claims.deviations = kw.get("deviations")
    f.claims.self_checks = kw.get("self_checks")
    f.claims.declared_clips_per_episode = kw.get("clips")
    f.claims.self_reported_status = kw.get("status")
    f.requested.clip_durations = kw.get("durs")
    f.requested.generated_seconds = kw.get("generated")
    f.requested.generated_seconds_source = kw.get("source")
    return f


class TestRegistry(unittest.TestCase):
    def test_every_checklist_field_has_a_consumer(self):
        names = [n for n, _ in _flatten(TrajectoryFacts())]
        self.assertTrue(names)
        for n in names:
            self.assertIn(n, CONSUMERS, f"{n} has no consumer")
            self.assertTrue(CONSUMERS[n], f"{n} has an empty consumer list")

    def test_no_consumer_names_a_field_that_does_not_exist(self):
        """The orphan test above checks one direction only. Continuity sat in
        CONSUMERS for a session without appearing in `_flatten` -- its fields
        worked, but `checklist_usage` stopped counting them and `compare()`
        never scored the LM backend on them."""
        names = {n for n, _ in _flatten(TrajectoryFacts())}
        for field in CONSUMERS:
            self.assertIn(field, names, f"CONSUMERS names {field}, schema does not")

    def test_every_group_on_the_facts_is_walked(self):
        """`_GROUPS` is what makes a dataclass group visible to the checklist
        machinery. Adding a group and forgetting it here fails silently."""
        import dataclasses
        from stage1_inputs.schema import _GROUPS
        for f in dataclasses.fields(TrajectoryFacts()):
            v = getattr(TrajectoryFacts(), f.name)
            if dataclasses.is_dataclass(v):
                self.assertIsInstance(v, _GROUPS, f"{f.name} is not in _GROUPS")

    def test_check_consumers_are_real_checks(self):
        for field, cons in CONSUMERS.items():
            for kind, target in cons:
                self.assertIn(kind, ("check", "flow", "config", "context"))
                if kind == "check":
                    self.assertIn(target, ALL_CHECKS, f"{field} -> unknown check {target}")

    def test_usage_reports_fired_consumers_only(self):
        f = _facts(audio_gen=0, produced=[])
        f.errors.n_recovered = 3
        ms = {"spec.audio_track": _m("spec.audio_track", Verdict.FAIL, reason="静音")}
        led = Ledger()
        flows.apply(ms, f, led)
        u = flows.checklist_usage(f, ms, led)
        self.assertIn("flow:audio_triangulation", u["fields"]["calls.audio_gen"]["fired"])
        self.assertIn("context:errors", u["fields"]["errors.n_recovered"]["fired"])
        # present, has consumers, none fired -> counted as unused on this run
        f2 = _facts(clips=[6, 6, 6])
        u2 = flows.checklist_usage(f2, {}, Ledger())
        self.assertIn("claims.declared_clips_per_episode", u2["unused_present"])
        self.assertEqual(u2["n_present_used"], 0)


class TestAudioTriangulation(unittest.TestCase):
    def test_generated_but_silent_is_an_execution_failure(self):
        ms = {"spec.audio_track": _m("spec.audio_track", Verdict.FAIL, reason="音轨存在但全程静音")}
        led = Ledger()
        flows.apply(ms, _facts(audio_gen=3), led)
        m = ms["spec.audio_track"]
        self.assertIs(m.verdict, Verdict.FAIL)
        self.assertIn("未合入", m.reason)
        self.assertEqual(m.detail["triangulation"], "planned_not_delivered")
        self.assertIn("audio_triangulation", led.rules)

    def test_produced_audio_files_count_as_generation(self):
        ms = {"spec.audio_track": _m("spec.audio_track", Verdict.FAIL, reason="静音")}
        f = _facts(audio_gen=None, produced=[ProducedArtifact(path="x/ep1_vo.wav")])
        flows.apply(ms, f, Ledger())
        self.assertEqual(ms["spec.audio_track"].detail["triangulation"], "planned_not_delivered")

    def test_never_generated_is_ignored(self):
        ms = {"spec.audio_track": _m("spec.audio_track", Verdict.FAIL, reason="静音")}
        flows.apply(ms, _facts(audio_gen=0, produced=[]), Ledger())
        m = ms["spec.audio_track"]
        self.assertIn("忽略", m.reason)
        self.assertEqual(m.detail["triangulation"], "ignored")

    def test_live_track_without_audio_calls_is_annotated_not_failed(self):
        ms = {"spec.audio_track": _m("spec.audio_track", Verdict.PASS)}
        led = Ledger()
        flows.apply(ms, _facts(audio_gen=0, produced=[]), led)
        self.assertIs(ms["spec.audio_track"].verdict, Verdict.PASS)
        self.assertEqual(ms["spec.audio_track"].detail["audio_source"],
                         "video_model_or_unrecorded")
        self.assertEqual([e.effect for e in led.entries if e.rule == "audio_triangulation"],
                         ["annotate"])


class TestOtherRules(unittest.TestCase):
    def test_deviation_attribution_annotates_without_moving_the_verdict(self):
        ms = {"spec.audio_track": _m("spec.audio_track", Verdict.FAIL, reason="静音")}
        f = _facts(deviations=["storyvideo_gen has audio integration commented out -- output is silent"])
        led = Ledger()
        flows.apply(ms, f, led)
        self.assertIs(ms["spec.audio_track"].verdict, Verdict.FAIL)
        self.assertIn("output is silent", ms["spec.audio_track"].detail["declared_deviation"])
        self.assertIn("deviation_attribution", led.rules)

    def test_requested_vs_delivered_names_the_stage(self):
        ms = {"o2.requested_params": _m("o2.requested_params", Verdict.FAIL,
                                        reason="请求画幅 16:9，任务要求 9:16"),
              "spec.aspect": _m("spec.aspect", Verdict.PASS)}
        led = Ledger()
        flows.apply(ms, _facts(), led)
        self.assertTrue(ms["o2.requested_params"].detail.get("spec.aspect_fixed_in_post"))
        ms2 = {"o2.requested_params": _m("o2.requested_params", Verdict.PASS),
               "spec.aspect": _m("spec.aspect", Verdict.FAIL, reason="画幅 4:3")}
        flows.apply(ms2, _facts(), led)
        self.assertEqual(ms2["spec.aspect"].detail.get("stage"), "assembly_or_export")
        self.assertEqual(sum(1 for e in led.entries if e.rule == "requested_vs_delivered"), 2)

    def test_seam_alignment_from_declared_structure(self):
        ms = {"claims.structure_consistent": _m("claims.structure_consistent", Verdict.PASS)}
        f = _facts(clips=[3], durs=[10.0, 10.0, 10.0])
        led = Ledger()
        flows.apply(ms, f, led, splice_times={"ep1.mp4": [10.1, 19.8, 29.9]})
        sa = ms["claims.structure_consistent"].detail["seam_alignment"]
        self.assertEqual((sa["matched"], sa["n"], sa["durations"]), (2, 2, "per_request"))
        self.assertIn("seam_alignment", led.rules)

    def test_pixels_corroborate_derived_figure_and_log_it(self):
        ms = {"provenance.duration_provenance": _m(
                  "provenance.duration_provenance", Verdict.FAIL, Scale.RATE, 0.8,
                  reason="交付 180s 超过生成 30s（6.00×）（生成时长为derived，待像素侧印证）",
                  source="derived"),
              "visual.near_duplicate_shots": _m("visual.near_duplicate_shots", Verdict.FAIL,
                                                Scale.RATE, 1.0, reason="6 组")}
        led = Ledger()
        flows.apply(ms, _facts(), led)
        self.assertIs(ms["provenance.duration_provenance"].verdict, Verdict.VOID)
        e = next(e for e in led.entries if e.rule == "pixels_corroborate_duration")
        self.assertEqual(e.effect, "upgrade")
        self.assertIn("visual.near_duplicate_shots", e.src)

    # ---- scale drift, attributed by whether the agent built an anchor ----
    @staticmethod
    def _drift_ms(n_clips, with_keyframe, *, drift=Verdict.FAIL,
                  ce=Verdict.FAIL):
        return {
            "visual.scale_consistency": _m(
                "visual.scale_consistency", drift, Scale.RATIO, 3.7,
                reason="跨镜头尺寸漂移是镜头内测量误差的 3.7 倍",
                worst_pair="cell phone|person", worst_pair_shots=[1, 2, 3]),
            "provenance.conditioning_edges": _m(
                "provenance.conditioning_edges", ce, Scale.RATE, 0.5,
                reason="4/8 个单位没有参考",
                clips={"n": n_clips, "with_keyframe": with_keyframe}),
        }

    def test_drift_with_no_keyframe_at_all_is_the_agents(self):
        """Nothing carried scale across the cut, and the pixels show it.

        Two unrelated channels agreeing is what upgrades a FAIL to VOID here,
        exactly as in pixels_corroborate_duration.
        """
        ms = self._drift_ms(8, 0)
        led = Ledger()
        flows.apply(ms, _facts(), led)
        self.assertIs(ms["provenance.conditioning_edges"].verdict, Verdict.FAIL)
        self.assertEqual(ms["visual.scale_consistency"].detail["attribution"], "agent")
        e = next(e for e in led.entries
                 if e.rule == "drift_corroborates_missing_conditioning")
        self.assertEqual(e.effect, "annotate")
        self.assertIn("visual.scale_consistency", e.src)

    def test_drift_despite_full_conditioning_is_the_models(self):
        """The agent anchored every clip and the size relation moved anyway.

        The defect still scores in O4 -- it shipped -- but conditioning_edges
        must not be upgraded: the agent did the thing it is judged on.
        """
        ms = self._drift_ms(8, 8)
        led = Ledger()
        flows.apply(ms, _facts(), led)
        self.assertIs(ms["provenance.conditioning_edges"].verdict, Verdict.FAIL)
        self.assertIs(ms["visual.scale_consistency"].verdict, Verdict.FAIL)
        self.assertEqual(ms["visual.scale_consistency"].detail["attribution"], "model")

    def test_partial_conditioning_moves_no_verdict(self):
        """Half-covered is not evidence either way; record it and stop.

        EVAL_DESIGN 6.2c (14): the over-corrections in this codebase came from
        putting a cut point where the data did not supply one.
        """
        ms = self._drift_ms(8, 3)
        led = Ledger()
        flows.apply(ms, _facts(), led)
        self.assertIs(ms["provenance.conditioning_edges"].verdict, Verdict.FAIL)
        self.assertEqual(ms["visual.scale_consistency"].detail["attribution"], "partial")
        self.assertEqual(ms["visual.scale_consistency"].detail["clips_with_keyframe"], "3/8")

    def test_no_drift_means_the_rule_never_fires(self):
        ms = self._drift_ms(8, 0, drift=Verdict.PASS)
        led = Ledger()
        flows.apply(ms, _facts(), led)
        self.assertIs(ms["provenance.conditioning_edges"].verdict, Verdict.FAIL)
        self.assertNotIn("drift_corroborates_missing_conditioning", led.rules)

    def test_self_checks_contradicted_lists_the_check(self):
        ms = {"spec.audio_track": _m("spec.audio_track", Verdict.FAIL, reason="静音")}
        led = Ledger()
        flows.apply(ms, _facts(self_checks={"audioOk": True}, audio_gen=0), led)
        self.assertIs(ms["claims.deliverables_consistent"].verdict, Verdict.FAIL)
        e = next(e for e in led.entries if e.rule == "self_checks_contradicted")
        self.assertIn("spec.audio_track", e.src)


# ------------------------------------------------ extract.py corrections --
def by_key(ms):
    return {m.key: m for m in ms}


class TestArtifactCorrections(unittest.TestCase):
    META = {"meta": {"fps": 24.0, "n_frames": 1440, "duration": 60.0},
            "splices": {"n_splices": 6, "segment_frames": [240] * 5 + [236, 4],
                        "splices": [{"frame": 240 * i, "t_s": 10.0 * i} for i in range(1, 6)]
                        + [{"frame": 1436, "t_s": 59.83}]}}

    def test_fade_out_frames_are_exempt(self):
        """claudecode ep2: four black frames at 59.83-60.00 s."""
        rep = {**self.META, "findings": {
            "black_frames": {"present": True, "n_frames": 4, "fraction": 0.0028,
                             "segments": [{"start_s": 59.83, "end_s": 60.0, "n_frames": 4}]},
            "join_artifacts": {"present": True, "n_join_artifacts": 1,
                               "join_artifacts": [{"frame": 1436, "t_s": 59.83}]}}}
        d = by_key(extract.from_artifact(rep))
        self.assertIs(d["artifact.black_frames"].verdict, Verdict.PASS)
        self.assertEqual(d["artifact.black_frames"].detail["edge_exempt_frames"], 4)
        self.assertIs(d["artifact.join_artifacts"].verdict, Verdict.PASS)

    def test_black_run_in_the_middle_still_counts(self):
        rep = {**self.META, "findings": {
            "black_frames": {"present": True, "n_frames": 4, "fraction": 0.0028,
                             "segments": [{"start_s": 30.0, "end_s": 30.17, "n_frames": 4}]}}}
        m = by_key(extract.from_artifact(rep))["artifact.black_frames"]
        self.assertIs(m.verdict, Verdict.FAIL)
        self.assertAlmostEqual(m.value, 0.0028)

    def test_module_verdict_wins_over_raw_fraction(self):
        """claudecode ep3: frozen_runs present=false, fraction 0.0083."""
        rep = {**self.META, "findings": {
            "frozen_runs": {"present": False, "n_frames": 12, "fraction": 0.0083,
                            "segments": []}}}
        m = by_key(extract.from_artifact(rep))["artifact.frozen_runs"]
        self.assertIs(m.verdict, Verdict.PASS)
        self.assertAlmostEqual(m.value, 0.0083)
        self.assertFalse(m.detail["module_present"])


class TestAvCorrections(unittest.TestCase):
    LIVE = {"cleanliness": {"rms": 0.03, "peak": 0.6, "signal_db": -20.0,
                            "clipped_runs": 0, "dynamic_range_db": 11.0,
                            "pause_fraction": 0.35, "n_speaker_clusters": 7,
                            "n_speech_windows": 10}}

    def test_dynamic_range_is_graded(self):
        m = by_key(extract.from_av(self.LIVE))["av.dynamic_range"]
        self.assertIs(m.verdict, Verdict.FAIL)
        self.assertEqual(m.scale, Scale.RATE)
        self.assertAlmostEqual(m.value, 1 / 12, places=3)
        low = {"cleanliness": {**self.LIVE["cleanliness"], "dynamic_range_db": 7.0}}
        self.assertAlmostEqual(by_key(extract.from_av(low))["av.dynamic_range"].value,
                               5 / 12, places=3)

    def test_speakers_relative_to_the_cast(self):
        four = by_key(extract.from_av(self.LIVE, expected_speakers=4))["av.speaker_consistency"]
        self.assertIs(four.verdict, Verdict.ABSTAIN)
        self.assertEqual(four.detail["n_clusters"], 7)
        seven = by_key(extract.from_av(self.LIVE, expected_speakers=7))["av.speaker_consistency"]
        self.assertIs(seven.verdict, Verdict.ABSTAIN)

    def test_fixed_rule_without_a_brief(self):
        m = by_key(extract.from_av(self.LIVE))["av.speaker_consistency"]
        self.assertIs(m.verdict, Verdict.ABSTAIN)
        self.assertEqual(m.scale, Scale.BOOLEAN)


class TestDeliveryCorrections(unittest.TestCase):
    def test_delivered_is_a_success_status(self):
        d = by_key(extract.from_delivery(3, Target(episodes=3), 180.0,
                                         _facts(status="delivered")))
        self.assertIs(d["claims.status_consistent"].verdict, Verdict.PASS)

    def test_container_rounding_is_not_padding(self):
        """claudecode: 180.1 s delivered against a stated 180 s."""
        d = by_key(extract.from_delivery(3, Target(episodes=3), 180.12,
                                         _facts(generated=180.0, source="stated")))
        self.assertIs(d["provenance.duration_provenance"].verdict, Verdict.ABSTAIN)
        self.assertNotIn("provenance.generation_yield", d)


if __name__ == "__main__":
    unittest.main()


class TestContractExplainsConsistency(unittest.TestCase):
    """A cross-shot consistency failure, attributed by where the chain broke.

    Same shape as drift_corroborates_missing_conditioning, asked about the
    text the agent wrote instead of the keyframes it drew. A character cannot
    stay herself across a cut if nothing ever said what she looks like.
    """

    @staticmethod
    def _ms(contract, prompts=None, *, consistency=Verdict.FAIL):
        out = {"visual.identity_consistency": _m(
            "visual.identity_consistency", consistency, Scale.RATIO, 2.4,
            reason="最差身份的跨镜头自相似是同镜头内噪声的 2.4 倍")}
        if contract is not None:
            out["brief.continuity_contract"] = _m(
                "brief.continuity_contract", contract,
                *((Scale.RATE, 0.96) if contract is Verdict.FAIL else
                  (Scale.RATE, 0.0) if contract is Verdict.PASS else (Scale.BOOLEAN, None)),
                reason="任务书的 26 个外观词里只写下 1 个")
        if prompts is not None:
            out["brief.specs_in_prompts"] = _m(
                "brief.specs_in_prompts", prompts,
                *((Scale.RATE, 0.0) if prompts is Verdict.PASS else (Scale.RATE, 0.7)),
                reason="")
        return out

    def test_unmeasured_contract_cannot_be_inferred_from_drift(self):
        """codex wrote no character appearance anywhere. Two unrelated
        channels agreeing is what upgrades the contract finding to VOID."""
        ms = self._ms(Verdict.ABSTAIN)
        led = Ledger()
        flows.apply(ms, _facts(), led)
        self.assertIs(ms["brief.continuity_contract"].verdict, Verdict.ABSTAIN)
        self.assertEqual(
            ms["visual.identity_consistency"].detail["attribution"], None)
        self.assertEqual(
            ms["visual.identity_consistency"].detail["chain_broke_at"], "unmeasured")
        e = next(e for e in led.entries if e.rule == "contract_explains_consistency")
        self.assertEqual(e.effect, "annotate")

    def test_written_and_quoted_but_still_inconsistent_is_the_models(self):
        ms = self._ms(Verdict.PASS, Verdict.PASS)
        led = Ledger()
        flows.apply(ms, _facts(), led)
        self.assertIs(ms["brief.continuity_contract"].verdict, Verdict.PASS)
        d = ms["visual.identity_consistency"].detail
        self.assertEqual((d["attribution"], d["chain_broke_at"]),
                         ("model", "generation"))

    def test_written_but_never_quoted_breaks_at_the_prompts(self):
        """The asset exists and was not used -- the text-side twin of the
        agent drawing four character sheets the delivery matches none of."""
        ms = self._ms(Verdict.PASS, Verdict.FAIL)
        led = Ledger()
        flows.apply(ms, _facts(), led)
        d = ms["visual.identity_consistency"].detail
        self.assertEqual((d["attribution"], d["chain_broke_at"]),
                         ("agent", "prompts"))
        self.assertIs(ms["brief.continuity_contract"].verdict, Verdict.PASS)

    def test_a_character_with_no_spec_at_all_is_the_agents(self):
        """codex: 3 of 4 characters have no appearance text anywhere. The
        chain demonstrably broke at step one -- FAIL, not ABSTAIN, so the
        contract keeps its verdict and only the attribution is recorded."""
        ms = self._ms(Verdict.FAIL, Verdict.FAIL)
        led = Ledger()
        flows.apply(ms, _facts(), led)
        self.assertIs(ms["brief.continuity_contract"].verdict, Verdict.FAIL)
        d = ms["visual.identity_consistency"].detail
        self.assertEqual((d["attribution"], d["chain_broke_at"]),
                         ("agent", "contract"))

    def test_unreadable_prompts_are_not_a_broken_chain(self):
        """Three of five harnesses log prompts only inside trajectory exports
        nothing deterministic can read. Blaming them for our blind spot would
        be the same error as scoring a silent track's zero words as a pass."""
        ms = self._ms(Verdict.PASS, Verdict.ABSTAIN)
        led = Ledger()
        flows.apply(ms, _facts(), led)
        d = ms["visual.identity_consistency"].detail
        self.assertIsNone(d["attribution"])
        self.assertEqual(d["chain_broke_at"], "unmeasured")

    def test_consistent_delivery_never_fires_the_rule(self):
        ms = self._ms(Verdict.ABSTAIN, consistency=Verdict.PASS)
        led = Ledger()
        flows.apply(ms, _facts(), led)
        self.assertIs(ms["brief.continuity_contract"].verdict, Verdict.ABSTAIN)
        self.assertNotIn("contract_explains_consistency", led.rules)

    def test_scale_and_identity_are_both_attributed(self):
        ms = self._ms(Verdict.PASS, Verdict.PASS)
        ms["visual.scale_consistency"] = _m(
            "visual.scale_consistency", Verdict.FAIL, Scale.RATIO, 3.7, reason="")
        flows.apply(ms, _facts(), Ledger())
        for k in ("visual.identity_consistency", "visual.scale_consistency"):
            self.assertEqual(ms[k].detail["attribution"], "model")
