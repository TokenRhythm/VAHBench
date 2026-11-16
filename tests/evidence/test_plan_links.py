"""Text <-> module linkage without module flags.

Every rule here joins a module's *timed* output to the plan's beats
(evidence.plan_spans) or the brief's rules, in the orchestrator. Pinned:

  * under a narration ban the plan's V.O. marks are attached to
    brief.no_voiceover;
  * penetration flags land on the plan's beats; smoke/water/glass beats turn
    the reading into "instrument, not physics" without moving the verdict;
  * the module's own abstention conditions (contact rate, resolvability,
    noise floors) abstain instead of scoring;
  * recurring identities are held against the cast size;
  * the speech-label filter no longer drops "Male speech" / "Conversation".

Three plan-driven diagnostics that used to be pinned here -- `av.mouth_speech`
with its narration exemption, `visual.scene_grouping` (plan rooms x verified
links) and `brief.characters_placed` (plan names x face clusters) -- were
deleted on 2026-09-22 when the diagnostic bucket was closed; the last test
class asserts they stay gone.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


import stage4_checks.reports as extract
import stage4_checks.cross_checks as flows                                     # noqa: E402
from stage1_inputs.brief import BriefFacts
from stage1_inputs.brief import Character                        # noqa: E402
from stage3_evidence.collector import EvidenceBus                               # noqa: E402
from stage4_checks.reports import _m                                         # noqa: E402
from stage4_checks.cross_checks import Ledger                                       # noqa: E402
from stage5_scoring.registry import ALL_CHECKS
from shared.types import Scale
from shared.types import Verdict                   # noqa: E402
from stage1_inputs.schema import PlannedShot
from stage1_inputs.schema import TrajectoryFacts     # noqa: E402


def by_key(ms):
    return {m.key: m for m in ms}


LIVE = {"rms": 0.12, "peak": 0.9, "signal_db": -18.0, "clipped_runs": [],
        "dynamic_range_db": 14.0, "pause_fraction": 0.2,
        "n_speaker_clusters": 3, "n_speech_windows": 10}


# ------------------------------------------------------------------ av ----
class TestVoiceFaceLabels(unittest.TestCase):
    def test_speech_labels_beyond_the_literal_word(self):
        """presence routes six AudioSet labels through the face check; the
        old `startswith("speech")` counted two of them."""
        wins = [{"verdicts": [{"sound": s, "verdict": "implausible"}]}
                for s in ("Male speech, man speaking", "Conversation", "Narration, monologue")]
        wins.append({"verdicts": [{"sound": "Speech", "verdict": "plausible"}]})
        wins.append({"verdicts": [{"sound": "Laughter", "verdict": "implausible"}]})
        m = by_key(extract.from_av({"cleanliness": LIVE,
                                    "presence": {"windows": wins}}))["av.voice_face_match"]
        self.assertEqual(m.detail["implausible"], 3)
        self.assertEqual(m.detail["plausible"], 1)
        self.assertTrue(extract.is_speech_label("Female speech, woman speaking"))
        self.assertFalse(extract.is_speech_label("Applause"))

    def test_mouth_windows_are_not_read(self):
        rep = {"cleanliness": LIVE, "presence": {"windows": []},
               "mouth": {"windows": [{"window": {"start": 0, "end": 10, "shot": 0},
                                      "verdict": "speech_without_mouth_movement"}]}}
        self.assertNotIn("av.mouth_speech", by_key(extract.from_av(rep)))


class TestPlanVsBriefVoiceover(unittest.TestCase):
    PLAN = {"ep1_final.mp4": [
        {"start": 0, "end": 10, "text": "INT. LOBBY -- Maya (V.O.): I kept the receipts"},
        {"start": 10, "end": 20, "text": "Victor confronts Maya at the desk"},
        {"start": 20, "end": 30, "text": "the elevator, doors closing"}]}

    def test_plan_vo_under_a_ban_is_attached_to_no_voiceover(self):
        ms = {"brief.no_voiceover": _m("brief.no_voiceover", Verdict.PASS, Scale.RATE, 0.0)}
        led = Ledger()
        flows.apply(ms, TrajectoryFacts(), led, plan_spans=self.PLAN, voiceover_allowed=False)
        nv = ms["brief.no_voiceover"]
        self.assertIs(nv.verdict, Verdict.PASS)               # presence owns the verdict
        self.assertTrue(nv.detail["planned_violation"])
        self.assertEqual(len(nv.detail["planned_voiceover_beats"]), 1)
        self.assertNotIn("speech_over_still_face_windows", nv.detail)
        self.assertIn("plan_vs_brief_voiceover", led.rules)
        self.assertNotIn("narration_exempts_mouth", led.rules)

    def test_a_clean_plan_under_a_ban_is_noted_as_consistent(self):
        ms = {"brief.no_voiceover": _m("brief.no_voiceover", Verdict.PASS, Scale.RATE, 0.0)}
        clean = {"ep1_final.mp4": [{"start": 0, "end": 10, "text": "Victor talks"}]}
        led = Ledger()
        flows.apply(ms, TrajectoryFacts(), led, plan_spans=clean, voiceover_allowed=False)
        self.assertEqual(ms["brief.no_voiceover"].detail["planned_voiceover_beats"], [])
        e = [e for e in led.entries if e.rule == "plan_vs_brief_voiceover"][0]
        self.assertIn("一致", e.note)


# ------------------------------------------------------------ physical ----
PEN_OK = {"coverage": {"frames_total": 132, "frames_with_contact": 117,
                       "frame_contact_rate": 0.886, "pairs_evaluated": 225,
                       "pairs_judged": 181},
          "unresolvable": {"n": 93, "of_judged": 181},
          "flagged": {"n": 21, "rate": 0.116,
                      "by_reason": {"mutual_occlusion": 21}},
          "per_shot": {"0": {"n_pairs": 24, "n_judged": 12, "n_flagged": 1},
                       "1": {"n_pairs": 30, "n_judged": 20, "n_flagged": 0},
                       "2": {"n_pairs": 40, "n_judged": 22, "n_flagged": 20}}}
SHOTS = [{"shot_id": 0, "start": 0.0, "end": 3.0}, {"shot_id": 1, "start": 3.0, "end": 8.0},
         {"shot_id": 2, "start": 8.0, "end": 13.0}]


class TestPhysicalGates(unittest.TestCase):
    def test_flags_carry_their_shots_and_times(self):
        m = by_key(extract.from_physical({"shots": SHOTS, "penetration": PEN_OK}))["physical.penetration"]
        self.assertIs(m.verdict, Verdict.FAIL)
        self.assertEqual([s["shot"] for s in m.detail["flagged_shots"]], [0, 2])
        self.assertEqual(m.detail["flagged_shots"][1]["start"], 8.0)
        self.assertEqual(m.detail["pairs_unresolvable"], 93)

    def test_low_contact_rate_abstains(self):
        pen = {**PEN_OK, "coverage": {**PEN_OK["coverage"], "frame_contact_rate": 0.1}}
        m = by_key(extract.from_physical({"shots": SHOTS, "penetration": pen}))["physical.penetration"]
        self.assertIs(m.verdict, Verdict.ABSTAIN)
        self.assertIn("指示性", m.reason)

    def test_mostly_unresolvable_contacts_abstain(self):
        pen = {**PEN_OK, "coverage": {**PEN_OK["coverage"], "pairs_judged": 20},
               "unresolvable": {"n": 15, "of_judged": 20}}
        m = by_key(extract.from_physical({"shots": SHOTS, "penetration": pen}))["physical.penetration"]
        self.assertIs(m.verdict, Verdict.ABSTAIN)
        self.assertIn("可解析", m.reason)

    def test_unreliable_noise_floors_abstain_scale_drift(self):
        drift = {"coverage": {"no_pair_recurs": False}, "drift_over_noise": {"value": 2.4},
                 "pairs": [{"pair": "chair|person", "shots": [0, 2], "drift_over_noise": None,
                            "noise_floor_unreliable": "3 obs"}]}
        m = by_key(extract.from_physical({"shots": SHOTS, "penetration": PEN_OK,
                                          "scale_drift": drift}))["visual.scale_consistency"]
        self.assertIs(m.verdict, Verdict.ABSTAIN)
        self.assertEqual(m.detail["pairs"][0]["noise_floor_unreliable"], "3 obs")

    def test_class_scale_drift_keeps_raw_pairs_without_scoring(self):
        drift = {"coverage": {"no_pair_recurs": False}, "drift_over_noise": {"value": 2.4},
                 "pairs": [{"pair": "chair|person", "shots": [0, 2], "drift_over_noise": 2.4}]}
        m = by_key(extract.from_physical({"shots": SHOTS, "penetration": PEN_OK,
                                          "scale_drift": drift}))["visual.scale_consistency"]
        self.assertIs(m.verdict, Verdict.ABSTAIN)
        self.assertEqual(m.detail["pairs"][0]["shots"], [0, 2])
        self.assertEqual(m.detail["drift_over_noise"], 2.4)


class TestPlanExplainsPhysical(unittest.TestCase):
    def _pen(self, shots):
        return {"physical.penetration": _m(
            "physical.penetration", Verdict.FAIL, Scale.RATE, 0.1,
            reason="21/225 组接触被判为穿模或互相遮挡", flagged_shots=shots,
            episode="ep1_final.mp4")}

    def test_every_flag_inside_a_smoke_beat_reads_as_instrument(self):
        ms = self._pen([{"shot": 0, "start": 0.0, "end": 3.0, "n_flagged": 1},
                        {"shot": 2, "start": 8.0, "end": 13.0, "n_flagged": 20}])
        plan = {"ep1_final.mp4": [
            {"start": 0, "end": 4, "text": "rain against the window, Maya in silhouette"},
            {"start": 4, "end": 8, "text": "Victor at the whiteboard"},
            {"start": 8, "end": 14, "text": "smoke drifts through the server room"}]}
        led = Ledger()
        flows.apply(ms, TrajectoryFacts(), led, plan_spans=plan)
        m = ms["physical.penetration"]
        self.assertIs(m.verdict, Verdict.FAIL)                  # the pixels shipped
        self.assertTrue(m.detail["depth_unreliable"])
        self.assertIn("边界不可靠", m.reason)
        self.assertEqual(m.detail["planned_beats"][1]["effects"], ["smoke"])
        e = [e for e in led.entries if e.rule == "plan_explains_physical"][0]
        self.assertEqual(e.effect, "reinterpret")

    def test_a_flag_in_a_plain_beat_only_annotates(self):
        ms = self._pen([{"shot": 0, "start": 0.0, "end": 3.0, "n_flagged": 1},
                        {"shot": 2, "start": 8.0, "end": 13.0, "n_flagged": 20}])
        plan = {"ep1_final.mp4": [
            {"start": 0, "end": 4, "text": "rain against the glass"},
            {"start": 8, "end": 14, "text": "Victor and Maya across the desk"}]}
        led = Ledger()
        flows.apply(ms, TrajectoryFacts(), led, plan_spans=plan)
        m = ms["physical.penetration"]
        self.assertNotIn("depth_unreliable", m.detail)
        self.assertEqual(len(m.detail["planned_beats"]), 2)
        self.assertEqual(m.detail["planned_beats"][1]["effects"], [])
        e = [e for e in led.entries if e.rule == "plan_explains_physical"][0]
        self.assertEqual(e.effect, "annotate")

    def test_no_plan_means_no_flow(self):
        ms = self._pen([{"shot": 0, "start": 0.0, "end": 3.0, "n_flagged": 1}])
        led = Ledger()
        flows.apply(ms, TrajectoryFacts(), led, plan_spans={})
        self.assertNotIn("planned_beats", ms["physical.penetration"].detail)
        self.assertNotIn("plan_explains_physical", led.rules)

    def test_drift_worst_pair_gets_its_beats(self):
        ms = {"visual.scale_consistency": _m(
            "visual.scale_consistency", Verdict.FAIL, Scale.RATIO, 3.7, reason="x",
            worst_pair="chair|person", worst_pair_spans=[[0.0, 3.0], [8.0, 13.0]],
            episode="ep1_final.mp4")}
        plan = {"ep1_final.mp4": [{"start": 0, "end": 4, "text": "wide on the lobby"},
                                  {"start": 8, "end": 14, "text": "close on the chair"}]}
        flows.apply(ms, TrajectoryFacts(), Ledger(), plan_spans=plan)
        self.assertEqual(ms["visual.scale_consistency"].detail["worst_pair_planned_beats"],
                         [["wide on the lobby"], ["close on the chair"]])


# -------------------------------------------------------------- visual ----
class TestIdentityCounts(unittest.TestCase):
    REP = {"degeneracy": {"verdict": {"degenerate": False}},
           "character_identity": {
               "per_identity_consistency": {"comparison_scope": "cross_shot", "worst": 0.62, "intra_shot": 0.85},
               "identities": [{"identity": i, "n_faces": 5, "cross_shot": i < 7,
                               "shots": [0, 1]} for i in range(9)]},
           "n_shots": 6}

    def test_counts_ride_on_identity_consistency(self):
        m = by_key(extract.from_visual(self.REP))["visual.identity_consistency"]
        self.assertEqual(m.detail["n_identities"], 9)
        self.assertEqual(m.detail["n_cross_shot_identities"], 7)

    def test_surplus_over_cast_is_named(self):
        ms = by_key(extract.from_visual(self.REP))
        led = Ledger()
        flows.apply(ms, TrajectoryFacts(), led, n_cast=4)
        m = ms["visual.identity_consistency"]
        self.assertEqual(m.detail["identity_surplus"], 1)    # 7 > 4 + 2
        self.assertIn("cast_sizes_identities", led.rules)
        ms = by_key(extract.from_visual(self.REP))
        flows.apply(ms, TrajectoryFacts(), Ledger(), n_cast=7)
        self.assertNotIn("identity_surplus", ms["visual.identity_consistency"].detail)


# ------------------------------------------------------- evidence + exp ----
class TestPlanOnTheClock(unittest.TestCase):
    """plan_spans / text_at on a stub bus."""

    PLAN = [PlannedShot(start=0, end=10, text="INT. LOBBY -- Victor applauds Maya", episode=1),
            PlannedShot(start=10, end=20, text="Maya at her desk, pencil behind ear", episode=1),
            PlannedShot(start=20, end=30, text="Evelyn enters", episode=1),
            PlannedShot(start=0, end=30, text="Lena in the workspace", episode=2)]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name) / "run"
        self.v = {}
        for n in (1, 2):
            p = root / f"output/ep{n}_final.mp4"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")
            self.v[n] = p
        vids = [SimpleNamespace(info=SimpleNamespace(path=self.v[n]), episode=n) for n in (1, 2)]
        facts = TrajectoryFacts()
        facts.continuity.shot_plan = self.PLAN
        self.bus = EvidenceBus(SimpleNamespace(root=root, run_id="t", target=None),
                               SimpleNamespace(videos=vids, found=True),
                               Path(self.tmp.name) / "_out", facts=facts)
        self.brief = BriefFacts(prompt_no=1)
        self.brief.characters = [Character("Maya Chen", "maya"), Character("Victor Hale", "victor")]

    def tearDown(self):
        self.tmp.cleanup()

    def test_plan_spans_follow_the_episode(self):
        self.assertEqual([s["text"][:4] for s in self.bus.plan_spans(self.v[1])],
                         ["INT.", "Maya", "Evel"])
        self.assertEqual([s["text"] for s in self.bus.plan_spans(self.v[2])],
                         ["Lena in the workspace"])
        self.assertEqual(self.bus.text_at(self.v[1], 8, 12),
                         ["INT. LOBBY -- Victor applauds Maya", "Maya at her desk, pencil behind ear"])
        self.assertEqual(self.bus.text_at(self.v[1], 40, 50), [])


# ------------------------------------------------------- closed bucket ----
class TestDiagnosticBucketIsClosed(unittest.TestCase):
    """A check either has a denominator or is deleted (SCORING_SYSTEM B.4).
    The nine keys removed on 2026-09-22 must not come back as measurements,
    and the two promoted ones must be scored."""

    DELETED = ("visual.subject_consistency", "visual.background_consistency",
               "visual.scene_grouping", "brief.characters_placed",
               "visual.reference_identity", "av.mouth_speech",
               "o7.dialogue_repetition", "provenance.segments_recovered",
               "trace.error_rate")

    def test_deleted_keys_are_not_declared(self):
        from stage5_scoring.registry import DIAGNOSTIC
        from stage5_scoring.capabilities import METHOD_DIAGNOSTIC_SOURCES
        self.assertEqual(set(DIAGNOSTIC), set(METHOD_DIAGNOSTIC_SOURCES))
        for k in self.DELETED:
            self.assertNotIn(k, ALL_CHECKS, k)

    def test_unbound_proxies_are_deferred(self):
        from stage5_scoring.registry import SCORED
        from stage5_scoring.registry import DEFERRED
        for key in ("visual.scale_consistency", "av.speaker_consistency"):
            self.assertNotIn(key, SCORED)
            self.assertIn(key, DEFERRED)

    def test_no_extractor_still_emits_a_deleted_key(self):
        rep = {"n_shots": 4,
               "subject": {"inter_shot": {"p10": 0.3}, "intra_shot": {"mean": 0.9}},
               "background": {"within_location": {"p10": 0.3}, "intra_shot": {"mean": 0.9},
                              "declared_grouping": {"source": "agent_shot_plan"}},
               "character_identity": {"reference_identity": {"n_references": 2, "n_matched": 1}},
               "degeneracy": {"verdict": {"degenerate": False}}}
        keys = set(by_key(extract.from_visual(rep)))
        self.assertFalse(keys & set(self.DELETED), keys & set(self.DELETED))
        art = {"meta": {"n_frames": 100, "duration": 4.0},
               "splices": {"n_splices": 2, "splices": [], "segment_frames": [50, 50]},
               "findings": {}}
        self.assertNotIn("provenance.segments_recovered", by_key(extract.from_artifact(art)))
        self.assertNotIn("trace.error_rate", by_key(extract.from_facts(TrajectoryFacts())))


if __name__ == "__main__":
    unittest.main()
