"""seam_checks -> one measurement -> leaf 1.2.2.b.

The fixture follows `seam_checks/run_seam_check.py` on claudecode's
prompt_01 episode 1 (10 seams, 11 source clips, 6.7 s trimmable of 22.8 s
still): the shape is pinned here so a renamed field fails loudly instead of
scoring a video as clean.
"""
from __future__ import annotations

import os
import sys
import unittest


import stage4_checks.reports as extract
import stage4_checks.cross_checks as flows                                # noqa: E402
from shared.types import Scale
from shared.types import Verdict                          # noqa: E402
from stage5_scoring.registry import SCORED                        # noqa: E402
from stage5_scoring.node import score_leaf
from stage5_scoring.node import DEFAULT_CUT_POINTS    # noqa: E402
from stage5_scoring.video.quality.consistency import CONSISTENCY  # noqa: E402
from stage4_checks.cross_checks import Ledger                                  # noqa: E402


def seam_row(k, t, p_head, p_pace, p_vis=0.001):
    return {"seam": k, "frame": int(t * 24), "t_s": t, "mechanism": "unknown",
            "head_motion": 0.3, "p_head": p_head, "tail_motion": 1.0, "p_tail": 0.5,
            "dead_frames": 12, "dead_s": 0.5, "ramp_s": 1.5, "pace_logratio": 1.2,
            "p_pace": p_pace, "dlum": -5.0, "p_vis": p_vis, "seam_fdiff": 70.0,
            "seam_hist": 0.3, "vis_score": 5000.0, "p_ramp": 0.2, "excess_ramp_s": 1.5,
            "excess_dead_s": 0.5}


REPORT = {
    "video": "ep1_e1.mp4", "meta": {"fps": 24.0, "n_frames": 1440, "duration": 60.0},
    "alpha": 0.05, "seams_given": 10, "status": "ok",
    "seams": [seam_row(0, 3.0, 0.20, 0.001), seam_row(1, 8.0, 0.01, 0.007),
              seam_row(2, 13.0, 0.02, 0.003), seam_row(3, 18.0, 0.30, 0.001),
              seam_row(4, 23.0, 0.01, 1.0)],
    "hidden_cuts": [{"frame": 700, "t_s": 29.2, "score": 6000.0, "fdiff": 80.0, "hist": 0.4}],
    "summary": {"n_seams": 5, "dead_head_rate": 0.6, "dead_tail_rate": 0.0, "pace_break_rate": 0.8,
                "slow_start_rate": 0.2, "total_dead_s": 2.5, "total_ramp_s": 7.5,
                "excess_ramp_s": 7.5, "excess_dead_s": 2.5, "n_hidden_cuts": 1,
                "n_visible_seams": 5, "q_dead": 0.51, "q_ramp": 1.0},
    "calibration": {"n": 330, "ks": 0.072, "frac_below_alpha": 0.042,
                    "fake_ramp_median_s": 0.0, "fake_dead_median_s": 0.0},
    "regret": {"status": "ok", "clips": [], "slow_clips": [], "trimmable_s": 6.711,
               "untrimmable_s": 16.125, "shipped_s": 22.833},
    "alignment": [],
}


def one(rep):
    ms = extract.from_seam(rep)
    assert len(ms) == 1, ms
    return ms[0]


class TestFromSeam(unittest.TestCase):
    def test_motion_change_without_excess_still_head_is_not_broken(self):
        row = dict(seam_row(0, 3, .9, .001), dead_s=0, excess_dead_s=0)
        rep = dict(REPORT, seams=[row], hidden_cuts=[], summary={"n_seams": 1})
        self.assertEqual(one(rep).value, 0)

    def test_broken_cuts_over_all_cuts_seen(self):
        m = one(REPORT)
        self.assertEqual(m.key, "seam.cut_continuity")
        self.assertEqual(m.verdict, Verdict.FAIL)
        self.assertEqual(m.scale, Scale.RATE)
        # seams 0,1,2,3,4 are all broken on head or pace, plus one hidden cut
        self.assertEqual((m.detail["k"], m.detail["n"]), (6, 6))
        self.assertEqual((m.detail["dead_heads"], m.detail["pace_breaks"], m.detail["hidden_cuts"]), (3, 4, 1))
        self.assertEqual(m.detail["hidden_at"], [29.2])
        self.assertAlmostEqual(m.detail["trimmable_s"], 6.711)
        self.assertAlmostEqual(m.detail["material_s"], 16.125)

    def test_a_cut_broken_two_ways_counts_once(self):
        rep = dict(REPORT, seams=[seam_row(0, 3.0, 0.01, 0.01), seam_row(1, 8.0, 0.5, 0.5)],
                   hidden_cuts=[], summary=dict(REPORT["summary"], n_seams=2, n_hidden_cuts=0))
        m = one(rep)
        self.assertEqual((m.detail["k"], m.detail["n"]), (1, 2))
        self.assertAlmostEqual(m.value, 0.5)

    def test_no_source_clips_only_drops_the_regret_detail(self):
        m = one(dict(REPORT, regret=None))
        self.assertEqual(m.verdict, Verdict.FAIL)
        self.assertNotIn("shipped_s", m.detail)

    def test_single_segment_is_not_applicable(self):
        rep = {"video": "x", "meta": {"fps": 24.0, "n_frames": 100, "duration": 4.0},
               "status": "no_seams", "seams": [], "hidden_cuts": [], "summary": None,
               "calibration": None, "regret": None}
        self.assertEqual(one(rep).verdict, Verdict.NOT_APPLICABLE)

    def test_no_visible_seam_leaves_hidden_cuts_unknown(self):
        rep = dict(REPORT, summary=dict(REPORT["summary"], n_visible_seams=0))
        m = one(rep)
        self.assertEqual(m.detail["hidden_cuts"], 0)
        self.assertEqual(m.detail["n"], 5)

    def test_clean_report_passes(self):
        rep = dict(REPORT, seams=[seam_row(0, 3.0, 0.5, 0.5)], hidden_cuts=[],
                   summary=dict(REPORT["summary"], n_seams=1, n_hidden_cuts=0))
        self.assertEqual(one(rep).verdict, Verdict.PASS)


class TestTree(unittest.TestCase):
    def test_key_is_a_source_of_identity_consistency(self):
        self.assertIn("seam.cut_continuity", SCORED)
        leaf = next(lf for lf in CONSISTENCY.leaves if lf.id == "1.2.2.b")
        self.assertEqual(leaf.sources, ("visual.identity_consistency", "seam.cut_continuity"))

    def test_leaf_averages_the_two_sources(self):
        leaf = next(lf for lf in CONSISTENCY.leaves if lf.id == "1.2.2.b")
        ms = {"visual.identity_consistency": extract._m("visual.identity_consistency", Verdict.PASS),
              "seam.cut_continuity": one(REPORT)}
        r = score_leaf(leaf, ms, DEFAULT_CUT_POINTS)
        self.assertEqual(r.n_measured, 2)
        self.assertAlmostEqual(r.score, 0.5)        # PASS and a rate past the suspect cut
        r = score_leaf(leaf, {"visual.identity_consistency": ms["visual.identity_consistency"],
                              "seam.cut_continuity": one(dict(REPORT, status="no_seams"))},
                       DEFAULT_CUT_POINTS)
        self.assertEqual((r.n_measured, r.score), (1, 1.0))

    def test_attribution_flow_writes_the_split(self):
        ms = {"seam.cut_continuity": one(REPORT)}
        ledger = Ledger()
        flows.handles_attribution(ms, ledger)
        notes = [e for e in ledger.entries if e.rule == "handles_attribution"]
        self.assertEqual(len(notes), 1)
        self.assertIn("6.7s", notes[0].note)
        self.assertIn("16.1s", notes[0].note)
        ledger = Ledger()
        flows.handles_attribution({"seam.cut_continuity": one(dict(REPORT, regret=None))}, ledger)
        self.assertEqual(ledger.entries, [])


if __name__ == "__main__":
    unittest.main()
