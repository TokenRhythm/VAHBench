#!/usr/bin/env python
"""Reference cut-continuity check (reimplementation).

For every cut the evaluator names (`--seam-times`), in the report shape
`stage4_checks.reports.from_seam` reads:

    p_head          how unusual the motion just after the cut is, among windows
                    of the same film that contain no cut. Image-to-video clips
                    open on a still; a cut that lands on one has a low p_head.
    dead_s          how long the picture stays still after the cut
    excess_dead_s   that, beyond the film's typical still length away from cuts
    p_pace          how unusual the change of motion across the cut is
    p_vis           how unusual the picture jump at the cut is ("visible")
    hidden_cuts     isolated picture jumps elsewhere, at least half as strong
                    as the weakest visible named cut: cuts the stated structure
                    does not have. With no visible named cut there is no
                    yardstick, and none is reported.

p values are empirical: (1 + count of reference windows at least as extreme)
/ (1 + number of reference windows). They are relative to this film only.
A rank alone would call a difference too small to see "unusual" in a film
whose motion is very even, so two effect-size conditions apply first:
`p_head` is 1.0 unless the head moves less than `still_head_share` of the
film's median window, and `p_vis` is 1.0 unless the jump is an isolated spike
by the same rule the artifact check uses for cuts. The untouched ranks are
kept as `p_head_rank` and `p_vis_rank`.

Not reimplemented: `regret` (which still seconds the source clips' slack
would have let the agent trim). It is reported as null; the evaluator uses it
only as an explanatory detail. The original detector's statistics and its
calibration procedure were not recoverable; every choice below is this
reimplementation's own and is uncalibrated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                     # noqa: E402

from vahref import features as ft                      # noqa: E402
from vahref import media                               # noqa: E402
from vahref.provenance import stamp                    # noqa: E402

PARAMS = {
    "alpha": 0.05,
    "window_seconds": 0.5,
    "snap_frames": 3,              # a named time is snapped to the nearest jump
    "still_mad_max": 0.50,         # same stillness level as the freeze check
    "min_reference_windows": 20,
    "still_head_share": 0.5,       # of the film's median window motion
    "cut_z_min": 10.0, "cut_mad_min": 10.0,
    "hidden_strength_share": 0.5,  # of the weakest visible named cut
    "edge_seconds": 1.0,
}


def _p_low(value: float, ref: np.ndarray) -> float:
    return float((1 + np.count_nonzero(ref <= value)) / (1 + len(ref)))


def _p_high(value: float, ref: np.ndarray) -> float:
    return float((1 + np.count_nonzero(ref >= value)) / (1 + len(ref)))


def _dead_run(m: np.ndarray, start: int, limit: float) -> int:
    k, n = 0, len(m)
    while start + k < n and m[start + k] <= limit:
        k += 1
    return k


def analyse(m: np.ndarray, fps: float, seam_times: list[float], p=PARAMS,
            local: np.ndarray | None = None) -> dict:
    """`m[i]`: picture change between frame i-1 and i, above the subtitle band.

    `local[i]` is the largest change in any block of the frame; stillness is
    judged on it, so a small moving region does not read as a still picture.
    Without it the mean change `m` is used.
    """
    m = np.asarray(m, dtype=np.float64)
    quiet = m if local is None else np.asarray(local, dtype=np.float64)
    n = len(m)
    fps = float(fps) if fps and fps > 0 else 24.0
    meta = {"fps": fps, "n_frames": n, "duration": round(n / fps, 6)}
    if not seam_times:
        return {"meta": meta, "status": "no_seams", "alpha": p["alpha"], "seams": [],
                "hidden_cuts": [], "summary": None, "calibration": None, "regret": None}
    H = max(2, int(round(fps * p["window_seconds"])))
    snap = int(p["snap_frames"])
    frames = []
    for t in seam_times:
        f = int(round(float(t) * fps))
        lo, hi = max(1, f - snap), min(n, f + snap + 1)
        if lo < hi:
            f = lo + int(np.argmax(m[lo:hi]))
        frames.append(min(max(f, 1), n - 1))

    z = ft.robust_z(m, half=max(2, int(round(fps))))
    spike = (m >= p["cut_mad_min"]) & (z >= p["cut_z_min"])
    edge = int(round(fps * p["edge_seconds"]))

    # Reference positions: far enough from every named cut, every other jump
    # and both ends that a window on either side is ordinary footage.
    blocked = np.zeros(n, dtype=bool)
    for f in [*frames, *np.flatnonzero(spike).tolist()]:
        blocked[max(0, f - H - 1): min(n, f + H + 2)] = True
    blocked[: H + 1] = True
    blocked[max(0, n - H - 1):] = True
    ref = np.flatnonzero(~blocked)
    csum = np.concatenate([[0.0], np.cumsum(m)])

    def head(f):                      # mean change over the H frames after f
        a, b = f + 1, min(n, f + 1 + H)
        return float((csum[b] - csum[a]) / max(1, b - a))

    def judged(f):                    # a full window on both sides of the cut
        return f + 1 + H <= n and f - H >= 0

    def tail(f):                      # mean change over the H frames before f
        a, b = max(0, f - H), f
        return float((csum[b] - csum[a]) / max(1, b - a))

    eps = 0.05
    enough = len(ref) >= p["min_reference_windows"]
    if enough:
        a, b = ref + 1, np.minimum(n, ref + 1 + H)
        ref_head = (csum[b] - csum[a]) / np.maximum(1, b - a)
        a2, b2 = np.maximum(0, ref - H), ref
        ref_tail = (csum[b2] - csum[a2]) / np.maximum(1, b2 - a2)
        ref_pace = np.abs(np.log((ref_tail + eps) / (ref_head + eps)))
        ref_jump = m[ref]
        typical_head = float(np.median(ref_head))
        # Typical still length away from any cut, from a spread of positions.
        probe_at = ref[:: max(1, len(ref) // 400)]
        fake_dead = np.array([_dead_run(quiet, int(g) + 1, p["still_mad_max"]) for g in probe_at]) / fps
        fake_dead_median = float(np.median(fake_dead))
    else:
        ref_head = ref_pace = ref_jump = np.zeros(0)
        fake_dead_median, typical_head = 0.0, 0.0

    seams = []
    for k, (t, f) in enumerate(zip(seam_times, frames)):
        hm, tm = head(f), tail(f)
        dead = _dead_run(quiet, f + 1, p["still_mad_max"])
        dead_s = dead / fps
        pace = abs(float(np.log((tm + eps) / (hm + eps))))
        if not judged(f):
            # Too close to either end for a window: nothing to compare.
            seams.append({"seam": k, "frame": int(f), "t_s": float(t),
                          "t_snapped_s": round(f / fps, 4), "head_motion": None,
                          "tail_motion": None, "p_head_rank": None, "p_head": None,
                          "dead_frames": int(dead), "dead_s": round(dead_s, 4),
                          "excess_dead_s": 0.0, "pace_logratio": None, "p_pace": None,
                          "seam_fdiff": round(float(m[f]), 4), "p_vis_rank": None,
                          "p_vis": None, "note": "no full window beside this cut"})
            continue
        seams.append({
            "seam": k, "frame": int(f), "t_s": float(t), "t_snapped_s": round(f / fps, 4),
            "head_motion": round(hm, 4), "tail_motion": round(tm, 4),
            "p_head_rank": _p_low(hm, ref_head) if enough else None,
            "p_head": ((_p_low(hm, ref_head) if hm <= p["still_head_share"] * typical_head else 1.0)
                       if enough else None),
            "dead_frames": int(dead), "dead_s": round(dead_s, 4),
            "excess_dead_s": round(max(0.0, dead_s - fake_dead_median), 4),
            "pace_logratio": round(pace, 4),
            "p_pace": _p_high(pace, ref_pace) if enough else None,
            "seam_fdiff": round(float(m[f]), 4),
            "p_vis_rank": _p_high(float(m[f]), ref_jump) if enough else None,
            "p_vis": ((_p_high(float(m[f]), ref_jump) if spike[f] else 1.0) if enough else None),
        })
    alpha = p["alpha"]
    visible = [s for s in seams if s["p_vis"] is not None and s["p_vis"] < alpha]

    hidden = []
    if visible:
        need = p["hidden_strength_share"] * min(s["seam_fdiff"] for s in visible)
        near = np.zeros(n, dtype=bool)
        for f in frames:
            near[max(0, f - H): min(n, f + H + 1)] = True
        for i in np.flatnonzero(spike & ~near):
            if i < edge or i >= n - edge or m[i] < need:
                continue
            lo, hi = max(1, i - 2), min(n, i + 3)
            if m[i] >= m[lo:hi].max() and not (hidden and i - hidden[-1]["frame"] <= 2):
                hidden.append({"frame": int(i), "t_s": round(i / fps, 4),
                               "score": round(float(m[i]), 4)})

    judged = [s for s in seams if s["p_head"] is not None]
    summary = {
        "n_seams": len(seams), "n_visible_seams": len(visible), "n_hidden_cuts": len(hidden),
        "dead_head_rate": (sum(s["p_head"] < alpha for s in judged) / len(judged)) if judged else None,
        "pace_break_rate": (sum(s["p_pace"] < alpha and s["excess_dead_s"] > 0 for s in judged)
                            / len(judged)) if judged else None,
        "total_dead_s": round(sum(s["dead_s"] for s in seams), 4),
        "excess_dead_s": round(sum(s["excess_dead_s"] for s in seams), 4),
    }
    return {"meta": meta, "status": "ok" if enough else "insufficient_reference",
            "alpha": alpha, "seams_given": len(seam_times), "seams": seams,
            "hidden_cuts": hidden, "summary": summary if enough else None,
            "calibration": {"n": int(len(ref)), "fake_dead_median_s": round(fake_dead_median, 4),
                            "median_window_motion": round(typical_head, 4),
                            "method": "empirical p values against cut-free windows of the same film"},
            "regret": None}


def check(path, seam_times: list[float]) -> dict:
    meta = media.probe(path)
    feat = ft.from_video(path)
    if feat.n == 0:
        raise media.MediaError("no decodable frames")
    fps = media.frame_clock(meta, feat.n)
    rep = analyse(feat.mad_top, fps, seam_times, local=feat.mad_block)
    rep["video"] = str(path)
    rep["implementation"] = stamp("seam", PARAMS, {"regret": "源片段裁剪余量归因未重新实现"})
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seam-times", default="")
    ap.add_argument("--source-clips", nargs="*", default=[])
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    try:
        times = [float(x) for x in a.seam_times.split(",") if x.strip()]
    except ValueError:
        print("seam check failed: --seam-times must be comma-separated seconds", file=sys.stderr)
        return 2
    try:
        rep = check(a.video, times)
    except (media.MediaError, OSError) as exc:
        print(f"seam check failed: {exc}", file=sys.stderr)
        return 2
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
