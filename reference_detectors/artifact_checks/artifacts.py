"""Frame-level picture defects and splice structure (reference reimplementation).

Reports, in the shape `stage4_checks.reports.from_artifact` and the
completeness ledger read:

    findings.black_frames / solid_frames / frozen_runs   {present, n_frames, fraction, segments}
    findings.tail_freeze                                 {present, held_frames, held_seconds}
    findings.join_artifacts                              {present, n_join_artifacts, join_artifacts, window_frames}
    splices                                              {n_splices, segment_frames, splices}

Not measured here and therefore absent: `flicker`, `burned_in_subtitles`.
Every threshold below is a stated choice of this reimplementation. None was
recovered from the original detector and none is calibrated on labels.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                    # noqa: E402

from vahref import features as ft                     # noqa: E402
from vahref import media                              # noqa: E402
from vahref.provenance import stamp                   # noqa: E402

PARAMS = {
    # A black frame: dark on average and without a bright region.
    "black_mean_max": 12.0, "black_p99_max": 40.0,
    # A solid frame: one flat tone that is not black.
    "solid_std_max": 3.0,
    # Frame i repeats frame i-1 when no 8x8 block of the 64x64 analysis frame
    # changes by more than this on average (grey levels, 0-255). The largest
    # block is used, not the whole-frame mean: a mouth moving in a locked-off
    # shot barely moves the mean and is not a held frame.
    "freeze_mad_max": 0.50,
    # A re-encoded keyframe inside a held run may differ this much.
    "freeze_bridge_mad_max": 2.5,
    # Identical frames shown for at least this long are a held run.
    "freeze_min_seconds": 1.0,
    # A held ending. Longer than the evaluator's one-second edge exemption,
    # so an ordinary fade or a short last-frame hold is not padding.
    "tail_min_seconds": 1.5,
    # A cut: an isolated jump of the picture above the subtitle band.
    "cut_z_min": 10.0, "cut_mad_min": 10.0, "cut_local_frames": 1,
    # A segment shorter than this between two cuts is a join glitch.
    "sliver_frames": 8,
    "join_window_frames": 2,
}


def _segments(mask, fps: float) -> list[dict]:
    return [{"start_frame": a, "end_frame": b, "n_frames": b - a + 1,
             "start_s": round(a / fps, 4), "end_s": round((b + 1) / fps, 4)}
            for a, b in ft.runs(mask)]


def _finding(mask, fps: float, n: int) -> dict:
    k = int(np.count_nonzero(mask))
    return {"present": k > 0, "n_frames": k,
            "fraction": (k / n) if n else 0.0, "segments": _segments(mask, fps)}


def frozen_mask(mad: np.ndarray, fps: float, p=PARAMS) -> np.ndarray:
    """True for every frame that only repeats the one before it, inside a
    held run of at least `freeze_min_seconds`. `mad` is the per-frame local
    change (`FrameFeatures.mad_block`)."""
    n = len(mad)
    held = np.zeros(n, dtype=bool)
    if n < 2:
        return held
    held[1:] = mad[1:] <= p["freeze_mad_max"]
    # A refreshed keyframe differs faintly from the frame before it and the
    # frame after differs back: bridge up to two such frames inside a hold.
    for a, b in ft.runs(~held):
        if (0 < a and b < n - 1 and b - a + 1 <= 2 and held[a - 1] and held[b + 1]
                and mad[a:b + 1].max() <= p["freeze_bridge_mad_max"]):
            held[a:b + 1] = True
    # `held` marks repeats, so k marks mean k + 1 identical frames on screen.
    need = max(2, int(round(fps * p["freeze_min_seconds"])))
    out = np.zeros(n, dtype=bool)
    for a, b in ft.runs(held):
        if b - a + 2 >= need:
            out[a:b + 1] = True
    return out


def find_cuts(mad_top: np.ndarray, fps: float, p=PARAMS):
    """Frame indices where a new segment starts, with their z scores."""
    n = len(mad_top)
    if n < 3:
        return [], np.zeros(n)
    z = ft.robust_z(mad_top, half=max(2, int(round(fps))))
    k = int(p["cut_local_frames"])
    cuts = []
    for i in range(1, n):
        if mad_top[i] < p["cut_mad_min"] or z[i] < p["cut_z_min"]:
            continue
        lo, hi = max(1, i - k), min(n, i + k + 1)
        if mad_top[i] >= mad_top[lo:hi].max():
            if cuts and i - cuts[-1] <= k:      # equal neighbours: keep the first
                continue
            cuts.append(i)
    return cuts, z


def analyse_features(feat: ft.FrameFeatures, fps: float, *,
                     container_duration: float | None = None, p=PARAMS) -> dict:
    n = feat.n
    fps = float(fps) if fps and fps > 0 else 24.0
    black = (feat.mean <= p["black_mean_max"]) & (feat.p99 <= p["black_p99_max"])
    solid = (feat.std <= p["solid_std_max"]) & ~black
    local = feat.mad_block if feat.mad_block is not None else feat.mad
    frozen = frozen_mask(local, fps, p)

    # A held ending. Black frames are left to the black-frame finding and its
    # edge exemption: a fade-out that ends on black is not a frozen picture.
    tail_frames = 0
    if n >= 2:
        i = n - 1
        while i >= 1 and local[i] <= p["freeze_mad_max"] and not black[i]:
            tail_frames += 1
            i -= 1
    tail_s = tail_frames / fps
    tail_present = tail_s >= p["tail_min_seconds"]

    cuts, z = find_cuts(feat.mad_top, fps, p)
    spike = (feat.mad_top >= p["cut_mad_min"]) & (z >= p["cut_z_min"]) if n else np.zeros(0, bool)
    bounds = [0, *cuts, n]
    segment_frames = [b - a for a, b in zip(bounds, bounds[1:]) if b > a] if n else []
    splices = [{"frame": int(i), "t_s": round(i / fps, 4), "z": round(float(z[i]), 2)}
               for i in cuts]

    w = int(p["join_window_frames"])
    seg_len_before = {c: bounds[k + 1] - bounds[k] for k, c in enumerate(cuts)}
    seg_len_after = {c: bounds[k + 2] - bounds[k + 1] for k, c in enumerate(cuts)}
    joins, absorbed = [], set()
    for c in cuts:
        if c in absorbed:
            continue
        issues = []
        lo, hi = max(0, c - w), min(n, c + w)
        if black[lo:hi].any():
            issues.append("dark")
        if solid[lo:hi].any():
            issues.append("flat")
        # A one-frame sliver shows as two jumps on adjacent frames, of which
        # only the larger is kept as a cut.
        adjacent = (c + 1 < n and spike[c + 1]) or (c - 1 >= 1 and spike[c - 1])
        if (seg_len_after[c] < p["sliver_frames"] or seg_len_before[c] < p["sliver_frames"]
                or adjacent):
            issues.append("sliver")
            # The cut that closes a sliver is the same join, not a second one.
            nxt = [x for x in cuts if x > c]
            if nxt and seg_len_after[c] < p["sliver_frames"]:
                absorbed.add(nxt[0])
        if issues:
            joins.append({"frame": int(c), "t_s": round(c / fps, 4), "issues": issues})

    picture_s = n / fps
    return {
        "meta": {"fps": fps, "n_frames": n, "duration": round(picture_s, 6),
                 "container_duration": container_duration,
                 "frame_times": "index / average frame rate"},
        "splices": {"n_splices": len(cuts), "z_cut": p["cut_z_min"],
                    "z_p99": round(float(np.percentile(z, 99)), 2) if n else None,
                    "subtitle_band_excluded": True,
                    "segment_frames": segment_frames, "splices": splices},
        "findings": {
            "black_frames": _finding(black, fps, n),
            "solid_frames": _finding(solid, fps, n),
            "frozen_runs": _finding(frozen, fps, n),
            "tail_freeze": {"present": bool(tail_present), "held_frames": int(tail_frames),
                            "held_seconds": round(tail_s, 4)},
            "join_artifacts": {"present": bool(joins), "n_join_artifacts": len(joins),
                               "join_artifacts": joins, "window_frames": w},
        },
        "implementation": stamp("artifact", p, {
            "flicker": "not reimplemented",
            "burned_in_subtitles": "not reimplemented"}),
    }


def analyse(path) -> dict:
    meta = media.probe(path)
    if not meta["has_video"]:
        raise media.MediaError("no video stream")
    feat = ft.from_video(path)
    if feat.n == 0:
        raise media.MediaError("no decodable frames")
    fps = media.frame_clock(meta, feat.n)
    rep = analyse_features(feat, fps, container_duration=meta["duration"])
    rep["video"] = str(path)
    rep["meta"].update(width=meta["width"], height=meta["height"],
                       nominal_fps=meta["fps"], probe_n_frames=meta["n_frames"])
    f = rep["findings"]
    rep["n_defects"] = sum(1 for k in ("black_frames", "solid_frames", "frozen_runs",
                                       "tail_freeze", "join_artifacts") if f[k]["present"])
    rep["clean"] = rep["n_defects"] == 0
    return rep
