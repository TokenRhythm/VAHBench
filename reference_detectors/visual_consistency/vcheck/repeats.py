"""Footage that plays again later on the same timeline (reference reimplementation).

`scan_video(path)` returns what `stage3_evidence.intervals.repeat_evidence`
reads:

    {"duplicate_union": [[start_s, end_s], ...], "duplicate_seconds": float,
     "n_spans": int, "abstained": ""}

A later stretch counts as a repeat of an earlier one when, at one constant
time offset and for at least `min_run_seconds`:

  * the pictures agree (thumbnail correlation), and
  * the *motion* agrees too (correlation of the change between samples).

The second condition separates a loop from a static shot whose frame-to-frame
change is noise. Stretches without measurable motion can extend a run but
cannot establish one. A third condition separates a loop from footage that
merely looks alike throughout, such as a locked camera with something moving
in front of it or a slow steady camera move: at a true repeat the earlier
footage matches *much better than the frame a moment away does*
(`residual_share`), and better than at the neighbouring offsets. Each test is
also made against the horizontally mirrored earlier footage. Only the later
occurrence is charged: the first showing is genuine footage.

Not covered: repeats played at a different speed, reversed, cropped or
recoloured. Motion that repeats exactly with a fixed period is, to this scan,
the same thing as a loop. Thresholds are stated choices, uncalibrated.
"""
from __future__ import annotations

import numpy as np

from vahref import features as ft
from vahref import media
from vahref.provenance import stamp

PARAMS = {
    "samples_per_second": 4.0,
    "max_samples": 3000,            # longer films are sampled more sparsely
    "frame_similarity_min": 0.92,
    "motion_similarity_min": 0.80,
    "motion_rms_min": 1.0,          # grey levels; below this a step is "still"
    "min_lag_seconds": 1.0,
    # Short chance agreements happen when little moves and it moves along one
    # line; three seconds with three quarters of the steps confirmed did not
    # occur by chance in 50 synthetic trials of that worst case.
    "min_run_seconds": 3.0,
    "min_confirmed_fraction": 0.75,
    "max_gap_steps": 1,
    # Mismatch with the earlier footage, as a share of the mismatch with the
    # adjacent sample of the same footage, must stay below this.
    "residual_share": 0.5,
}

NONE, NEUTRAL, CONFIRMED = 0, 1, 2


def _unit(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = v.reshape(len(v), -1).astype(np.float32)
    flat = flat - flat.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(flat, axis=1)
    ok = norm > 1e-3
    out = np.zeros_like(flat)
    out[ok] = flat[ok] / norm[ok, None]
    return out, norm


def _relation(a, a_ok, d, d_ok, b, e, p):
    """Picture similarity and the per (later k, earlier j) state.

    a/d are the later picture and motion; b/e the earlier ones (possibly
    mirrored, which leaves their validity flags unchanged).
    """
    sim = a @ b.T
    same = (sim >= p["frame_similarity_min"]) & a_ok[:, None] & a_ok[None]
    moving = d_ok[:, None] & d_ok[None]
    still = ~d_ok[:, None] & ~d_ok[None]
    agree = (d @ e.T) >= p["motion_similarity_min"]
    state = np.where(same & moving & agree, CONFIRMED,
                     np.where(same & still, NEUTRAL, NONE)).astype(np.uint8)
    return sim, state


def _diag_mean(sim: np.ndarray, lag: int, k0: int, k1: int) -> float:
    """Mean of sim[k, k - lag] for k in [k0, k1], where defined."""
    lo = max(k0, lag)
    if lag < 0 or lo > k1:
        return -1.0
    ks = np.arange(lo, k1 + 1)
    return float(sim[ks, ks - lag].mean())


def _diagonal_runs(sim: np.ndarray, state: np.ndarray, min_lag: int, min_run: int,
                   p, adjacent: np.ndarray) -> list[tuple[int, int]]:
    """(first, last) later-sample indices of accepted runs, any offset.

    `adjacent[k]` is the picture similarity of sample k and sample k - 1 of
    the same footage. A run is kept only where

      * its offset is a ridge of the picture similarity: it agrees at least as
        well as one sample earlier and one later, and
      * its mismatch with the earlier footage is a small share of the mismatch
        between adjacent samples.

    A slow steady camera move resembles itself more the smaller the offset, so
    it has no ridge. A locked camera with something moving in front of it
    resembles itself at every offset about as well as at the next sample, so
    it fails the second test. A loop passes both at exactly its own period.
    """
    n = len(state)
    found = []
    if not (state == CONFIRMED).any():
        return found
    gap = int(p["max_gap_steps"])
    for lag in range(min_lag, n - min_run + 1):
        diag = np.diagonal(state, offset=-lag)          # state[k, k - lag], k = lag..n-1
        if np.count_nonzero(diag == CONFIRMED) < max(2, min_run // 2):
            continue
        hit = diag != NONE
        if gap:
            for a, b in ft.runs(~hit):
                if 0 < a and b < len(hit) - 1 and b - a + 1 <= gap:
                    hit[a:b + 1] = True
        for a, b in ft.runs(hit):
            length = b - a + 1
            if length < min_run:
                continue
            conf = np.count_nonzero(diag[a:b + 1] == CONFIRMED)
            if conf / length < p["min_confirmed_fraction"]:
                continue
            k0, k1 = a + lag, b + lag
            here = _diag_mean(sim, lag, k0, k1)
            if here < _diag_mean(sim, lag - 1, k0, k1) or here < _diag_mean(sim, lag + 1, k0, k1):
                continue
            step_mismatch = 1.0 - float(adjacent[max(1, k0):k1 + 1].mean())
            if (1.0 - here) <= p["residual_share"] * step_mismatch:
                found.append((k0, k1))
    return found


def scan_thumbs(thumbs: np.ndarray, fps: float, p=PARAMS) -> dict:
    """Repeat scan on per-frame thumbnails (n, s, s), any s divisible by 16."""
    n = len(thumbs)
    fps = float(fps) if fps and fps > 0 else 24.0
    step = max(1, int(round(fps / p["samples_per_second"])))
    step = max(step, -(-n // int(p["max_samples"])))
    idx = np.arange(0, n, step)
    dt = step / fps
    min_lag = max(1, int(round(p["min_lag_seconds"] / dt)))
    min_run = max(2, int(round(p["min_run_seconds"] / dt)))
    base = {"duplicate_union": [], "duplicate_seconds": 0.0, "n_spans": 0,
            "abstained": "", "sample_step_s": round(dt, 4), "n_samples": int(len(idx)),
            "implementation": stamp("visual.repeats", p)}
    if len(idx) < min_lag + min_run + 1:
        return base                                    # too short to hold a repeat
    t = np.asarray(thumbs)[idx].astype(np.float32)
    s = t.shape[1] // 16
    if s > 1:
        t = t.reshape(len(t), 16, s, 16, s).mean(axis=(2, 4))
    a, an = _unit(t)
    a_ok = an > 16 * 0.5                                # not a flat frame
    d, dn = _unit(t[1:] - t[:-1])
    d_ok = (dn / 16.0) >= p["motion_rms_min"]
    a, a_ok = a[:-1], a_ok[:-1]                         # picture k goes with motion k -> k+1
    tf = t[:, :, ::-1]
    af, _ = _unit(tf)
    df, _ = _unit(tf[1:] - tf[:-1])
    adjacent = np.ones(len(a), dtype=np.float32)
    adjacent[1:] = np.einsum("ij,ij->i", a[1:], a[:-1])
    last = len(d) - 1
    spans = []
    for b, e, kind in ((a, d, "same"), (af[:-1], df, "mirrored")):
        sim, state = _relation(a, a_ok, d, d_ok, b, e, p)
        for k0, k1 in _diagonal_runs(sim, state, min_lag, min_run, p, adjacent):
            # Motion k spans samples k and k + 1; a run reaching the last
            # sample runs to the end of the picture.
            end = n / fps if k1 >= last else (idx[k1] + step) / fps
            spans.append((idx[k0] / fps, min(n / fps, end), kind))
    merged = []
    for s0, s1, _ in sorted(spans):
        if merged and s0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], s1)
        else:
            merged.append([s0, s1])
    base.update(duplicate_union=[[round(float(x), 3), round(float(y), 3)] for x, y in merged],
                duplicate_seconds=round(float(sum(y - x for x, y in merged)), 3),
                n_spans=len(merged),
                n_mirrored_runs=sum(1 for *_, k in spans if k == "mirrored"))
    return base


def scan_video(path) -> dict:
    try:
        meta = media.probe(path)
        feat = ft.from_video(path)
    except (media.MediaError, OSError) as exc:
        return {"duplicate_union": None, "duplicate_seconds": 0.0, "n_spans": 0,
                "abstained": f"无法解码：{exc}", "implementation": stamp("visual.repeats", PARAMS)}
    if feat.n == 0:
        return {"duplicate_union": None, "duplicate_seconds": 0.0, "n_spans": 0,
                "abstained": "没有可解码的帧", "implementation": stamp("visual.repeats", PARAMS)}
    fps = media.frame_clock(meta, feat.n)
    out = scan_thumbs(feat.thumbs, fps)
    out["duration"] = round(feat.n / fps, 4)
    return out
