"""Was the delivery stretched beyond the footage its source clips hold?

Two layers, kept apart on purpose:

`check_retiming(final_meta, source_metas)` reads metadata only. It can clear a
delivery (the picture is no longer than its sources) and it can name a
*candidate* slowdown (longer picture, same or similar frame count), but it
cannot prove one: equal frame counts say nothing about whether the frames are
the same footage. A candidate is therefore `undetermined`, never a failure.

`verify_correspondence(final_thumbs, source_thumbs)` compares pictures. Under
a uniform stretch, the delivered frame at relative position x must show the
source frame at the same relative position. The slowdown is verified only if
that holds and two controls fail: a deliberately misaligned pairing (static
content matches everything), and, when the frame counts differ, pairing by
frame number (sources played at their own speed and followed by something
else, such as a held last frame, also make the picture longer). Then
`check_retiming(..., correspondence=...)` turns the candidate into a failure
with `correspondence_verified: True`.

Covered: uniform slowdowns, whether by lowering the frame rate or by
duplicating / interpolating frames. Not covered: non-uniform speed ramps, and
slowdowns of footage whose source clips the run did not keep.
"""
from __future__ import annotations

import numpy as np

PARAMS = {
    "duration_tolerance": 0.02,       # relative; below this the lengths agree
    "frame_count_tolerance": 0.01,    # relative; "frame count preserved"
    "n_samples": 24,
    "match_correlation_min": 0.90,
    "match_fraction_min": 0.80,
    "search_frames": 1,               # rounding slack around each sample
    "control_margin": 0.10,           # aligned must beat misaligned by this share
}


def _result(ok, undetermined, detail, **extra) -> dict:
    out = {"ok": bool(ok), "undetermined": bool(undetermined), "detail": detail,
           "correspondence_verified": False}
    out.update(extra)
    return out


def check_retiming(final: dict, sources: list[dict], *, correspondence: dict | None = None,
                   p=PARAMS) -> dict:
    if not sources:
        return _result(True, True, "no source clips found to compare against")
    fin_s, fin_n = final.get("video_duration"), final.get("n_frames")
    src_s = [s.get("video_duration") for s in sources]
    src_n = [s.get("n_frames") for s in sources]
    if not fin_s or not fin_n or any(not x for x in src_s) or any(not x for x in src_n):
        return _result(True, True, "picture duration or frame count unavailable")
    total_s, total_n = float(sum(src_s)), int(sum(src_n))
    fin_s, fin_n = float(fin_s), int(fin_n)
    preserved = abs(fin_n - total_n) <= max(1, p["frame_count_tolerance"] * total_n)
    base = {"source_duration": round(total_s, 6), "final_duration": round(fin_s, 6),
            "source_frames": total_n, "final_frames": fin_n,
            "frame_count_preserved": bool(preserved),
            "stretch": round(fin_s / total_s, 6) if total_s > 0 else None}
    if fin_s <= total_s * (1.0 + p["duration_tolerance"]):
        return _result(True, False, "", **base)
    detail = (f"picture {fin_s:.2f}s exceeds source footage {total_s:.2f}s"
              + ("; frame count preserved" if preserved else ""))
    if correspondence and correspondence.get("verified"):
        out = _result(False, False, detail + "; frames correspond to the sources", **base)
        out["correspondence_verified"] = True
        out["correspondence"] = correspondence
        return out
    out = _result(True, True, detail + "; frame correspondence not established", **base)
    if correspondence:
        out["correspondence"] = correspondence
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a, b = a - a.mean(), b - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den < 1e-6:
        return 0.0                      # a flat frame identifies nothing
    return float(a @ b) / den


def verify_correspondence(final_thumbs: np.ndarray, source_thumbs: np.ndarray, p=PARAMS) -> dict:
    """Does delivered frame at relative position x show the source frame at x?"""
    nf, ns = len(final_thumbs), len(source_thumbs)
    if nf < 4 or ns < 4:
        return {"verified": False, "reason": "too few frames"}
    k = int(p["n_samples"])
    xs = (np.arange(k) + 0.5) / k
    slack = int(p["search_frames"])
    aligned, control = [], []
    for x in xs:
        i = min(nf - 1, int(round(x * (nf - 1))))
        j = min(ns - 1, int(round(x * (ns - 1))))
        best = max(_corr(final_thumbs[i], source_thumbs[q])
                   for q in range(max(0, j - slack), min(ns, j + slack + 1)))
        aligned.append(best)
        # The same delivered frame against footage half the source away.
        m = (j + ns // 2) % ns
        control.append(max(_corr(final_thumbs[i], source_thumbs[q])
                           for q in range(max(0, m - slack), min(ns, m + slack + 1))))
    aligned, control = np.array(aligned), np.array(control)
    hit = float(np.mean(aligned >= p["match_correlation_min"]))
    wrong = float(np.mean(control >= p["match_correlation_min"]))
    out = {"n_samples": k, "aligned_match_fraction": round(hit, 4),
           "misaligned_match_fraction": round(wrong, 4),
           "aligned_median_correlation": round(float(np.median(aligned)), 4)}
    if hit < p["match_fraction_min"]:
        return {**out, "verified": False, "reason": "delivered frames do not follow the sources"}
    if wrong > hit - p["control_margin"]:
        # Every frame resembles every other: nothing ties a frame to a moment.
        return {**out, "verified": False,
                "reason": "content too static to tie delivered frames to source frames"}
    if abs(nf - ns) > max(1, p["frame_count_tolerance"] * ns):
        # Different frame counts: is this a stretch, or the sources at their
        # own speed with something appended? Pair by frame number and see.
        same = [_corr(final_thumbs[i], source_thumbs[i])
                for i in (min(nf - 1, int(round(x * (nf - 1)))) for x in xs) if i < ns]
        if len(same) >= max(4, k // 4):
            unstretched = float(np.mean(np.array(same) >= p["match_correlation_min"]))
            out["unstretched_match_fraction"] = round(unstretched, 4)
            if unstretched >= p["match_fraction_min"]:
                return {**out, "verified": False,
                        "reason": "delivered frames match the sources at their original speed"}
    return {**out, "verified": True, "reason": ""}
