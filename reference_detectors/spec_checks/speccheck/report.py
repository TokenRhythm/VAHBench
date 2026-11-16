"""One delivered file -> the `spec` report the evaluator reads."""
from __future__ import annotations

import re
from math import gcd

from speccheck.probe import probe
from speccheck import timing
from vahref import media
from vahref.provenance import stamp

PARAMS = {"duration_tolerance": 0.05, "aspect_tolerance": 0.01}


def _thumbs(path):
    from vahref import features
    return features.from_video(path).thumbs


def _aspect(w, h) -> str | None:
    if not w or not h:
        return None
    g = gcd(int(w), int(h)) or 1
    return f"{int(w) // g}:{int(h) // g}"


def _parse_resolution(text):
    m = re.fullmatch(r"\s*(\d+)\s*[x×*]\s*(\d+)\s*", str(text or "").lower())
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.fullmatch(r"\s*(\d+)p\s*", str(text or "").lower())
    return (None, int(m.group(1))) if m else None


def _parse_aspect(text):
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*[:：/x]\s*(\d+(?:\.\d+)?)\s*", str(text or ""))
    return (float(m.group(1)), float(m.group(2))) if m else None


def _check(ok, detail="", **extra):
    return {"ok": bool(ok), "detail": detail, **extra}


def _unknown(detail, **extra):
    return {"ok": True, "undetermined": True, "detail": detail, **extra}


def check_video(path, source_clips=None, target_duration=None,
                target_resolution=None, target_aspect=None) -> dict:
    meta = probe(path)
    source_clips = [str(s) for s in (source_clips or [])]
    checks: dict[str, dict] = {}

    fps = meta.get("fps")
    checks["frame_rate"] = (_check(True, fps=fps) if fps else _unknown("frame rate unavailable"))

    dur = meta.get("duration")
    if target_duration and dur:
        err = abs(float(dur) - float(target_duration)) / float(target_duration)
        checks["duration"] = _check(err <= PARAMS["duration_tolerance"],
                                    "" if err <= PARAMS["duration_tolerance"] else
                                    f"duration {dur:.2f}s, target {float(target_duration):.2f}s",
                                    duration=dur, target=float(target_duration))
    else:
        checks["duration"] = _unknown("no target given" if dur else "duration unavailable",
                                      duration=dur)

    w, h = meta.get("width"), meta.get("height")
    want = _parse_resolution(target_resolution)
    if not want:
        checks["resolution"] = _unknown("no target given")
    elif not w or not h:
        checks["resolution"] = _unknown("picture size unavailable")
    elif want[0] is None:
        ok = min(w, h) == want[1]
        checks["resolution"] = _check(ok, "" if ok else f"short side {min(w, h)}, required {want[1]}")
    else:
        ok = (w, h) == want
        checks["resolution"] = _check(ok, "" if ok else f"{w}x{h}, required {want[0]}x{want[1]}")

    wa = _parse_aspect(target_aspect)
    if not wa:
        checks["aspect"] = _unknown("no target given")
    elif not w or not h:
        checks["aspect"] = _unknown("picture size unavailable")
    else:
        got, need = w / h, wa[0] / wa[1]
        ok = abs(got - need) / need <= PARAMS["aspect_tolerance"]
        checks["aspect"] = _check(ok, "" if ok else f"aspect {_aspect(w, h)}, required {target_aspect}")

    checks["audio"] = _check(bool(meta.get("has_audio")),
                             "" if meta.get("has_audio") else "no audio stream")
    rot = meta.get("rotation_metadata") or 0
    checks["rotation_metadata"] = _check(rot == 0, "" if rot == 0 else
                                         f"display orientation relies on rotation metadata ({rot})")

    # Source timing. A source that cannot be probed makes the inventory
    # incomplete: the comparison is then unknown, never "no slowdown".
    source_meta, missing = [], []
    for s in source_clips:
        try:
            source_meta.append(probe(s))
        except (OSError, media.MediaError):
            missing.append(s)
    if missing:
        rt = timing._result(True, True, f"{len(missing)} source clip(s) could not be read")
    else:
        rt = timing.check_retiming(meta, source_meta)
        if rt.get("undetermined") and source_meta and rt.get("source_duration"):
            try:
                import numpy as np
                final_t = _thumbs(path)
                src_t = np.concatenate([_thumbs(s) for s in source_clips])
                rt = timing.check_retiming(
                    meta, source_meta,
                    correspondence=timing.verify_correspondence(final_t, src_t))
            except (OSError, media.MediaError, ValueError) as exc:
                rt["detail"] += f"; pictures could not be compared ({type(exc).__name__})"
    checks["retiming"] = rt

    return {"video": str(path), "meta": meta, "source_clips": source_clips,
            "source_meta": source_meta, "checks": checks,
            "implementation": stamp("spec", {**PARAMS, "retiming": timing.PARAMS})}
