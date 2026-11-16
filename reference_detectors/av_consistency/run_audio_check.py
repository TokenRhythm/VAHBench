#!/usr/bin/env python
"""Reference audio check: is there a signal, and is it clipped.

Measured: `audio_stream.present`, and in `cleanliness` the track level
(`rms`, `peak`, `signal_db`) and `clipped_runs`.

NOT measured, reported as null so the evaluator abstains instead of passing:

  * `dynamic_range_db` / `pause_fraction` -- the scored rule compares the
    level in speech against the level in pauses. Which stretches the original
    treated as pauses was decided by code that was not handed over; a level
    percentile chosen here would pass almost everything and say nothing.
  * speaker clusters and voice / face presence windows -- these used an
    audio-event model whose checkpoint is unidentified, and are diagnostics
    that never enter a score.

Long silence is measured by the evaluator itself (ffmpeg silencedetect), not
here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                     # noqa: E402

from vahref import media                               # noqa: E402
from vahref.provenance import stamp                    # noqa: E402

SR = 16000                 # rate of `extract_audio`, kept for callers
PARAMS = {
    # Samples pinned at the largest value the format can hold. A clean
    # full-scale tone touches it for a sample or two at each crest; a clipped
    # one stays there.
    "full_scale_code": 32767, "full_scale_min_samples": 8,
    # A flat top below full scale (clipped earlier, then turned down): the
    # same value exactly, at the track's peak, for this many samples. The
    # crest of a clean low tone repeats a value for only a few.
    "flat_top_min_samples": 16, "flat_top_tolerance_lsb": 0,
    "flat_top_peak_share": 0.98, "flat_top_min_peak": 0.25,
    "max_listed_runs": 500,
}
NOT_MEASURED = {
    "cleanliness.dynamic_range_db": "原始停顿判定未恢复，未重新实现",
    "cleanliness.pause_fraction": "原始停顿判定未恢复，未重新实现",
    "cleanliness.n_speaker_clusters": "诊断项，未重新实现",
    "presence": "声音事件/人脸对应为诊断项，原模型未确定，未重新实现",
}


def audio_stream_present(path) -> bool:
    return bool(media.probe(path)["has_audio"])


def extract_audio(path) -> np.ndarray:
    """Mono float32 at SR. Raises RuntimeError when the file has no audio."""
    try:
        pcm, _ = media.decode_pcm(path, sample_rate=SR, mono=True)
    except media.MediaError as exc:
        raise RuntimeError(str(exc)) from exc
    return (pcm[:, 0].astype(np.float32) / 32768.0)


def _runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    if not mask.any():
        return []
    edge = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edge == 1), np.flatnonzero(edge == -1) - 1
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a + 1 >= min_len]


def clipped_runs(pcm: np.ndarray, sr: int, p=PARAMS) -> list[dict]:
    """Runs of flat extrema per channel, on int16 samples (n, channels)."""
    out = []
    for ch in range(pcm.shape[1]):
        x = pcm[:, ch].astype(np.int32)
        mag = np.abs(x)
        peak = int(mag.max()) if len(mag) else 0
        if peak == 0:
            continue
        found = {}
        full = mag >= int(p["full_scale_code"])
        for a, b in _runs(full, int(p["full_scale_min_samples"])):
            found[(a, b)] = "full_scale"
        if peak / 32768.0 >= p["flat_top_min_peak"]:
            near = mag >= p["flat_top_peak_share"] * peak
            same = np.zeros(len(x), dtype=bool)
            same[1:] = np.abs(np.diff(x)) <= p["flat_top_tolerance_lsb"]
            flat = near & same
            for a, b in _runs(flat, int(p["flat_top_min_samples"]) - 1):
                a = max(0, a - 1)                       # include the run's first sample
                if not any(a <= fb and b >= fa for fa, fb in found):
                    found[(a, b)] = "flat_top"
        for (a, b), kind in sorted(found.items()):
            out.append({"channel": ch, "start_s": round(a / sr, 5), "end_s": round((b + 1) / sr, 5),
                        "n_samples": b - a + 1, "kind": kind})
    return out


def cleanliness(pcm: np.ndarray, sr: int, p=PARAMS) -> dict:
    if pcm.size == 0:
        return {"rms": 0.0, "peak": 0.0, "signal_db": -200.0, "clipped_runs": [],
                "n_clipped_runs": 0, "dynamic_range_db": None, "pause_fraction": None,
                "n_speaker_clusters": None, "n_speech_windows": None}
    # In blocks: a long stereo track converted to float in one piece would
    # need several times its own size in memory. Power is averaged over the
    # channels, not taken from their mix: two channels in opposite phase
    # cancel in a mix and would read as silence.
    total, top, block = 0.0, 0, 1 << 20
    for a in range(0, len(pcm), block):
        part = pcm[a:a + block].astype(np.float64) / 32768.0
        total += float(np.sum(part * part))
        top = max(top, float(np.abs(part).max()))
    rms = float(np.sqrt(total / pcm.size))
    peak = float(top)
    runs = clipped_runs(pcm, sr, p)
    return {"rms": rms, "peak": peak,
            "signal_db": float(20.0 * np.log10(rms)) if rms > 1e-10 else -200.0,
            "clipped_runs": runs[: int(p["max_listed_runs"])], "n_clipped_runs": len(runs),
            "dynamic_range_db": None, "pause_fraction": None,
            "n_speaker_clusters": None, "n_speech_windows": None}


def check(path) -> dict:
    meta = media.probe(path)
    rep = {"video": str(path), "audio_stream": {"present": bool(meta["has_audio"])},
           "not_measured": NOT_MEASURED,
           "implementation": stamp("av", PARAMS, NOT_MEASURED)}
    if not meta["has_audio"]:
        return rep
    pcm, sr = media.decode_pcm(path)
    rep["audio_stream"].update(sample_rate=sr, channels=int(pcm.shape[1]),
                               decoded_seconds=round(len(pcm) / sr, 4))
    rep["cleanliness"] = cleanliness(pcm, sr)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-visual", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    try:
        rep = check(a.video)
    except (media.MediaError, OSError) as exc:
        print(f"audio check failed: {exc}", file=sys.stderr)
        return 2
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
