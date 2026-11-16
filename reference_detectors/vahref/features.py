"""One pass over the decoded frames -> the per-frame numbers every check reads.

    mean, std, p99   luma statistics of the whole analysis frame
    mad[i]           mean absolute difference between frame i and i-1
    mad_top[i]       the same over the top three quarters of the frame, so a
                     subtitle changing in the lower band is not read as a cut
    mad_block[i]     the largest such difference over an 8x8 grid of blocks: a
                     small moving region in an otherwise still picture changes
                     the whole-frame mean by almost nothing, and must not be
                     taken for a held frame
    thumbs           a THUMB_SIDE x THUMB_SIDE grey thumbnail per frame

Everything downstream works on these arrays, which is what lets the analysis
be tested on synthetic arrays without any media tool installed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vahref import media

SUBTITLE_BAND = 0.25      # lower share of the frame ignored by cut detection
BLOCK_GRID = 8            # blocks per side for the local-change measure


@dataclass
class FrameFeatures:
    n: int
    mean: np.ndarray
    std: np.ndarray
    p99: np.ndarray
    mad: np.ndarray
    mad_top: np.ndarray
    thumbs: np.ndarray        # (n, THUMB_SIDE, THUMB_SIDE) uint8
    mad_block: np.ndarray | None = None

    def thumbs16(self) -> np.ndarray:
        t = self.thumbs.astype(np.float32)
        s = t.shape[1] // 16
        return t.reshape(self.n, 16, s, 16, s).mean(axis=(2, 4))


def _thumb(frames: np.ndarray, side: int) -> np.ndarray:
    k, h, w = frames.shape
    f = h // side
    return (frames.reshape(k, side, f, side, f).astype(np.float32).mean(axis=(2, 4))
            + 0.5).astype(np.uint8)


def from_chunks(chunks) -> FrameFeatures:
    """Accumulate features from an iterable of uint8 arrays (k, H, W)."""
    means, stds, p99s, mads, tops, thumbs, blocks = [], [], [], [], [], [], []
    prev = None
    for block in chunks:
        block = np.asarray(block, dtype=np.uint8)
        if block.ndim != 3 or not len(block):
            continue
        f = block.astype(np.float32)
        means.append(f.mean(axis=(1, 2)))
        stds.append(f.std(axis=(1, 2)))
        p99s.append(np.percentile(f.reshape(len(f), -1), 99, axis=1))
        lead = f[:1] if prev is None else prev[None]
        d = np.abs(f - np.concatenate([lead, f[:-1]]))
        rows = max(1, int(round(f.shape[1] * (1.0 - SUBTITLE_BAND))))
        mads.append(d.mean(axis=(1, 2)))
        tops.append(d[:, :rows].mean(axis=(1, 2)))
        g = BLOCK_GRID
        bh, bw = d.shape[1] // g, d.shape[2] // g
        blocks.append(d[:, :bh * g, :bw * g].reshape(len(d), g, bh, g, bw)
                      .mean(axis=(2, 4)).max(axis=(1, 2)))
        thumbs.append(_thumb(block, media.THUMB_SIDE))
        prev = f[-1]
    if not means:
        z = np.zeros(0, dtype=np.float32)
        return FrameFeatures(0, z, z, z, z, z,
                             np.zeros((0, media.THUMB_SIDE, media.THUMB_SIDE), np.uint8), z)
    cat = np.concatenate
    return FrameFeatures(int(sum(len(m) for m in means)), cat(means), cat(stds),
                         cat(p99s), cat(mads), cat(tops), cat(thumbs), cat(blocks))


def from_video(path) -> FrameFeatures:
    return from_chunks(media.iter_gray_frames(path))


def runs(mask) -> list[tuple[int, int]]:
    """Maximal runs of True as (first, last) index pairs."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    edge = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edge == 1), np.flatnonzero(edge == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def robust_z(x: np.ndarray, half: int, floor: float = 0.5) -> np.ndarray:
    """(x - local median) / (1.4826 * local MAD + floor) over +-half samples.

    The centre sample is excluded from its own neighbourhood, so one isolated
    spike cannot raise the level it is compared against.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return np.zeros(0)
    half = max(1, int(half))
    pad = np.pad(x, half, mode="reflect") if n > half else np.pad(x, half, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1)
    keep = np.ones(2 * half + 1, dtype=bool)
    keep[half] = False
    nb = win[:, keep]
    med = np.median(nb, axis=1)
    mad = np.median(np.abs(nb - med[:, None]), axis=1)
    return (x - med) / (1.4826 * mad + floor)
