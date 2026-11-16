"""Synthetic frame and audio arrays with known defects. No media tool needed."""
from __future__ import annotations

import sys

import numpy as np

from stage4_checks.dispatch import REFERENCE_ROOT, REFERENCE_SCRIPTS

FPS = 24.0
SIDE = 64


def use_reference(*modules: str) -> None:
    """Import the reference packages, not whatever `detectors/` may hold."""
    for path in [REFERENCE_ROOT, *[REFERENCE_SCRIPTS[m].parent for m in modules]]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def scene(n: int, seed: int, speed: float = 1.0) -> np.ndarray:
    """A blocky textured scene drifting sideways `speed` pixels per frame."""
    rng = np.random.default_rng(seed)
    width = SIDE + int(np.ceil(n * speed)) + 8
    coarse = rng.integers(40, 215, size=(8, width // 8 + 2))
    tex = np.kron(coarse, np.ones((8, 8)))[:SIDE]
    out = np.empty((n, SIDE, SIDE), dtype=np.uint8)
    for i in range(n):
        x = int(i * speed)
        out[i] = tex[:, x:x + SIDE]
    return out


def smooth_drift(n: int, seed: int, speed: float = 0.1) -> np.ndarray:
    """Long-wavelength pattern drifting slowly: a steady camera move."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:SIDE, 0:SIDE].astype(np.float64)
    phases = rng.uniform(0, 2 * np.pi, size=3)
    out = np.empty((n, SIDE, SIDE), dtype=np.uint8)
    for i in range(n):
        s = i * speed
        v = (60 * np.sin(2 * np.pi * (x + s) / 64 + phases[0])
             + 40 * np.sin(2 * np.pi * (y + 0.5 * s) / 80 + phases[1])
             + 20 * np.sin(2 * np.pi * (x + y + s) / 96 + phases[2]))
        out[i] = np.clip(128 + v, 0, 255)
    return out


def still(n: int, seed: int, noise: float = 0.0) -> np.ndarray:
    frame = scene(1, seed)[0].astype(np.float64)
    rng = np.random.default_rng(seed + 1)
    out = np.repeat(frame[None], n, axis=0)
    if noise:
        out = out + rng.normal(0, noise, size=out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def features(frames: np.ndarray):
    use_reference()
    from vahref import features as ft
    return ft.from_chunks([frames[i:i + 100] for i in range(0, len(frames), 100)])
