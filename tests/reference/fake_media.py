"""Stand-in ffprobe / ffmpeg executables for tests that must cross a process
boundary on machines without FFmpeg.

A "video" is an .npz archive (whatever its file name) holding

    frames  uint8 (n, 64, 64)      fps  float
    audio   int16 (m, channels)    sr   int        (both optional)
    container_extra  float seconds the container runs past the picture

The stand-ins answer exactly the invocations `reference_detectors/vahref/media.py`
and the evaluator's own probe and silence scan make. They prove the plumbing
(arguments, pipes, report files, cache, scoring), not FFmpeg's behaviour:
the real-tool tests in test_reference_media.py do that where FFmpeg exists.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import numpy as np

_COMMON = r'''
import json, sys
import numpy as np

def load(path):
    path = path[5:] if path.startswith("file:") else path
    with open(path, "rb") as fh:
        z = np.load(fh, allow_pickle=False)
        return {k: z[k] for k in z.files}

def arg_after(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default
'''

_FFPROBE = _COMMON + r'''
try:
    v = load(sys.argv[-1])
except Exception:
    sys.exit(1)
n, fps = len(v["frames"]), float(v["fps"])
picture = n / fps
extra = float(v["container_extra"]) if "container_extra" in v else 0.0
streams = [{"codec_type": "video", "nb_frames": str(n), "avg_frame_rate": f"{int(round(fps))}/1",
            "duration": f"{picture:.6f}", "width": 720, "height": 1280}]
if "audio" in v:
    a, sr = v["audio"], int(v["sr"])
    streams.append({"codec_type": "audio", "duration": f"{len(a) / sr:.6f}",
                    "sample_rate": str(sr), "channels": int(a.shape[1])})
print(json.dumps({"format": {"duration": f"{picture + extra:.6f}"}, "streams": streams}))
'''

_FFMPEG = _COMMON + r'''
import os
if "-version" in sys.argv:
    print("ffmpeg version fake-for-tests"); sys.exit(0)
try:
    v = load(arg_after("-i"))
except Exception:
    sys.stderr.write("Invalid data found when processing input\n"); sys.exit(1)
out = sys.stdout.buffer
if "silencedetect" in " ".join(sys.argv):
    sys.exit(0)                                   # no silence lines: none found
if "rawvideo" in sys.argv:
    if "-fps_mode" in sys.argv and os.environ.get("FAKE_FFMPEG_OLD"):
        sys.stderr.write("Unrecognized option 'fps_mode'.\n"); sys.exit(1)
    if "-fps_mode" not in sys.argv and "-vsync" not in sys.argv:
        sys.stderr.write("frames would be duplicated or dropped\n"); sys.exit(1)
    frames = v["frames"].astype(np.uint8)
    if os.environ.get("FAKE_FFMPEG_NOISY"):
        # A damaged stream: far more diagnostics than a pipe buffer holds,
        # written while frames are still being produced.
        for i in range(len(frames)):
            sys.stderr.write("error while decoding MB 12 34, bytestream -5\n" * 80)
            sys.stderr.flush()
            out.write(frames[i].tobytes()); out.flush()
        sys.exit(0)
    if os.environ.get("FAKE_FFMPEG_PARTIAL"):
        out.write(frames[: len(frames) // 2].tobytes()); out.flush()
        sys.stderr.write("Invalid data found when processing input\n"); sys.exit(1)
    out.write(frames.tobytes()); sys.exit(0)
if "s16le" in sys.argv:
    if "audio" not in v:
        sys.stderr.write("Stream map '0:a:0' matches no streams.\n"); sys.exit(1)
    a, sr = v["audio"].astype(np.float64), int(v["sr"])
    want_sr, want_ch = int(arg_after("-ar", sr)), int(arg_after("-ac", a.shape[1]))
    if want_ch == 1 and a.shape[1] > 1:
        a = a.mean(axis=1, keepdims=True)
    if want_sr != sr:
        idx = np.minimum(len(a) - 1, (np.arange(int(len(a) * want_sr / sr)) * sr / want_sr).astype(int))
        a = a[idx]
    out.write(np.round(a).astype("<i2").tobytes()); sys.exit(0)
sys.stderr.write("unsupported fake ffmpeg invocation\n"); sys.exit(1)
'''


def install(directory: Path) -> Path:
    """Write executable `ffprobe` and `ffmpeg` stand-ins; return their folder."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, body in (("ffprobe", _FFPROBE), ("ffmpeg", _FFMPEG)):
        path = directory / name
        path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return directory


def write_video(path: Path, frames: np.ndarray, fps: float = 24.0, audio=None, sr: int = 48000,
                container_extra: float = 0.0) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"frames": np.asarray(frames, dtype=np.uint8), "fps": np.float64(fps),
            "container_extra": np.float64(container_extra)}
    if audio is not None:
        data["audio"], data["sr"] = np.asarray(audio, dtype=np.int16), np.int64(sr)
    with open(path, "wb") as fh:                       # keep the caller's file name
        np.savez(fh, **data)
    return path


def path_with(directory: Path) -> str:
    return str(directory) + os.pathsep + os.environ.get("PATH", "")
