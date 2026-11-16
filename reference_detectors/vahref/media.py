"""ffprobe metadata and ffmpeg decoding. No OpenCV, no model weights.

Frames are decoded once, in display order, scaled to a small grey image.
`-fps_mode passthrough` keeps exactly the frames the stream holds: the
default constant-rate output would duplicate or drop frames to fit the
container rate, which is the very thing the timing checks must see.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

FRAME_SIDE = 64          # analysis frame, grey, FRAME_SIDE x FRAME_SIDE
THUMB_SIDE = 32          # stored per frame for correspondence and repeats


class MediaError(RuntimeError):
    """The file could not be probed or decoded. Never a measurement."""


def tool(name: str) -> str | None:
    """ffmpeg / ffprobe from PATH; a sibling of the other when only one is found."""
    exe = shutil.which(name)
    if exe:
        return exe
    other = shutil.which("ffprobe" if name == "ffmpeg" else "ffmpeg")
    if other:
        cand = Path(other).with_name(name + Path(other).suffix)
        if cand.is_file():
            return str(cand)
    return None


def source(path) -> str:
    """The input as FFmpeg must be given it. A bare name such as
    `ep1 12:30.mp4` would be read as a protocol, and one starting with `-`
    as an option; the explicit file protocol avoids both."""
    return "file:" + str(path)


def ffprobe_json(path, *extra: str, timeout: float = 120.0) -> dict:
    exe = tool("ffprobe")
    if exe is None:
        raise MediaError("ffprobe not on PATH")
    cmd = [exe, "-v", "error", "-print_format", "json", "-show_format",
           "-show_streams", *extra, source(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, encoding="utf-8",
                             errors="replace", timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MediaError(f"ffprobe failed: {type(exc).__name__}") from exc
    if out.returncode != 0:
        raise MediaError("ffprobe could not read the file")
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError as exc:
        raise MediaError("ffprobe returned unparseable output") from exc
    if not isinstance(data, dict):
        raise MediaError("ffprobe returned unexpected output")
    return data


def _num(value) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if x == x and x not in (float("inf"), float("-inf")) else None


def _rate(text) -> float | None:
    if not text or "/" not in str(text):
        return _num(text)
    a, _, b = str(text).partition("/")
    n, d = _num(a), _num(b)
    return (n / d) if n and d else None


def _rotation(stream: dict) -> int:
    for side in stream.get("side_data_list") or []:
        if isinstance(side, dict) and side.get("rotation") is not None:
            r = _num(side.get("rotation"))
            if r is not None:
                return int(round(r)) % 360
    r = _num((stream.get("tags") or {}).get("rotate"))
    return int(round(r)) % 360 if r is not None else 0


def meta_from_ffprobe(path, data: dict, count=None) -> dict:
    """Stream metadata as the spec report's `meta` block.

    `n_frames` and `video_duration` describe the picture stream only. They are
    never derived from the container duration: an audio track that outlasts the
    picture would otherwise invent frames that do not exist. `count`, when
    given, is called once to count video packets if the container records no
    frame count.
    """
    streams = data.get("streams") or []
    vid = next((s for s in streams if s.get("codec_type") == "video"
                and not (s.get("disposition") or {}).get("attached_pic")), None)
    aud = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = data.get("format") or {}
    meta = {"path": str(path), "fps": None, "n_frames": 0, "duration": _num(fmt.get("duration")),
            "video_duration": None, "width": None, "height": None,
            "coded_width": None, "coded_height": None,
            "has_audio": aud is not None,
            "audio_duration": _num(aud.get("duration")) if aud else None,
            "audio_sample_rate": int(_num(aud.get("sample_rate")) or 0) if aud else None,
            "audio_channels": int(_num(aud.get("channels")) or 0) if aud else None,
            "rotation_metadata": 0, "has_video": vid is not None,
            "frame_count_source": None}
    if vid is None:
        return meta
    fps = _rate(vid.get("avg_frame_rate")) or _rate(vid.get("r_frame_rate"))
    meta["fps"] = fps
    n = _num(vid.get("nb_frames"))
    source = "container" if n else None
    if not n and count is not None:
        try:
            counted = count() or {}
        except MediaError:
            counted = {}
        cs = next((s for s in (counted.get("streams") or [])
                   if s.get("codec_type") == "video"), None) or {}
        n = _num(cs.get("nb_read_packets")) or _num(cs.get("nb_read_frames"))
        source = "counted_packets" if n else None
    meta["n_frames"] = int(n) if n else 0
    meta["frame_count_source"] = source
    vd = _num(vid.get("duration"))
    if vd is None and meta["n_frames"] and fps:
        vd = meta["n_frames"] / fps
    meta["video_duration"] = vd
    rot = _rotation(vid)
    w, h = vid.get("width"), vid.get("height")
    meta["coded_width"], meta["coded_height"] = w, h
    if rot in (90, 270) and w and h:
        w, h = h, w                      # what a player shows
    meta["width"], meta["height"] = w, h
    meta["rotation_metadata"] = rot
    return meta


def probe(path) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(str(path))
    data = ffprobe_json(path)
    return meta_from_ffprobe(
        path, data,
        count=lambda: ffprobe_json(path, "-select_streams", "V:0", "-count_packets"))


def frame_clock(meta: dict, n_decoded: int) -> float:
    """Frames per second for turning a decoded frame index into seconds.

    The picture stream's own duration over the frames actually decoded is the
    better clock than a container rate the muxer rounded -- but only when the
    decode is complete. If the decoded count disagrees with the count the
    container states, the nominal rate is used and timestamps are not
    stretched to fit a partial or padded decode.
    """
    nominal = float(meta.get("fps") or 0) or 24.0
    duration, stated = meta.get("video_duration"), meta.get("n_frames") or 0
    if not duration or not n_decoded:
        return nominal
    if stated and abs(n_decoded - stated) > max(2, 0.02 * stated):
        return nominal
    return n_decoded / float(duration)


def _decode_cmd(exe: str, path, side: int, passthrough: str) -> list[str]:
    flag = (["-fps_mode", "passthrough"] if passthrough == "fps_mode"
            else ["-vsync", "passthrough"])
    # V, not v: the first real video stream, never an attached cover image.
    return [exe, "-v", "error", "-nostdin", "-i", source(path), "-map", "0:V:0",
            "-an", "-sn", "-dn", "-vf", f"scale={side}:{side}:flags=area,format=gray",
            *flag, "-f", "rawvideo", "-pix_fmt", "gray", "-"]


def iter_gray_frames(path, side: int = FRAME_SIDE, chunk: int = 512,
                     timeout: float = 3600.0):
    """Yield uint8 arrays (k, side, side) in decode order."""
    exe = tool("ffmpeg")
    if exe is None:
        raise MediaError("ffmpeg not on PATH")
    size = side * side
    last_error = "ffmpeg produced no frames"
    for mode in ("fps_mode", "vsync"):
        # Diagnostics go to a file, not a pipe: a damaged stream can log more
        # than a pipe holds while frames are still being read, and nothing
        # would drain it.
        with tempfile.TemporaryFile() as log:
            try:
                proc = subprocess.Popen(_decode_cmd(exe, path, side, mode),
                                        stdout=subprocess.PIPE, stderr=log)
            except OSError as exc:
                raise MediaError(f"ffmpeg failed: {type(exc).__name__}") from exc
            got, rc = 0, None
            try:
                while True:
                    buf = proc.stdout.read(size * chunk)
                    if not buf:
                        break
                    k = len(buf) // size
                    if k:
                        got += k
                        yield np.frombuffer(buf[:k * size], dtype=np.uint8).reshape(k, side, side)
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                raise MediaError("ffmpeg did not finish") from exc
            finally:
                if proc.poll() is None:          # consumer stopped early, or an error
                    proc.kill()
                    proc.wait()
                proc.stdout.close()
            log.seek(0)
            err = log.read(65536).decode("utf-8", "replace")
        if rc == 0 and got:
            return
        if got:
            # Frames followed by a failure are a partial decode. Measuring
            # them would describe a shorter film than the one delivered.
            raise MediaError("ffmpeg stopped with an error after decoding part of the picture")
        last_error = "ffmpeg could not decode the picture stream"
        # Older builds do not know -fps_mode; newer ones dropped -vsync.
        if "fps_mode" in err or "vsync" in err or "Unrecognized option" in err:
            continue
        break
    raise MediaError(last_error)


def decode_pcm(path, sample_rate: int | None = None, mono: bool = False,
               timeout: float = 3600.0):
    """First audio stream as int16, shape (n, channels). Raises when absent."""
    exe = tool("ffmpeg")
    if exe is None:
        raise MediaError("ffmpeg not on PATH")
    meta = probe(path)
    if not meta["has_audio"]:
        raise MediaError("no audio stream")
    sr = int(sample_rate or meta.get("audio_sample_rate") or 0)
    ch = 1 if mono else int(meta.get("audio_channels") or 0)
    if sr <= 0 or ch <= 0:
        raise MediaError("audio stream has no usable sample rate or channel count")
    cmd = [exe, "-v", "error", "-nostdin", "-i", source(path), "-map", "0:a:0", "-vn",
           "-sn", "-dn", "-ac", str(ch), "-ar", str(sr), "-f", "s16le",
           "-acodec", "pcm_s16le", "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MediaError(f"ffmpeg failed: {type(exc).__name__}") from exc
    if out.returncode != 0:
        raise MediaError("ffmpeg could not decode the audio stream")
    data = np.frombuffer(out.stdout, dtype="<i2")
    data = data[: (len(data) // ch) * ch].reshape(-1, ch)
    return data, sr
