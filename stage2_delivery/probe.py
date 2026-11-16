"""ffprobe wrapper. No cv2, no torch -- this runs in the orchestrator process.

Deliberately thin: everything downstream reads duration / geometry / audio
presence, and nothing else. Real frame rate and real duration for the *scoring*
path come from av_consistency/timeline.py, not from here -- container metadata
out of an agent harness lies often enough that spec_checks exists because of it.
What this module is for is candidate discovery, where a lying container is
still a usable sort key.
"""
from __future__ import annotations

import json
import os
import shutil
import stat as statmod
import subprocess
from dataclasses import dataclass
from pathlib import Path

FFPROBE = shutil.which("ffprobe")


def _stat(p: Path) -> os.stat_result | None:
    r"""stat() that survives a long path. None when the file really is absent.

    Python's stat()/is_file() answer "no such file" for paths past roughly 240
    characters unless they carry the \\?\ prefix. Twenty files in this dataset
    sit at 243-249 characters absolute, among them videoclaw's episode files --
    so `locate_delivery` skipped them and three runs that had in fact delivered
    were reported as having produced nothing. ffprobe itself handles the long
    path fine; only Python's own filesystem calls need the prefix, so it is
    applied as a fallback and cache keys for normal paths stay unchanged.
    """
    try:
        return p.stat()
    except OSError:
        pass
    if os.name != "nt":
        return None
    try:
        return Path("\\\\?\\" + os.path.abspath(str(p))).stat()
    except OSError:
        return None


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    ok: bool
    duration: float | None = None
    width: int | None = None
    height: int | None = None
    n_frames: int | None = None
    avg_fps: float | None = None
    has_audio: bool = False
    audio_duration: float | None = None
    # The picture stream's own length. `duration` is the container's, which
    # an audio track outlasting the last frame makes longer than the picture.
    video_duration: float | None = None
    size_bytes: int = 0
    error: str = ""

    @property
    def aspect(self) -> str | None:
        if not self.width or not self.height:
            return None
        from math import gcd
        g = gcd(self.width, self.height)
        return f"{self.width // g}:{self.height // g}"


def _f(v) -> float | None:
    try:
        x = float(v)
        return x if x == x else None      # drop NaN
    except (TypeError, ValueError):
        return None


def probe(path: str | Path, timeout: float = 30.0) -> MediaInfo:
    p = Path(path)
    st = _stat(p)
    if st is None or not statmod.S_ISREG(st.st_mode):
        return MediaInfo(p, False, error="not a file")
    if FFPROBE is None:
        return MediaInfo(p, False, error="ffprobe not on PATH")

    cmd = [FFPROBE, "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", str(p)]
    try:
        out = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace",
                             timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return MediaInfo(p, False, error="ffprobe timeout")
    if out.returncode != 0:
        return MediaInfo(p, False, error=(out.stderr or "").strip()[:200])
    try:
        d = json.loads(out.stdout)
    except json.JSONDecodeError as e:
        return MediaInfo(p, False, error=f"unparseable ffprobe json: {e}")

    streams = d.get("streams") or []
    vid = next((s for s in streams if s.get("codec_type") == "video"), None)
    aud = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = d.get("format") or {}

    duration = _f(fmt.get("duration")) or (_f(vid.get("duration")) if vid else None)
    n_frames = None
    avg_fps = None
    if vid is not None:
        try:
            n_frames = int(vid.get("nb_frames"))
        except (TypeError, ValueError):
            n_frames = None
        rate = vid.get("avg_frame_rate") or vid.get("r_frame_rate") or ""
        if "/" in rate:
            num, _, den = rate.partition("/")
            n, dn = _f(num), _f(den)
            if n and dn:
                avg_fps = n / dn

    # Never derived from the container duration: that would hide exactly the
    # overhang this field exists to expose.
    video_duration = _f(vid.get("duration")) if vid else None
    if video_duration is None and n_frames and avg_fps:
        video_duration = n_frames / avg_fps

    return MediaInfo(
        path=p,
        ok=vid is not None,
        duration=duration,
        video_duration=video_duration,
        width=int(vid["width"]) if vid and vid.get("width") else None,
        height=int(vid["height"]) if vid and vid.get("height") else None,
        n_frames=n_frames,
        avg_fps=avg_fps,
        has_audio=aud is not None,
        audio_duration=_f(aud.get("duration")) if aud else None,
        size_bytes=st.st_size,
        error="" if vid is not None else "no video stream",
    )


class ProbeCache:
    """Memoise on (path, nanosecond mtime, size), including quick replacements."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._d: dict[str, dict] = {}
        if self.path and self.path.is_file():
            try:
                self._d = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._d = {}

    @staticmethod
    def _key(p: Path) -> str:
        st = _stat(p)
        if st is None:
            return str(p)
        # A same-size edit can occur within one second. Namespace the new keys
        # so persisted second-resolution entries cannot be mistaken for these.
        # v3: entries carry video_duration; a v2 entry would read as "unknown".
        return f"probe-v3|{p.resolve()}|{st.st_mtime_ns}|{st.st_size}"

    def get(self, path: str | Path) -> MediaInfo:
        p = Path(path)
        k = self._key(p)
        hit = self._d.get(k)
        if hit is not None:
            return MediaInfo(path=p, **hit)
        info = probe(p)
        fields = {x: y for x, y in info.__dict__.items() if x != "path"}
        self._d[k] = fields
        return info

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._d), encoding="utf-8")
