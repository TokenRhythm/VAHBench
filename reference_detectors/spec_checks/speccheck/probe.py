"""Stream metadata for one file. Frames are never inferred from audio length."""
from __future__ import annotations

from os.path import exists

from vahref import media


def _ffprobe(path, *extra) -> dict:
    return media.ffprobe_json(path, *extra)


def probe(path) -> dict:
    if not exists(path):
        raise FileNotFoundError(str(path))
    data = _ffprobe(path)
    return media.meta_from_ffprobe(
        path, data,
        count=lambda: _ffprobe(path, "-select_streams", "V:0", "-count_packets"))
