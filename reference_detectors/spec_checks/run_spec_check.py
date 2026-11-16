#!/usr/bin/env python
"""Reference specification check: container metadata and source timing."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]

from speccheck.report import check_video               # noqa: E402
from vahref.media import MediaError                    # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-duration", type=float)
    ap.add_argument("--target-resolution")
    ap.add_argument("--target-aspect")
    ap.add_argument("--source-clips", nargs="*", default=[])
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    try:
        rep = check_video(a.video, source_clips=a.source_clips,
                          target_duration=a.target_duration,
                          target_resolution=a.target_resolution,
                          target_aspect=a.target_aspect)
    except (MediaError, OSError) as exc:
        print(f"spec check failed: {exc}", file=sys.stderr)
        return 2
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    failed = [k for k, c in rep["checks"].items() if not c.get("ok")]
    if not a.quiet:
        print(json.dumps({"video": a.video, "failed": failed}))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
