#!/usr/bin/env python
"""Reference artifact check: black / solid / frozen frames, splices, joins."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]

import artifacts                                       # noqa: E402
from vahref.media import MediaError                    # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    try:
        rep = artifacts.analyse(a.video)
    except (MediaError, OSError) as exc:
        print(f"artifact check failed: {exc}", file=sys.stderr)
        return 2
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    if not a.quiet:
        print(json.dumps({"video": a.video, "clean": rep["clean"],
                          "n_splices": rep["splices"]["n_splices"]}))
    return 0 if rep["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
