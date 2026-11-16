#!/usr/bin/env python
"""Reference visual pass: the temporal repeat scan only.

Cross-shot face identity (ArcFace in the paper) is NOT implemented here: the
original face model pack and its settings were not recoverable, and choosing
one would not reproduce the benchmark. The report therefore carries no
`character_identity` block and the evaluator abstains on identity.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]

from vcheck.repeats import scan_video                  # noqa: E402
from vahref.provenance import stamp                    # noqa: E402

NOT_MEASURED = {
    "character_identity": "人脸身份检测未重新实现：原始人脸模型包与设置未恢复",
    "scene_reference_check": "参考图相似度仅为诊断项，未重新实现",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--shots-json")
    ap.add_argument("--no-visual", action="store_true")
    ap.add_argument("--reference-images", nargs="*", default=[])
    ap.add_argument("--scene-references", nargs="*", default=[])
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    shots = []
    if a.shots_json:
        try:
            shots = json.loads(Path(a.shots_json).read_text(encoding="utf-8")).get("shots") or []
        except (OSError, ValueError):
            shots = []
    repeats = scan_video(a.video)
    rep = {"video": a.video, "duration": repeats.get("duration"), "shots": shots,
           "degeneracy": {"repeats": repeats, "near_duplicate_shots": []},
           "not_measured": NOT_MEASURED,
           "implementation": stamp("visual", {}, NOT_MEASURED)}
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    if not a.quiet:
        print(json.dumps({"video": a.video, "duplicate_seconds": repeats["duplicate_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
