"""Every reference report says what produced it, so it cannot be mistaken
for a report of the original detectors."""
from __future__ import annotations

IMPLEMENTATION = "vahbench-reference-reimplementation"
IMPLEMENTATION_VERSION = "2026.11.1"


def stamp(module: str, parameters: dict | None = None,
          not_measured: dict | None = None) -> dict:
    return {
        "name": IMPLEMENTATION,
        "version": IMPLEMENTATION_VERSION,
        "module": module,
        "original_detector": False,
        "calibration": "uncalibrated",
        "note": ("Clean-room reimplementation from the evaluator's report "
                 "contract; not the detector used for the published results."),
        "parameters": dict(parameters or {}),
        "not_measured": dict(not_measured or {}),
    }
