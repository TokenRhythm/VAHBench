"""Every measurement key the pipeline emits, and what the tree does with it.

Four disjoint groups:

    SCORED        read by a leaf of the tree (video/quality/*, trajectory/*)
    COMPLETENESS  read by the completeness ledger (video/completeness.py) --
                  seconds, not scores; they never enter Q or P
    DEFERRED      unbuilt, or excluded for every entrant because its proxy
                  lacks the correspondence needed by the rubric
    DIAGNOSTIC    optional production methods, reported without scoring

`ALL_CHECKS` is built from this file, so the registry is the single
place a key has to be declared for `flows`, `trajectory_schema.CONSUMERS` and
the tests to know it exists.
"""
from __future__ import annotations

from shared.types import Check
from stage5_scoring.video.quality import QUALITY
from stage5_scoring.process import TRAJECTORY
from stage5_scoring.capabilities import UNSUPPORTED_QUALITY_SOURCES
from stage5_scoring.capabilities import METHOD_DIAGNOSTIC_SOURCES
from stage5_scoring.node import source_mapping

# ---- scored: every source of every leaf ----------------------------------
_scored: dict[str, Check] = {}
for _cat in (*QUALITY, *TRAJECTORY):
    for _leaf in _cat.leaves:
        for _key in _leaf.sources:
            _scored.setdefault(_key, Check(_key, implemented=_leaf.implemented,
                                           deferred="" if _leaf.implemented else "extractor"))

# ---- completeness inputs: read as seconds by the ledger -------------------
# Everything that answers "how much of this file is real, newly generated
# footage" belongs to C and is charged there exactly once. The delivery
# defects below used to be scored a second time as an occurrence rate in
# 1.2.3.a; that double charge is what moving them here removes.
COMPLETENESS_INPUTS: dict[str, Check] = {k: Check(k) for k in (
    "spec.duration",                     # delivered vs target seconds
    "spec.retiming",                     # source vs final seconds
    "provenance.duration_provenance",    # delivered vs generated seconds
    "o2.episode_count",                  # found vs required episodes
    "artifact.tail_freeze",              # frozen tail seconds
    "artifact.frozen_runs",              # held-frame seconds and their count
    "artifact.black_frames",             # black seconds and their count
    "artifact.solid_frames",             # flat-colour seconds and their count
    "artifact.join_artifacts",           # broken seams: frames and rate
    "visual.near_duplicate_shots",       # repeated shot seconds
)}

# ---- diagnostics: optional production methods -----------------------------
DIAGNOSTIC: dict[str, Check] = {k: Check(k) for k in METHOD_DIAGNOSTIC_SOURCES}

# ---- deferred: declared, not emitted --------------------------------------
DEFERRED_CHECKS: dict[str, Check] = {
    "vlm.artifact_quality": Check("vlm.artifact_quality", implemented=False, deferred="compute"),
    "vlm.narrative_coherence": Check("vlm.narrative_coherence", implemented=False, deferred="compute"),
    "vlm.narration_visual_alignment": Check("vlm.narration_visual_alignment",
                                            implemented=False, deferred="compute"),
    "av.lip_sync": Check("av.lip_sync", implemented=False, deferred="module"),
}
# The two unbuilt trajectory leaves declare their keys through the tree.
for _k, _c in list(_scored.items()):
    if _k in METHOD_DIAGNOSTIC_SOURCES:
        del _scored[_k]
    elif not _c.implemented or _k in UNSUPPORTED_QUALITY_SOURCES:
        DEFERRED_CHECKS[_k] = (Check(_k, implemented=False, deferred="evidence")
                               if _k in UNSUPPORTED_QUALITY_SOURCES else _c)
        del _scored[_k]

SCORED: dict[str, Check] = dict(_scored)
SOURCE_MAPPINGS = {}
for _cat in (*QUALITY, *TRAJECTORY):
    for _leaf in _cat.leaves:
        for _key in _leaf.sources:
            if _key in SCORED or _key in METHOD_DIAGNOSTIC_SOURCES:
                _mapping = source_mapping(_key, _leaf.mapping)
                assert _key not in SOURCE_MAPPINGS or SOURCE_MAPPINGS[_key] == _mapping
                SOURCE_MAPPINGS[_key] = _mapping

ALL_CHECKS: dict[str, Check] = {**SCORED, **COMPLETENESS_INPUTS, **DIAGNOSTIC, **DEFERRED_CHECKS}

IMPLEMENTED = frozenset(SCORED)
PROVISIONAL = frozenset(k for k, c in DIAGNOSTIC.items() if c.provisional)
DEFERRED = frozenset(DEFERRED_CHECKS)
PLANNED = frozenset(k for k, c in ALL_CHECKS.items()
                    if not c.implemented and not c.deferred)

# Adapters exist, but their current detector output cannot answer the scored
# question. Keep these gaps visible at rubric level as well as per run.
EVIDENCE_LIMITATIONS = dict(UNSUPPORTED_QUALITY_SOURCES)


def completeness_of_rubric() -> dict:
    """How much of the declared tree exists. Distinct from a run's coverage."""
    n = len(ALL_CHECKS)
    # Capability gaps are excluded from run coverage, not counted as failures
    # of individual entrants. They still count against rubric completeness.
    unbuilt = sum(1 for cat in (*QUALITY, *TRAJECTORY)
                  for leaf in cat.leaves if not leaf.implemented)
    return {
        "n_checks": n,
        "n_scored": len(SCORED),
        "n_unimplemented_leaves": unbuilt,
        "n_sources_with_evidence_limitations": len(EVIDENCE_LIMITATIONS),
        "evidence_limitations": dict(EVIDENCE_LIMITATIONS),
        "n_completeness_inputs": len(COMPLETENESS_INPUTS),
        "n_diagnostic": len(DIAGNOSTIC),
        "n_deferred": len(DEFERRED),
        "fraction": (len(SCORED) + len(COMPLETENESS_INPUTS)) / n if n else 0.0,
        "planned": sorted(PLANNED),
        "deferred": {k: ALL_CHECKS[k].deferred for k in sorted(DEFERRED)},
        "diagnostic": sorted(DIAGNOSTIC),
    }
