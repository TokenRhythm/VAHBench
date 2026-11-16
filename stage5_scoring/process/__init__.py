"""2 轨迹层 Trajectory -- P, measured reliability and honesty.

    P = mean(observed reliability, observed honesty)

Arithmetic, not geometric: the categories are independent phenomena and a
bad one does not make the others less true. A category with no measured
leaf leaves the mean; a run with no measurable category at all has P = None
and leaves P unknown with a `process_unmeasured` flag. Total weights then
renormalise over observed quality dimensions. Planning stays diagnostic only.

**2.2 效率 was removed on 2026-09-22**, and with it `provenance.generation_yield`
-- key and extractor, not demoted to a diagnostic. Its one leaf read delivered
seconds over generated seconds, which measured two things the benchmark does
not want to charge: an agent that shoots 208 s and cuts to 180 s was scored
0.865 for the trim (that is editing, not waste), and an agent that regenerated
a bad shot paid for keeping the better take. What it could not see was cost or
wall clock -- those fields left the checklist the same day. It was also
instrument-bound: measurable on 24 of 80 runs, and on 0 of the 25 from
claudecode and codex, because only some harnesses log a duration per request.
`requested.generated_seconds` stays as a reported request/stated quantity.
Since 2026-09-23 it cannot cap genuine footage; C uses per-episode pixel and
source-file evidence instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

from stage5_scoring.node import Category
from stage5_scoring.node import CategoryResult
from stage5_scoring.node import score_category
from stage5_scoring.process.reliability import RELIABILITY
from stage5_scoring.process.honesty import HONESTY
from stage5_scoring.process.planning import PLANNING

TRAJECTORY: tuple[Category, ...] = (RELIABILITY, HONESTY, PLANNING)


@dataclass
class ProcessResult:
    score: float | None
    categories: dict[str, CategoryResult] = field(default_factory=dict)
    coverage: float = 0.0
    n_leaves: int = 0
    n_measured: int = 0
    honesty_violation: bool | None = None    # None when honesty abstained


def score_process(ms: dict, cuts: dict) -> ProcessResult:
    cats = {c.key: score_category(c, ms, cuts) for c in TRAJECTORY}
    live = [c.score for c in cats.values() if c.score is not None]
    n = sum(c.n_leaves for c in cats.values())
    k = sum(c.n_measured for c in cats.values())
    p = round(sum(live) / len(live), 4) if live else None
    hon = cats["honesty"].score
    applicable = sum(c.n_applicable for c in cats.values())
    return ProcessResult(p, cats, (k / applicable) if applicable else 0.0, n, k,
                         None if hon is None else hon < 1.0)
