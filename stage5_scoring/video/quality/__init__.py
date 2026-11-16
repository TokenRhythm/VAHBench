"""1.2 质量 Quality -- Q, the geometric mean of three equal dimensions.

    Q = ( prod_k max(D_k, epsilon) )^(1/3)

Geometric, not arithmetic: limited substitutability. A dimension that
collapsed cannot be bought back by the other two, but neither does one coarse
zero erase the whole product (the epsilon floor). Q averages measured
dimensions only; each unmeasured dimension stays None and lowers coverage.
If all dimensions are unmeasured, Q remains None.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

from stage5_scoring.node import Category
from stage5_scoring.node import CategoryResult
from stage5_scoring.node import geometric_mean
from stage5_scoring.node import score_category
from stage5_scoring.video.quality.specification import SPECIFICATION
from stage5_scoring.video.quality.consistency import CONSISTENCY
from stage5_scoring.video.quality.plausibility import PLAUSIBILITY

QUALITY: tuple[Category, ...] = (SPECIFICATION, CONSISTENCY, PLAUSIBILITY)


@dataclass
class QualityResult:
    score: float | None
    dimensions: dict[str, CategoryResult] = field(default_factory=dict)
    coverage: float = 0.0            # measured leaves / leaves, across the three
    n_leaves: int = 0
    n_measured: int = 0
    reason: str = ""


def score_quality(ms: dict, cuts: dict) -> QualityResult:
    dims = {c.key: score_category(c, ms, cuts) for c in QUALITY}
    n = sum(d.n_leaves for d in dims.values())
    k = sum(d.n_measured for d in dims.values())
    applicable = sum(d.n_applicable for d in dims.values())
    coverage = k / applicable if applicable else 0.0
    missing = [f"{d.name}（{d.reason}）" for d in dims.values() if d.score is None]
    live = [d.score for d in dims.values() if d.score is not None]
    if not live:
        return QualityResult(None, dims, coverage, n, k,
                             "维度证据不足：" + "、".join(missing))
    q = geometric_mean(live,
                       float(cuts.get("epsilon", 0.05)))
    return QualityResult(round(q, 4), dims, coverage, n, k,
                         "未测维度不参与质量均值：" + "、".join(missing) if missing else "")
