"""The scoring tree. Directory layout == tree layout (SCORING_SYSTEM.md B).

    S = 100 * C * Q^(3/4) * P^(1/4)

    1  video/                     成片层
       1.1  completeness.py       C   seconds ledger -> per-episode IoU
       1.2  quality/              Q   geometric mean of
            1.2.1 specification.py    规格合规
            1.2.2 consistency.py      一致性
            1.2.3 plausibility.py     合理性
    2  process/                轨迹层
       2.1  reliability.py        P   arithmetic mean of
       2.2  honesty.py
       2.3  planning.py
    node.py       Leaf / Category, check_score, the shared cut points
    registry.py   every measurement key: scored / completeness / diagnostic / deferred
    score.py      the root: CaseScore, tier
"""
from stage5_scoring.node import DEFAULT_CUT_POINTS
from stage5_scoring.node import Category
from stage5_scoring.node import Leaf
from stage5_scoring.node import LeafResult
from stage5_scoring.node import CategoryResult
from stage5_scoring.node import check_score
from stage5_scoring.node import score_leaf
from stage5_scoring.node import score_category
from stage5_scoring.video import EpisodeLedger
from stage5_scoring.video import CompletenessResult
from stage5_scoring.video import ledger_from_reports
from stage5_scoring.video import score_completeness
from stage5_scoring.video import QUALITY
from stage5_scoring.video import QualityResult
from stage5_scoring.video import score_quality
from stage5_scoring.process import TRAJECTORY
from stage5_scoring.process import ProcessResult
from stage5_scoring.process import score_process              # noqa: F401
from stage5_scoring.score import CaseScore
from stage5_scoring.score import score_case
from stage5_scoring.score import tier_of
from stage5_scoring.score import TIERS                      # noqa: F401
import stage5_scoring.registry as registry                                                         # noqa: F401


def tree_spec() -> dict:
    """The static tree, for reports and docs: ids, names, sources, flags."""
    def cat(c):
        return {"id": c.id, "key": c.key, "name": c.name, "note": c.note,
                "leaves": [{"id": lf.id, "key": lf.key, "name": lf.name,
                            "sources": list(lf.sources), "combine": lf.combine,
                            "mapping": lf.mapping,
                            "source_mappings": {k: registry.SOURCE_MAPPINGS.get(k, lf.mapping) for k in lf.sources},
                            "implemented": lf.implemented}
                           for lf in c.leaves]}
    return {
        "formula": "S = 100 * C * Q^(1-w) * P^w",
        "video": {
            "completeness": {"id": "1.1", "inputs": sorted(registry.COMPLETENESS_INPUTS)},
            "quality": {"id": "1.2", "aggregate": "geometric_mean",
                        "dimensions": [cat(c) for c in QUALITY]},
        },
        "trajectory": {"id": "2", "aggregate": "mean",
                       "categories": [cat(c) for c in TRAJECTORY]},
        "diagnostic": sorted(registry.DIAGNOSTIC),
        "deferred": sorted(registry.DEFERRED),
    }
