"""Tree nodes and the two reductions every branch shares.

    check_score   Measurement -> [0, 1] or None        (one measurement)
    score_leaf    Leaf        -> LeafResult             (mean / min of its sources)
    score_category Category   -> CategoryResult         (mean of its leaves)

Read SCORING_SYSTEM.md B.3 alongside this. Two rules carry the design:

  * ABSTAIN is excluded from every score denominator and lowers coverage.
    VOID is a measured zero. Unsupported capabilities are excluded uniformly.
  * No hand weights anywhere. Siblings are equal; the only free numbers are
    the shared cut points below, and they are archived with every report.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import math

from shared.types import BANDS
from shared.types import Measurement
from shared.types import Scale
from shared.types import Verdict
from stage5_scoring.capabilities import EXCLUDED_SOURCES

DEFAULT_CUT_POINTS: dict = {
    "version": "2026-09-26-ranking-v15-cfloor2",
    # r = observed / own noise floor; a rate is a defect fraction.
    "noise_ratio": {"clean": 1.5, "suspect": 3.0},
    "defect_rate": {"clean": 0.05, "suspect": 0.20},
    "hard_defect_rate": {"clean": 0.002, "suspect": 0.01},
    # Geometric mean floor per quality dimension (HDI practice): one coarse
    # zero must not annihilate Q.
    "epsilon": 0.05,
    # Completeness: the three duration bands (pass / partial / fail) on the
    # relative shortfall, the absolute tolerance floor under the pass band,
    # the per-episode assembly point, and the deduction share that makes a
    # delivery degenerate.
    "completeness": {"tolerance_s": 2.0, "band_pass": 0.05, "band_fail": 0.20,
                     "band_fail_score": 0.5, "assembly_deduction": 0.1,
                     "tier_degenerate_deduction_frac": 0.5},
    # Exponent of P in S = C * Q^(1-w) * P^w. 1/4 = four equal facets.
    "process_weight": 0.25,
    # Evidence-quality flag only; observed-facet totals are still reported.
    "coverage": {"min_for_total": 0.60},
    "subtitle_alignment": {"tolerance_s": 0.5, "min_text_match": 0.8, "max_cue_s": 7.0},
    "status": "uncalibrated",
}

# These counts already measure the fraction of evaluated opportunities that
# failed. A 20% cutoff would erase the distinction between 20% and 100% bad.
FRACTION_SOURCES = frozenset({"seam.cut_continuity", "physical.penetration"})


def source_mapping(key: str, default: str) -> str:
    return "fraction" if key in FRACTION_SOURCES else default


# ---------------------------------------------------------------- nodes --
@dataclass(frozen=True)
class Leaf:
    """A scored line of the rubric: one number from one or more measurements."""

    id: str                      # tree address, e.g. "1.2.1.a"
    key: str                     # stable name, e.g. "format_compliance"
    name: str                    # label for reports
    sources: tuple[str, ...]     # measurement keys it reads
    combine: str = "mean"        # "mean" | "min" over its sources
    mapping: str = "band"        # "band" | "fraction" | "jeffreys"
    implemented: bool = True     # False: listed in the tree, always abstains
    note: str = ""


@dataclass(frozen=True)
class Category:
    """A group of leaves: a quality dimension or a trajectory category."""

    id: str
    key: str
    name: str
    leaves: tuple[Leaf, ...]
    note: str = ""

    @property
    def source_keys(self) -> tuple[str, ...]:
        out: list[str] = []
        for lf in self.leaves:
            out.extend(k for k in lf.sources if k not in out)
        return tuple(out)


@dataclass
class LeafResult:
    id: str
    key: str
    name: str
    score: float | None                      # None == abstained
    per_source: dict[str, float | None] = field(default_factory=dict)
    n_sources: int = 0
    n_measured: int = 0
    reasons: list[str] = field(default_factory=list)   # negative feedback only
    abstain_reason: str = ""
    n_applicable: int = 0
    missing_sources: list[str] = field(default_factory=list)
    not_applicable: bool = False

    @property
    def abstained(self) -> bool:
        return self.score is None


@dataclass
class CategoryResult:
    id: str
    key: str
    name: str
    score: float | None
    coverage: float                          # measured leaves / leaves
    n_leaves: int
    n_measured: int
    leaves: list[LeafResult] = field(default_factory=list)
    observed_score: float | None = None
    source_coverage: float = 0.0
    n_applicable: int = 0
    missing_sources: list[str] = field(default_factory=list)
    reason: str = ""


# ----------------------------------------------------------- reductions --
def _band_score(m: Measurement, cuts: dict) -> float:
    """Numeric magnitude: linear between the shared clean / suspect cuts."""
    if m.scale is Scale.BOOLEAN:
        return 0.0
    band = cuts.get(BANDS[m.scale]) or cuts["defect_rate"]
    clean, suspect = float(band["clean"]), float(band["suspect"])
    v = float(m.value) if m.value is not None else float("inf")
    if v <= clean:
        return 1.0
    if v >= suspect:
        return 0.0
    return (suspect - v) / (suspect - clean)


def check_score(m: Measurement | None, cuts: dict, mapping: str = "band") -> float | None:
    """One measurement -> [0, 1], or None when it leaves the denominator.

    Boolean checks use verdicts. Numeric checks use their magnitude,
    irrespective of PASS/FAIL labels. VOID is zero; ABSTAIN/NA are None:

      band      linear between clean and suspect on the measurement's scale
      fraction  1 - value, for measurements whose value is a "missing share"
      jeffreys  1 - (k + 1/2) / (n + 1) from detail["k"], detail["n"];
                falls back to the band when the counts are absent
    """
    if m is None or m.verdict in (Verdict.ABSTAIN, Verdict.NOT_APPLICABLE):
        return None
    if m.verdict is Verdict.VOID:
        return 0.0
    mapping = source_mapping(m.key, mapping)
    if mapping == "jeffreys":
        k, n = m.detail.get("k"), m.detail.get("n")
        if k is not None and n is not None and float(n) > 0:
            return max(0.0, min(1.0, 1.0 - (float(k) + 0.5) / (float(n) + 1.0)))
        # Counts absent: a PASS without counts is still a pass; a FAIL uses
        # the band so an old-format measurement never scores as unknown.
    if m.scale is Scale.BOOLEAN:
        return 1.0 if m.verdict is Verdict.PASS else 0.0
    if m.value is None or not math.isfinite(float(m.value)) or float(m.value) < 0:
        return None
    if mapping == "fraction" and m.scale is not Scale.BOOLEAN and m.value is not None:
        return max(0.0, min(1.0, 1.0 - float(m.value)))
    return _band_score(m, cuts)


def score_leaf(leaf: Leaf, ms: dict[str, Measurement], cuts: dict,
               ) -> LeafResult:
    """Mean (or min) of the leaf's measured sources; abstains when none measured."""
    per: dict[str, float | None] = {}
    reasons: list[str] = []
    for key in leaf.sources:
        m = ms.get(key)
        supported = leaf.implemented and key not in EXCLUDED_SOURCES
        s = check_score(m, cuts, leaf.mapping) if supported else None
        per[key] = s
        if m is not None and s is not None and s < 1.0 and m.reason:
            reasons.append(f"{key}: {m.reason}")
    vals = [v for v in per.values() if v is not None]
    applicable = [k for k in leaf.sources if leaf.implemented
                  and k not in EXCLUDED_SOURCES
                  and (k not in ms or ms[k].verdict is not Verdict.NOT_APPLICABLE)]
    missing = [k for k in applicable if per[k] is None or
               (k in ms and ms[k].detail.get("missing_episodes"))]
    na = not applicable
    if not vals:
        why = "; ".join(f"{k}: {ms[k].reason}" for k in leaf.sources
                        if k in ms and ms[k].reason)[:200] or "全部来源无法测量"
        if not leaf.implemented:
            why = "尚未实现"
        elif leaf.sources and all(k in EXCLUDED_SOURCES for k in leaf.sources):
            why = "本版本不计分：" + "；".join(EXCLUDED_SOURCES[k] for k in leaf.sources)
        return LeafResult(leaf.id, leaf.key, leaf.name, None, per,
                          len(leaf.sources), 0, reasons, why, len(applicable), missing, na)
    score = min(vals) if leaf.combine == "min" else sum(vals) / len(vals)
    return LeafResult(leaf.id, leaf.key, leaf.name, round(score, 4), per,
                      len(leaf.sources), len(vals), reasons, "", len(applicable), missing, na)


def score_category(cat: Category, ms: dict[str, Measurement], cuts: dict) -> CategoryResult:
    """Equal mean of observations; missing evidence lowers coverage uniformly."""
    results = [score_leaf(lf, ms, cuts)
               for lf in cat.leaves]
    live = [r.score for r in results if r.score is not None]
    observed_leaves = []
    for leaf, result in zip(cat.leaves, results):
        values = [s for s in result.per_source.values() if s is not None]
        if values:
            observed_leaves.append(min(values) if leaf.combine == "min" else sum(values) / len(values))
    observed = round(sum(observed_leaves) / len(observed_leaves), 4) if observed_leaves else None
    score = round(sum(live) / len(live), 4) if live else None
    n = len(results)
    applicable = sum(not r.not_applicable for r in results)
    ns = sum(r.n_applicable for r in results)
    ks = sum(r.n_measured for r in results)
    ks -= sum(1 for r in results for k in r.missing_sources if r.per_source.get(k) is not None)
    missing = [k for r in results for k in r.missing_sources]
    reason = ""
    if missing and live:
        reason = '条件分：缺测项不计分，覆盖率另报：' + '、'.join(missing)
    elif not live:
        reason = '本维度无适用项' if not applicable else '本维度无可测来源'
    return CategoryResult(cat.id, cat.key, cat.name, score,
                          (len(live) / applicable) if applicable else 0.0, n, len(live), results,
                          observed, ks / ns if ns else 0.0, applicable, missing, reason)


def geometric_mean(values: list[float], epsilon: float) -> float:
    import math
    floored = [max(float(v), epsilon) for v in values]
    return math.exp(sum(math.log(v) for v in floored) / len(floored))


def measurement_rank(m, cuts, mapping="band"):
    """Same mapping for episode selection and scoring; ties retain worse evidence."""
    score = check_score(m, cuts, mapping)
    if score is None:
        return (2 if m.verdict is Verdict.ABSTAIN else 3, 0, 0, 0)
    value = float(m.value or 0)
    severity = {Verdict.VOID: 0, Verdict.FAIL: 1, Verdict.PASS: 2}[m.verdict]
    return (0, score, -value, severity)
