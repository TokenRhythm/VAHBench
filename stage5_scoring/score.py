"""Completion times an observed-facet weighted geometric mean.

Full observations retain S=100*C*Q**(1-w)*max(P,epsilon)**w. Missing
facets are omitted and original weights renormalised, never filled with
zero or one. Coverage/observability remain separate from the numeric total.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import math

from shared.types import Measurement
from shared.types import Verdict
from stage5_scoring.node import DEFAULT_CUT_POINTS
from stage5_scoring.video.completeness import CompletenessResult
from stage5_scoring.video.completeness import EpisodeLedger
from stage5_scoring.video.completeness import score_completeness
from stage5_scoring.video.quality import QualityResult
from stage5_scoring.video.quality import score_quality
from stage5_scoring.process import ProcessResult
from stage5_scoring.process import score_process
from stage5_scoring.registry import COMPLETENESS_INPUTS
from stage5_scoring.registry import DIAGNOSTIC

TIERS = ("NONE", "UNASSEMBLED", "DEGENERATE", "OFF_SPEC", "PARTIAL", "COMPLETE")


@dataclass
class CaseScore:
    S: float | None
    C: float | None
    Q: float | None
    P: float | None
    tier: str
    completeness: CompletenessResult
    quality: QualityResult
    process: ProcessResult
    coverage: float                        # measured scored leaves / scored leaves
    sufficient: bool
    process_unmeasured: bool = False
    honesty_violation: bool | None = None
    abstained: list[str] = field(default_factory=list)
    voided: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    completeness_inputs: dict = field(default_factory=dict)
    negative_feedback: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    score_is_upper_bound: bool = False
    score_is_partial: bool = False
    aggregation: dict = field(default_factory=dict)


def completion_factor(raw, *, delivered, video_generated, epsilon):
    """Apply the common numeric C floor without requiring a production method.

    A confirmed empty delivery has raw C=0 and receives the same floor.
    An unavailable measurement is never fabricated from the floor.
    video_generated is retained for caller compatibility, not eligibility.
    """
    base = raw if delivered else 0.0
    eligible = base is not None
    value = None if base is None else max(base, epsilon)
    return value, {"raw": base, "minimum": epsilon, "eligible": eligible,
                   "applied": value is not None and base is not None and value > base}


def aggregate_total(c, dimensions, p, cuts):
    """Renormalise the original facet weights over observed factors only.

    Missing Q/P is never replaced with zero or one. C remains the linear
    delivery gate. With no quality/process observations the explicitly
    labelled result measures completeness only, not full video quality.
    """
    w = float(cuts.get("process_weight", .25))
    eps = float(cuts.get("epsilon", .05))
    live = {k: v for k, v in dimensions.items() if v is not None}
    factors = {k: ((1-w)/3, max(v, eps)) for k, v in live.items() if w < 1}
    if p is not None and w > 0:
        factors["process"] = (w, max(p, eps))
    mass = sum(weight for weight, _ in factors.values())
    weights = {k: weight/mass for k, (weight, _) in factors.items()} if mass else {}
    factor = math.prod(value ** weights[k] for k, (_, value) in factors.items()) if mass else 1.0
    total = None if c is None else round(100 * c * factor, 2)
    return total, {"policy": "observed_facets_v1", "weights": weights,
                   "measured_quality_dimensions": list(live),
                   "process_observed": p is not None,
                   "basis": "observed_facets" if mass else "completeness_only",
                   "missing_values_imputed": False}


def tier_of(comp: CompletenessResult, *, delivered: bool, assembled: bool,
            cuts: dict) -> str:
    if not delivered:
        return "NONE"
    if not assembled:
        return "UNASSEMBLED"
    cc = cuts.get("completeness") or {}
    if comp.delivered_s > 0 and comp.deductions_s > comp.delivered_s * float(
            cc.get("tier_degenerate_deduction_frac", 0.5)):
        return "DEGENERATE"
    # The three duration bands, read off C: it is exactly 1.0 when every
    # episode is on spec, and the fail band's own score is the line below
    # which the delivery is off spec.
    if comp.score is None:
        return "PARTIAL"
    if comp.score < float(cc.get("band_fail_score", 0.5)):
        return "OFF_SPEC"
    if comp.score < 1.0 - 1e-9:
        return "PARTIAL"
    return "COMPLETE"


def score_case(measurements: dict[str, Measurement],
               ledgers: list[EpisodeLedger],
               *, n_required: int | None, delivered: bool, assembled: bool = True,
               generated_cap_s: float | None = None,
               generated_s: float | None = None, generated_source: str = "",
               video_generated: bool | None = None,
               tolerance_s: float | None = None, tolerance_source: str = "",
               cuts: dict | None = None) -> CaseScore:
    cuts = {**DEFAULT_CUT_POINTS, **(cuts or {})}
    w = float(cuts.get("process_weight", 0.25))
    notes: list[str] = []

    comp = score_completeness(ledgers if delivered else [], n_required, cuts,
                              generated_cap_s=generated_cap_s,
                              generated_s=generated_s, generated_source=generated_source,
                              tolerance_s=tolerance_s, tolerance_source=tolerance_source)
    if comp.is_upper_bound:
        notes.append("完成度证据不完整：C 及由此得到的 S 仅为上界；缺口：" + "、".join(comp.evidence_gaps))
    C, floor = completion_factor(comp.score, delivered=delivered,
                                 video_generated=video_generated,
                                 epsilon=float(cuts.get("epsilon", .05)))
    qual = score_quality(measurements, cuts)
    proc = score_process(measurements, cuts)

    # Unknown observations lower coverage; optional production methods do not.
    n_leaves = sum(c.n_applicable for c in (*qual.dimensions.values(), *proc.categories.values()))
    n_meas = qual.n_measured + proc.n_measured
    coverage = (n_meas / n_leaves) if n_leaves else 0.0
    need = float((cuts.get("coverage") or {}).get("min_for_total", 0.6))
    sufficient = (coverage >= need and qual.score is not None
                  and (proc.score is not None or w == 0))
    quality_sources = sum(r.n_applicable for c in qual.dimensions.values() for r in c.leaves)
    quality_measured = sum(c.source_coverage * sum(r.n_applicable for r in c.leaves)
                           for c in qual.dimensions.values())
    source_coverage = quality_measured / quality_sources if quality_sources else 0.0
    sufficient = sufficient and source_coverage >= need
    partial = any(c.missing_sources for c in (*qual.dimensions.values(), *proc.categories.values()))
    if partial:
        notes.append("存在缺测：S/Q/P 为已测证据条件分，不能作为完整评价或直接排名；缺测不扣零分，覆盖率另报")

    tier = tier_of(comp, delivered=delivered, assembled=assembled, cuts=cuts)
    process_unmeasured = proc.score is None

    S, aggregation = aggregate_total(C, {k: d.score for k, d in qual.dimensions.items()},
                                     proc.score, cuts)
    aggregation["completeness_floor"] = floor
    if floor["applied"]:
        notes.append(f"统一完成度保底：原始 C={floor['raw']:.4f}，计分 C={C:.4f}；原始时长与扣减不变")
    if C is None:
        notes.append(f"完成度测量未完成：{comp.reason}；这是评测失败，不是 Agent 零分")
    if not sufficient:
        notes.append(f"证据覆盖另报：叶子 {coverage:.0%}，质量来源 {source_coverage:.0%}；"
                     "已测有效项权重归一，未测项不填零、不填满分，低覆盖不单独阻断总分")
    if aggregation["basis"] == "completeness_only":
        notes.append("当前仅有完成度得分，未测质量或过程；不能解释为质量满分")
    if cuts.get("status") == "uncalibrated":
        notes.append("cut points 未经人标校准：读连续值与判定，不要直接引用分数")

    abst = [f"{r.id} {r.key}（{r.abstain_reason}）" if r.abstain_reason else f"{r.id} {r.key}"
            for cat in (*qual.dimensions.values(), *proc.categories.values())
            for r in cat.leaves if r.abstained]
    voided = [f"{k}: {m.reason}" for k, m in measurements.items()
              if m.verdict is Verdict.VOID]
    def _block(keys):
        return {k: {"verdict": measurements[k].verdict.value,
                    "value": measurements[k].value, "reason": measurements[k].reason}
                for k in keys if k in measurements}

    diag = _block(DIAGNOSTIC)
    # C's own inputs, with the verdict text the modules wrote. They are not
    # scored anywhere -- their seconds and counts are in the ledger -- but a
    # reader of the completeness block needs the sentence that goes with the
    # number ("black_frames: 影响 24 帧").
    comp_inputs = _block(COMPLETENESS_INPUTS)
    feedback = [s for cat in (*qual.dimensions.values(), *proc.categories.values())
                for r in cat.leaves if r.score is not None and r.score < 1.0
                for s in r.reasons]

    return CaseScore(S, None if C is None else round(C, 4), qual.score, proc.score,
                     tier, comp, qual, proc, round(coverage, 4), sufficient,
                     process_unmeasured, proc.honesty_violation,
                     abst, voided, diag, comp_inputs, feedback, notes,
                     S is not None and S > 0 and comp.is_upper_bound,
                     partial, aggregation)
