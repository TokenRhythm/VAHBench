"""Shared measurement vocabulary.

The rubric itself is a tree and lives in `shared/rubric/` (SCORING_SYSTEM.md
B); this module holds what every branch of it speaks:

    Verdict      PASS / FAIL / VOID / ABSTAIN / NOT_APPLICABLE
    Scale        BOOLEAN / RATE / HARD_RATE / RATIO -> one shared cut-point band each
    Measurement  one check's result
    Check        one measurable key and the conditions under which it counts

Three reasons a check yields no score, kept apart because they mean opposite
things for the denominator:

    NOT_APPLICABLE   the task never offered the evidence (no reference assets
                     were supplied, so reference fidelity cannot be asked).
                     Leaves the denominator entirely -- it is not a gap.
    ABSTAIN          the check applies but could not be measured. Leaves the
                     score denominator uniformly, lowers coverage, and marks
                     an otherwise measurable score as conditional.
    VOID             a measured failure. Scores zero and stays in the denominator.
                     Dependent checks of the same absence are not extra VOIDs.

`ALL_CHECKS` and its partitions live in `stage5_scoring.registry`.
Consumers import that registry explicitly; this vocabulary has no tree dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import Enum


class Verdict(str, Enum):
    """One of five outcomes. See EVAL_DESIGN.md 6.2.

    ABSTAIN means unavailable evidence, excluded uniformly from score means.
    It still reduces coverage; it is never silently a pass or a zero.
    VOID is a measured zero. NOT_APPLICABLE removes an item from scoring.
    """

    PASS = "PASS"
    FAIL = "FAIL"
    ABSTAIN = "ABSTAIN"                  # applies, could not measure
    NOT_APPLICABLE = "NOT_APPLICABLE"    # does not apply to this task
    VOID = "VOID"                        # measured, but the video voided it


class Scale(str, Enum):
    """How a measurement's magnitude should be read.

    Each scale maps to one globally shared band of cut points rather than one
    band per item, which keeps the calibration budget small (EVAL_DESIGN.md 6.1).
    """

    RATIO = "ratio"      # multiples of the measurement's own noise floor
    RATE = "rate"        # a statistical defect fraction in [0, 1]
    HARD_RATE = "hard"   # a fraction where any visible occurrence is a defect
    BOOLEAN = "boolean"  # exact predicate, verdict carries everything


# HARD_RATE exists because black frames and API error rates do not belong on
# one tolerance: five percent of a 1440-frame episode is three seconds of
# black, which a single shared band was scoring as flawless.
BANDS = {
    Scale.RATIO: "noise_ratio",
    Scale.RATE: "defect_rate",
    Scale.HARD_RATE: "hard_defect_rate",
}


@dataclass
class Measurement:
    """One check's result."""

    key: str
    verdict: Verdict
    scale: Scale = Scale.BOOLEAN
    value: float | None = None      # the ratio or rate; None for BOOLEAN
    reason: str = ""                # required for anything but PASS
    detail: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scale is not Scale.BOOLEAN and self.verdict in (
                Verdict.PASS, Verdict.FAIL) and self.value is None:
            raise ValueError(f"{self.key}: scale={self.scale.value} needs a value")

    @property
    def measurable(self) -> bool:
        return self.verdict not in (Verdict.ABSTAIN, Verdict.NOT_APPLICABLE)


@dataclass(frozen=True)
class Check:
    """One measurable thing, and the conditions under which it counts."""

    key: str
    implemented: bool = True
    # Unbuilt on purpose, with the reason. The three VLM checks are deferred
    # on compute: a local VLM over every episode of 84 runs is hours to days
    # on this CPU-only machine, and they are the one reproducibility class
    # that is only deterministic inside a fixed environment (EVAL_DESIGN 5.3).
    # Reported apart from `planned`, so "not built yet" and "not built by
    # decision" do not read the same.
    deferred: str = ""
    # Measured and reported, but kept out of every denominator until it is
    # calibrated. Whole-frame DINO consistency is the case: cross-shot
    # dissimilarity legitimately exceeds within-shot noise on any multi-shot
    # narrative (shots differ by design), so the RATIO band -- tuned for
    # like-against-like ratios such as scale drift -- scored fidelity 0.00 for
    # every run in the dataset. visual_consistency's own README says not to
    # read this as the identity number. A dimension at 0 for 100% of entrants
    # is an instrument reading, not a finding; the raw r stays in the report.
    provisional: bool = False
