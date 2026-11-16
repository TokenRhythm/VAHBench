"""1 成片层 Video -- what was delivered, and how good it is.

    1.1 completeness  C  (video/completeness.py)
    1.2 quality       Q  (video/quality/)
"""
from stage5_scoring.video.completeness import EpisodeLedger
from stage5_scoring.video.completeness import CompletenessResult
from stage5_scoring.video.completeness import ledger_from_reports
from stage5_scoring.video.completeness import score_completeness
from stage5_scoring.video.quality import QUALITY
from stage5_scoring.video.quality import QualityResult
from stage5_scoring.video.quality import score_quality           # noqa: F401
