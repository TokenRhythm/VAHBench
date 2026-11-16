"""2.1 Reliability: unresolved agent failures per identifiable production task.

Only independently terminal, task-bound image/video/audio/assembly operations
enter the denominator. Retrying or using a fallback on the SAME output can
recover a failure. Pending, unlinked or ambiguously attributed operations make
the inventory incomplete, so it abstains. The score is the observed success
fraction; Jeffreys' posterior error mean is reported as uncertainty context,
not mixed into the score or used to reward a longer trace.
"""
from __future__ import annotations

from stage5_scoring.node import Category
from stage5_scoring.node import Leaf

RELIABILITY = Category(
    "2.1", "reliability", "可靠性", (
        Leaf("2.1.a", "unrecovered_failures", "未恢复错误",
             ("trace.unrecovered_failures",), mapping="fraction"),
    ),
    note="按可核验的生产任务计数；同任务恢复不扣分；未知结果不冒充成功；平滑值仅报告")
