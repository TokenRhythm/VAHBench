"""1.2.2 Consistency: cross-shot face identity and cut continuity only.

Detector reports -> checks/adapters.py -> per-source band scores -> mean.
The two sources share leaf 1.2.2.b. Disabled leaves retain their public ids
for compatibility and do not participate in scoring or its denominator.

Missing evidence leaves the score denominator and lowers coverage, uniformly
with all branches. With no measurements the dimension is undefined.
"""
from __future__ import annotations

from stage5_scoring.node import Category
from stage5_scoring.node import Leaf

CONSISTENCY = Category(
    "1.2.2", "consistency", "一致性", (
        # Historical rubric slots; only b is active in this version.
        Leaf("1.2.2.a", "character_anchoring", "角色锚定",
             ("brief.characters_anchored",), mapping="fraction", implemented=False),
        Leaf("1.2.2.b", "identity_consistency", "身份一致",
             ("visual.identity_consistency", "seam.cut_continuity")),
        Leaf("1.2.2.c", "scene_anchoring", "场景锚定",
             ("visual.scene_reference",), mapping="fraction", implemented=False),
        Leaf("1.2.2.d", "voice_consistency", "声音一致",
             ("av.speaker_consistency",), implemented=False),
    ),
    note="当前版本仅身份和切点连续性可计分；自制参考图及无对应关系的声音代理统一停用")
