"""1.2.1 规格合规 Specification -- does the delivery meet what was stated.

Two active leaves, each a satisfaction fraction over exact predicates:

  a. format_compliance   resolution / aspect / audio track present
  b. brief_requirements  the brief's decidable clauses: deliverables list, word
                         cap, mandated line and its timing band, narration ban
  c. requested_params    retained for attribution only, uniformly excluded

Duration is deliberately absent: it is a completeness quantity (1.1), and a
60 s brief met with 40 s of footage is scored once, in C.
"""
from __future__ import annotations

from stage5_scoring.node import Category
from stage5_scoring.node import Leaf

SPECIFICATION = Category(
    "1.2.1", "specification", "规格合规", (
        Leaf("1.2.1.a", "format_compliance", "格式合规",
             ("spec.resolution", "spec.aspect", "spec.audio_track")),
        Leaf("1.2.1.b", "brief_requirements", "任务书条款",
             ("brief.deliverables", "brief.word_cap", "brief.mandated_line",
              "brief.mandated_line_placement", "brief.no_voiceover"), mapping="fraction"),
        Leaf("1.2.1.c", "requested_params", "请求参数",
             ("o2.requested_params",), mapping="fraction"),
    ),
    note="满足比例；时长归完成度 1.1")
