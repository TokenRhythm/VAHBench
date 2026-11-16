"""2.3 Optional planning diagnostics, excluded from scores and coverage.

  a. character_assets    named characters (and delivered clips) for which the
                         agent built a reference image
  b. continuity_contract the brief's wardrobe terms written into the agent's
                         own bible / plan
  c. specs_in_prompts    those terms quoted into its generation prompts
  d. shot_plan           the brief's locations reached by the shot plan

These describe one possible workflow, not requirements on how an agent
must produce video. capabilities.py excludes all four sources uniformly.
Their original measurements remain available for explanation and auditing.
"""
from __future__ import annotations

from stage5_scoring.node import Category
from stage5_scoring.node import Leaf

PLANNING = Category(
    "2.3", "planning", "规划", (
        Leaf("2.3.a", "character_assets", "角色资产",
             ("provenance.conditioning_edges",), mapping="fraction"),
        Leaf("2.3.b", "continuity_contract", "连续性契约",
             ("brief.continuity_contract",), mapping="fraction"),
        Leaf("2.3.c", "specs_in_prompts", "规格进 prompt",
             ("brief.specs_in_prompts",), mapping="fraction"),
        Leaf("2.3.d", "shot_plan", "分镜计划",
             ("brief.scene_plan",), mapping="fraction"),
    ),
    note="轨迹内闭合，零像素")
