"""1.2.3 Plausibility: equal mean of three active leaves.

a: resolvable contact defects, converted to a band score by node.py.
b: mean of clipping, dynamic-range and long-silence scores.
c: subtitle/ASR timed word alignment, used directly as a satisfaction fraction.

Size and voice/face proxies retain their public keys but capabilities.py
excludes them for every entrant. Delivery padding belongs only to C.
Missing evidence leaves the score denominator and lowers coverage; all missing
makes this dimension undefined. Detection and score mapping are kept separate.
"""
from __future__ import annotations

from stage5_scoring.node import Category
from stage5_scoring.node import Leaf

PLAUSIBILITY = Category(
    "1.2.3", "plausibility", "合理性", (
        Leaf("1.2.3.a", "physical_defects", "物理缺陷",
             ("physical.penetration", "visual.scale_consistency")),
        Leaf("1.2.3.b", "audio_defects", "音频缺陷",
             ("av.clipping", "av.dynamic_range", "av.long_silence")),
        Leaf("1.2.3.c", "av_alignment", "声画对应",
             ("av.voice_face_match", "brief.subtitles_aligned"), mapping="fraction",
             note="字幕的双向词对齐率已是连续质量分，不再经通用缺陷档位重复压缩"),
    ),
    note="缺陷率经共享档位映射；缺测影响覆盖率并标记条件分，代理项统一停用")
