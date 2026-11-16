"""Fixed exclusions for this benchmark version, identical for every entrant.

An extractor returning a proxy does not make that proxy a quality metric.
Evidence proxies require the stated correspondence. Intermediate request
parameters are excluded by ownership: final specification is already scored.
"""

UNSUPPORTED_QUALITY_SOURCES = {
    "brief.no_voiceover": "画外对白不等于旁白；缺少对白与叙述的可靠区分证据",
    "o2.requested_params": "中间生成请求不代表交付规格；最终画幅/分辨率由 spec 计分，请求仅用于归因",
    "brief.characters_anchored": "参赛者自制角色图不是任务提供的角色真值",
    "visual.scene_reference": "参赛者自制关键帧不是任务提供的场景真值",
    "av.speaker_consistency": "缺少角色与声音、跨片段说话人的对应证据",
    "av.voice_face_match": "缺少实际说话者与画面人脸的绑定证据",
    "visual.scale_consistency": "缺少跨镜头同一物体实例的对应证据",
}

# A production strategy is not a task requirement. Keep these measurements
# visible for diagnosis, but exclude them identically for every entrant --
# including entrants whose parser recovered a detailed plan/reference graph.
METHOD_DIAGNOSTIC_SOURCES = {
    "provenance.conditioning_edges": "参考资产是可选制作方法，不以是否使用或覆盖比例计分",
    "brief.continuity_contract": "独立连续性文档是可选制作方法，不以文档是否存在计分",
    "brief.specs_in_prompts": "任务约束可以由多种方式实现，不要求逐次提示词重复描述",
    "brief.scene_plan": "显式分镜计划是可选制作方法，最终要求由交付物检查",
}

EXCLUDED_SOURCES = {**UNSUPPORTED_QUALITY_SOURCES, **METHOD_DIAGNOSTIC_SOURCES}
