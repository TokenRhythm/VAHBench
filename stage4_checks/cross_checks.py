"""The flow ledger: every inference that crossed a channel, written down.

Three channels feed the rubric -- the brief (what was asked), the trajectory
(what the agent did and said) and the delivery (what shipped) -- and the
value of reading them together is exactly the set of inferences none of them
supports alone:

    a derived "generated 30 s" figure becomes VOID only when the pixels show
    the loop; a silent track turns "58 words < 125" from a pass into a void;
    an audio_gen count of zero turns "silent" from "muxing failed" into
    "never generated"; a keyframe the agent drew becomes the reference a
    delivered shot is held against.

Each such inference is a *flow*: a source (what was read), a destination
(the check it changed or configured) and an effect. Flows are recorded here
so a report shows which cross-channel rules fired on a run, instead of
leaving a reader to reconstruct them from two sections.

Nothing in this module measures anything. `apply()` re-reads measurements
that already exist in the light of facts from another channel; the evidence
bus and the expectations log their own flows into the same ledger.

Effects:
    void         a measured pass/fail became VOID (the video voided it)
    fail         a pass became a FAIL on another channel's evidence
    upgrade      a FAIL became VOID because a second channel agreed
    reinterpret  verdict unchanged, reason rewritten with the other channel
    exempt       a FAIL became a PASS because another channel explains every
                 flagged window (no rule uses it since `narration_exempts_mouth`
                 went with `av.mouth_speech` on 2026-09-22; kept in the vocabulary)
    annotate     detail added, verdict and magnitude untouched
    configure    an upstream fact changed how a downstream module was run

The plan-text rules at the bottom share one primitive: `plan_spans`, the
agent's shot plan for each delivered episode as `{video name: [{start, end,
text}]}` (evidence.plan_spans). A module reports what it measured at a time;
the plan says what was meant to be there. No module is handed a flag for
this -- the join is made here, on the module's own timestamps.
"""
from __future__ import annotations

import re
from dataclasses import asdict
from dataclasses import dataclass

from shared.types import Measurement
from shared.types import Scale
from shared.types import Verdict
from stage1_inputs.schema import CONSUMERS
from stage1_inputs.schema import _flatten

AUDIO_EXT = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus")


@dataclass
class Flow:
    rule: str
    src: list[str]
    dst: str
    effect: str
    note: str = ""


class Ledger:
    def __init__(self) -> None:
        self.entries: list[Flow] = []

    def add(self, rule: str, src, dst: str, effect: str, note: str = "") -> None:
        if isinstance(src, str):
            src = [src]
        self.entries.append(Flow(rule, [str(s) for s in src], dst, effect, note[:240]))

    @property
    def rules(self) -> set[str]:
        return {e.rule for e in self.entries}

    def to_list(self) -> list[dict]:
        return [asdict(e) for e in self.entries]


# ------------------------------------------------------------- helpers ----
def _replace(ms: dict, key: str, verdict, reason: str, *, scale=None, value=None,
             detail_extra: dict | None = None) -> None:
    old = ms.get(key)
    ms[key] = Measurement(
        key=key, verdict=verdict,
        scale=scale if scale is not None else (old.scale if old else Scale.BOOLEAN),
        value=value if value is not None else (old.value if old else None),
        reason=reason, detail={**(old.detail if old else {}), **(detail_extra or {})})


def _annotate(ms: dict, key: str, **extra) -> None:
    m = ms.get(key)
    if m is not None:
        m.detail.update(extra)


# --------------------------------------------------------------- rules ----
def pixels_corroborate_duration(ms: dict, ledger: Ledger) -> None:
    """A *derived* trajectory figure (typical clip length x call count) may
    reach FAIL on its own but not VOID -- either factor can be wrong. When the
    pixel side independently reports the video as degenerate, two unrelated
    channels agree and the verdict is upgraded."""
    dp = ms.get("provenance.duration_provenance")
    if dp is None or dp.verdict is not Verdict.FAIL:
        return
    dup, frozen = ms.get("visual.near_duplicate_shots"), ms.get("artifact.frozen_runs")
    agreeing = [k for k, m in (("visual.near_duplicate_shots", dup),
                               ("artifact.frozen_runs", frozen))
                if m is not None and m.verdict in (Verdict.FAIL, Verdict.VOID)
                and (m.value or 0) > 0]
    if not agreeing:
        return
    _replace(ms, dp.key, Verdict.VOID,
             dp.reason.split("（生成时长")[0] + "；像素侧印证：镜头近重复/冻结",
             detail_extra={"corroborated_by": agreeing})
    ledger.add("pixels_corroborate_duration", agreeing + ["requested.generated_seconds"],
               dp.key, "upgrade", "轨迹推得的生成时长不足，像素侧独立看到重复/冻结：FAIL→VOID")


def drift_corroborates_missing_conditioning(ms: dict, ledger: Ledger) -> None:
    """Cross-shot scale drift, read against whether the agent built an anchor.

    This is the one plausibility finding the agent could have prevented. A
    clip generated from text alone has no way to know how big the chair was
    in the previous shot; a clip conditioned on a keyframe does. So the same
    measured drift means opposite things about the agent depending on whether
    it carried continuity forward, and the trajectory is the only channel that
    knows which happened.

    Two unambiguous ends, and nothing invented in between -- the same caution
    EVAL_DESIGN 6.2c (14) and (17) were written about:

        no keyframe at all    the agent never created a channel for scale to
                              flow across the cut, and the pixels show the
                              consequence. Two unrelated channels agree, so
                              conditioning_edges is upgraded FAIL -> VOID by
                              the same rule as pixels_corroborate_duration.
        every clip anchored   the agent did its part and the model ignored it.
                              The defect stays scored in O4 -- it shipped --
                              but it is attributed to the model, not the agent.
        partial               coverage recorded, no verdict moved.
    """
    drift = ms.get("visual.scale_consistency")
    ce = ms.get("provenance.conditioning_edges")
    if drift is None or drift.verdict is not Verdict.FAIL:
        return
    if ce is None or ce.verdict is Verdict.ABSTAIN:
        return
    clips = (ce.detail or {}).get("clips") or {}
    n, kf = int(clips.get("n") or 0), int(clips.get("with_keyframe") or 0)
    if not n:
        return

    if kf == 0:
        _annotate(ms, drift.key, attribution="agent", clips_with_keyframe=f"0/{n}")
        if ce.verdict is Verdict.FAIL:
            _annotate(ms, ce.key, corroborated_by=["visual.scale_consistency"])
            ledger.add("drift_corroborates_missing_conditioning",
                       ["visual.scale_consistency"], ce.key, "annotate",
                       "漂移只用于解释；规划按条件资产实测缺失比例计分，不二次加重")
        else:
            ledger.add("drift_corroborates_missing_conditioning",
                       ["visual.scale_consistency"], drift.key, "annotate",
                       "零条件边 + 实测漂移：信息从未在镜头之间流动，归因 agent")
    elif kf >= n:
        _annotate(ms, drift.key, attribution="model", clips_with_keyframe=f"{kf}/{n}")
        ledger.add("drift_corroborates_missing_conditioning",
                   ["provenance.conditioning_edges"], drift.key, "annotate",
                   f"{kf}/{n} 个片段都有条件关键帧，漂移仍发生：agent 已尽责，归因视频模型")
    else:
        _annotate(ms, drift.key, attribution="partial", clips_with_keyframe=f"{kf}/{n}")
        ledger.add("drift_corroborates_missing_conditioning",
                   ["provenance.conditioning_edges"], drift.key, "annotate",
                   f"条件边覆盖 {kf}/{n}，不足以把漂移判给任何一方，只记录")


CONSISTENCY_CHECKS = ("visual.identity_consistency", "visual.scale_consistency")


def contract_explains_consistency(ms: dict, ledger: Ledger) -> None:
    """A cross-shot consistency failure, attributed by where the chain broke.

    The same question `drift_corroborates_missing_conditioning` asks about
    keyframes, asked about the text the agent wrote. A character cannot stay
    the same across a cut if nothing ever said what she looks like, and the
    trajectory is the only channel that knows whether anything did:

        brief "navy officewear, silver pencil behind one ear"
          -> brief.continuity_contract   did the agent write it down
          -> brief.specs_in_prompts      did its own prompts quote it
          -> visual.identity_consistency did the delivery hold it

    Every branch is decided by an upstream check's own verdict. Nothing here
    invents a cut point, and "could not be measured" is kept distinct from
    "was not done" -- three of five harnesses log their prompts only inside
    trajectory exports nothing deterministic can read, and reporting that as a
    broken chain would blame them for our blind spot.

        contract ABSTAIN    nothing was written and the pixels show the
                            consequence. Two unrelated channels agree, so the
                            contract finding is upgraded -> VOID, as in
                            pixels_corroborate_duration.
        contract FAIL       some character has no appearance spec anywhere.
                            The chain demonstrably broke at step one: agent.
        prompts FAIL        the spec was written and the agent's own prompts
                            never quoted it -- an asset built and not used,
                            which is still the agent's: agent.
        both PASS           the agent did everything the trajectory can show
                            and the delivery drifted anyway: the model.
        prompts ABSTAIN     unmeasured. No attribution, no verdict moved.
    """
    failed = [k for k in CONSISTENCY_CHECKS
              if (m := ms.get(k)) is not None and m.verdict is Verdict.FAIL]
    if not failed:
        return
    contract = ms.get("brief.continuity_contract")
    prompts = ms.get("brief.specs_in_prompts")
    if contract is None:
        return

    detail = {"contract": contract.verdict.value,
              "contract_deficit": contract.value,
              "prompts": prompts.verdict.value if prompts is not None else None,
              "prompts_deficit": prompts.value if prompts is not None else None}

    def mark(attribution, stage, note, *, src=None, upgrade=False):
        for k in failed:
            _annotate(ms, k, attribution=attribution, chain_broke_at=stage, **detail)
        if upgrade:
            _replace(ms, contract.key, Verdict.VOID,
                     (contract.reason or "轨迹里没有连续性文本")
                     + "；像素侧印证：成片确有跨镜头不一致",
                     detail_extra={"corroborated_by": failed})
            ledger.add("contract_explains_consistency", failed, contract.key,
                       "upgrade", note)
        else:
            ledger.add("contract_explains_consistency",
                       src or ["brief.continuity_contract"], failed[0],
                       "annotate", note)

    if contract.verdict is Verdict.ABSTAIN:
        mark(None, "unmeasured",
             "连续性文本不可测；成片漂移不能反证轨迹未做规划，保留缺测")
    elif contract.verdict is Verdict.FAIL:
        mark("agent", "contract",
             "有角色的外观从未被写下，成片跨镜头不一致：链条断在第一环，归因 agent")
    elif prompts is None or prompts.verdict is Verdict.ABSTAIN:
        mark(None, "unmeasured",
             "外观写全了，但逐次 prompt 不可确定性恢复：无法判断链条是否继续，不归因")
    elif prompts.verdict is Verdict.FAIL:
        mark("agent", "prompts",
             "外观写下了却没进自己的 prompt：建了资产没有使用，归因 agent",
             src=["brief.specs_in_prompts"])
    else:
        mark("model", "generation",
             "外观写全了、也进了自己的 prompt，成片仍不一致：agent 已尽责，归因视频模型",
             src=["brief.continuity_contract", "brief.specs_in_prompts"])


def self_checks_contradicted(ms: dict, facts, ledger: Ledger) -> None:
    """The agent's own audioOk / durationOk / resolutionOk against ours.
    claudecode wrote audioOk=true; whether the track carries signal is ours
    to hear."""
    checks = getattr(getattr(facts, "claims", None), "self_checks", None) or {}
    contradicted, keys = [], []
    for claim, key in (("audioOk", "spec.audio_track"), ("durationOk", "spec.duration"),
                       ("resolutionOk", "spec.resolution")):
        m = ms.get(key)
        if checks.get(claim) is True and m is not None \
                and m.verdict in (Verdict.FAIL, Verdict.VOID):
            contradicted.append(f"自述 {claim}=true，实测 {m.reason or key + ' 未达标'}")
            keys.append(key)
    if not contradicted:
        if checks:
            observable = all(value is False or (ms.get(key) is not None and
                             ms[key].verdict is Verdict.PASS)
                             for claim, key in (("audioOk", "spec.audio_track"),
                                                ("durationOk", "spec.duration"),
                                                ("resolutionOk", "spec.resolution"))
                             if (value := checks.get(claim)) is not None)
            _annotate(ms, "claims.deliverables_consistent", self_checks=checks,
                      self_checks_consistent=True if observable else None)
            ledger.add("self_checks_contradicted", ["claims.self_checks"],
                       "claims.deliverables_consistent", "annotate",
                       f"自检 {', '.join(sorted(checks))}：" + ("与实测一致" if observable else "仍缺独立测量"))
        return
    _replace(ms, "claims.deliverables_consistent", Verdict.FAIL,
             "；".join(contradicted)[:200], scale=Scale.BOOLEAN,
             detail_extra={"self_checks": checks})
    ledger.add("self_checks_contradicted", ["claims.self_checks"] + keys,
               "claims.deliverables_consistent", "fail", "；".join(contradicted)[:160])


def _produced_audio(facts) -> int:
    out = 0
    for a in (getattr(facts, "produced", None) or []):
        kind = str(a.kind or "").lower()
        if kind in ("audio", "tts", "speech", "music") or \
                str(a.path).lower().endswith(AUDIO_EXT):
            out += 1
    return out


def audio_triangulation(ms: dict, facts, ledger: Ledger) -> None:
    """A silent delivery means different things depending on the trajectory.

    audio_gen > 0 (or audio files produced) + silent  -> generated, never muxed
                                                         (execution failure)
    audio_gen == 0 and nothing produced       + silent  -> the audio requirement
                                                         was ignored
    audio_gen == 0                            + signal  -> the video model
                                                         supplied the track
    The verdict on spec.audio_track is the pixel side's and does not move;
    what changes is *which* failure the reader is looking at (EVAL_DESIGN 4.5
    ③: ignoring a requirement is the worse signal).
    """
    at = ms.get("spec.audio_track")
    if at is None or at.verdict is Verdict.ABSTAIN:
        return
    n_gen = getattr(getattr(facts, "calls", None), "audio_gen", None)
    n_files = _produced_audio(facts)
    if at.verdict in (Verdict.FAIL, Verdict.VOID):
        if (n_gen or 0) > 0 or n_files > 0:
            tri, why = "planned_not_delivered", (
                f"轨迹里有 {n_gen or 0} 次音频生成调用、{n_files} 个音频产物，"
                "成片却无声：生成了未合入（执行失败）")
        elif n_gen == 0:
            tri, why = "ignored", "轨迹里没有任何音频生成，成片无声：音频要求被忽略"
        else:
            tri, why = "unknown", "轨迹未记录音频生成情况"
        _replace(ms, at.key, at.verdict, f"{at.reason}；{why}",
                 detail_extra={"triangulation": tri, "audio_gen_calls": n_gen,
                               "audio_files_produced": n_files})
        ledger.add("audio_triangulation", ["calls.audio_gen", "produced"], at.key,
                   "reinterpret", why)
    elif n_gen == 0 and n_files == 0:
        _annotate(ms, at.key, audio_source="video_model_or_unrecorded",
                  audio_gen_calls=0)
        ledger.add("audio_triangulation", ["calls.audio_gen"], at.key, "annotate",
                   "轨迹无音频生成调用而成片有声：音轨由视频模型内生或未被记录")


_DEVIATION_KEYS = {
    "spec.audio_track": re.compile(r"silent|audio|sound|mute|voice", re.I),
    "av.clipping": re.compile(r"silent|audio|sound|mute", re.I),
    "av.dynamic_range": re.compile(r"silent|audio|sound|mute", re.I),
    "av.speaker_consistency": re.compile(r"silent|audio|sound|mute|voice|tts", re.I),
    "spec.aspect": re.compile(r"aspect|ratio|\d+:\d+", re.I),
    "o2.requested_params": re.compile(r"aspect|ratio|\d+:\d+|resolution|\d{3,4}p", re.I),
    "spec.resolution": re.compile(r"resolution|\d{3,4}x\d{3,4}|\d{3,4}p", re.I),
    "spec.duration": re.compile(r"duration|length|seconds", re.I),
    "o2.episode_count": re.compile(r"episode|stage|intervention|skipped", re.I),
}


def deviation_attribution(ms: dict, facts, ledger: Ledger) -> None:
    """run.json `deviations` are the harness operator's notes on how the run
    was set up ("storyvideo_gen has audio integration commented out
    upstream -- output is silent"). They never excuse a finding -- the
    benchmark scores the harness as it ran -- but a FAIL that a declared
    deviation predicts is a different reading from one nobody saw coming, so
    the match is attached to the measurement and logged."""
    devs = getattr(getattr(facts, "claims", None), "deviations", None) or []
    if not devs:
        return
    matched = 0
    for key, pat in _DEVIATION_KEYS.items():
        m = ms.get(key)
        if m is None or m.verdict not in (Verdict.FAIL, Verdict.VOID):
            continue
        hits = [d for d in devs if pat.search(str(d))]
        if not hits:
            continue
        matched += 1
        _annotate(ms, key, declared_deviation=str(hits[0])[:200])
        ledger.add("deviation_attribution", ["claims.deviations"], key, "annotate",
                   f"该失败与 run.json 声明的偏差一致：{str(hits[0])[:120]}")
    if not matched:
        ledger.add("deviation_attribution", ["claims.deviations"], "report", "annotate",
                   f"run.json 声明了 {len(devs)} 条偏差，没有一条对应到任何失败")


def requested_vs_delivered(ms: dict, ledger: Ledger) -> None:
    """Declared intent against the delivered file, for aspect and resolution.
    Wrong request + right delivery = fixed in post (crop/scale); right request
    + wrong delivery = broken at assembly/export. Neither moves a verdict --
    both checks already carry theirs -- but the stage that went wrong is
    now named."""
    rq = ms.get("o2.requested_params")
    if rq is None or rq.verdict is Verdict.ABSTAIN:
        return
    for key, label, marker in (("spec.aspect", "画幅", "画幅"),
                               ("spec.resolution", "分辨率", "短边")):
        m = ms.get(key)
        if m is None or m.verdict is Verdict.ABSTAIN:
            continue
        bad_req = rq.verdict is Verdict.FAIL and marker in (rq.reason or "")
        bad_out = m.verdict in (Verdict.FAIL, Verdict.VOID)
        if bad_req and not bad_out:
            note = f"生成请求的{label}与任务不符，但交付前已修正（后期裁切/缩放）"
            _annotate(ms, "o2.requested_params", **{f"{key}_fixed_in_post": True})
            dst = "o2.requested_params"
        elif bad_out and not bad_req:
            note = f"生成请求的{label}正确，交付时被改变：问题出在合成/导出阶段"
            _annotate(ms, key, stage="assembly_or_export")
            dst = key
        else:
            continue
        ledger.add("requested_vs_delivered",
                   ["requested.aspect_ratios", "requested.resolutions", key],
                   dst, "annotate", note)


def seam_alignment(ms: dict, facts, splice_times: dict, ledger: Ledger,
                   tolerance: float = 1.0) -> None:
    """Declared clip structure -> predicted seam timestamps -> measured
    splices. `claims.structure_consistent` compares counts; this compares
    positions, which is what tells a trimmed clip from a re-ordered one. It
    annotates only: a seam that moved is a cut the agent made, not a defect."""
    st = ms.get("claims.structure_consistent")
    if st is None or st.verdict is Verdict.ABSTAIN or not splice_times:
        return
    clips = getattr(facts.claims, "declared_clips_per_episode", None) or []
    durs = getattr(facts.requested, "clip_durations", None) or []
    if not clips or not durs:
        return
    per_request = len(durs) == sum(clips)
    typical = sum(durs) / len(durs)
    rows, pos = [], 0
    for i, (video, measured) in enumerate(sorted(splice_times.items())):
        if i >= len(clips):
            break
        n = clips[i]
        ep_durs = durs[pos: pos + n] if per_request else [typical] * n
        pos += n
        expected, t = [], 0.0
        for d in ep_durs[:-1]:
            t += float(d)
            expected.append(round(t, 2))
        matched = sum(1 for e in expected
                      if any(abs(e - m) <= tolerance for m in measured))
        rows.append({"video": str(video), "expected": expected,
                     "measured": [round(float(m), 2) for m in measured],
                     "matched": matched, "n": len(expected)})
    if not rows:
        return
    n_exp = sum(r["n"] for r in rows)
    n_hit = sum(r["matched"] for r in rows)
    _annotate(ms, st.key, seam_alignment={
        "rows": rows, "matched": n_hit, "n": n_exp,
        "durations": "per_request" if per_request else "typical"})
    note = (f"声明片长推得的 {n_exp} 个拼接点，{n_hit} 个在像素拼接点 ±{tolerance:.0f}s 内"
            + ("" if n_exp == 0 or n_hit / n_exp >= 0.5 else
               "：段数一致但位置不符，片段被裁剪或重排"))
    ledger.add("seam_alignment", ["claims.declared_clips_per_episode",
                                  "requested.clip_durations", "artifact.splices"],
               st.key, "annotate", note)


# ------------------------------------------------------ plan-text rules ----
# Script conventions for narration: "V.O.", "(V.O.)", "voiceover", "narrator",
# "O.S." (off screen). Matched on the plan beat's own text.
VO_RE = re.compile(r"\bv\.\s?o\.?(?=\W|$)|\bvo\b|voice-?\s?over|\bnarrat(?:ion|or|ed|es|ing)\b"
                   r"|off-?\s?screen|\bo\.s\.(?=\W|$)", re.I)
# Content on which monocular depth loses its edges: participating media,
# specular or transparent surfaces, motion blur, deliberate unreality. A
# penetration flag inside such a beat is a depth-map artefact until proven
# otherwise, and the plan is the only channel that says the beat has it.
FX_RE = re.compile(r"\b(smoke|fog|haze|mist|steam|rain|downpour|storm|underwater|splash\w*"
                   r"|snow|dust|explosions?|explod\w*|fire|flames?|sparks?|embers?"
                   r"|reflections?|mirrors?|blur\w*|slow[- ]?motion|dream\w*|strobe"
                   r"|lens flare|hologram\w*)\b", re.I)


def _beats_at(spans: list[dict], start, end) -> list[dict]:
    if start is None or end is None:
        return []
    return [s for s in spans
            if min(float(s["end"]), float(end)) - max(float(s["start"]), float(start)) > 0]


def _spans_for(plan_spans: dict, m: Measurement) -> list[dict]:
    """The plan for the episode a merged measurement came from. Measurements
    carry `episode` (the file name, or full path for duplicate names) from run_eval; a map with one
    episode needs no key."""
    if not plan_spans:
        return []
    ep = (m.detail or {}).get("episode")
    if ep in plan_spans:
        return plan_spans[ep] or []
    if len(plan_spans) == 1:
        return next(iter(plan_spans.values())) or []
    return []


def plan_vs_brief_voiceover(ms: dict, plan_spans: dict, voiceover_allowed,
                            ledger: Ledger) -> None:
    """Under a narration ban, the plan speaks to brief.no_voiceover as well
    as presence's `offscreen_voice`: did the agent *write* narration into a
    beat -- a planned violation. It never moves the verdict, which presence
    owns; it is attached so a PASS on presence alone can be read against it,
    which is the triangulation EVAL_DESIGN 4.5 ③ asks for: ignored the rule,
    or planned around it.

    Until 2026-09-22 this also attached av/mouth's speech-over-a-still-face
    windows. That measurement (`av.mouth_speech`) was deleted with the
    diagnostic bucket, and with it `narration_exempts_mouth`, the rule that
    read the plan's V.O. marks to exempt those windows."""
    if voiceover_allowed is not False:
        return
    nv = ms.get("brief.no_voiceover")
    if nv is None:
        return
    hits = [(name, b) for name, spans in (plan_spans or {}).items()
            for b in (spans or []) if VO_RE.search(b["text"])]
    if not hits:
        if plan_spans:
            _annotate(ms, nv.key, planned_voiceover_beats=[])
            ledger.add("plan_vs_brief_voiceover", ["continuity.shot_plan"], nv.key,
                       "annotate", "分镜计划里没有任何一拍标注旁白/V.O.，与任务书禁令一致")
        return
    extra = {"planned_voiceover_beats": [
        {"episode": n, "start": b["start"], "end": b["end"], "text": b["text"][:80]}
        for n, b in hits[:8]], "planned_violation": True}
    notes = [f"分镜计划有 {len(hits)} 拍标注旁白/V.O.（任务书禁止）"]
    if nv.verdict is Verdict.PASS:
        notes.append("presence 未见画外人声：计划了旁白但成片里没听出来，或旁白配了在场的脸")
    _annotate(ms, nv.key, **extra)
    ledger.add("plan_vs_brief_voiceover", ["continuity.shot_plan"], nv.key,
               "annotate", "；".join(notes))


def plan_explains_physical(ms: dict, plan_spans: dict, ledger: Ledger) -> None:
    """Attach planned context to physical findings, without changing scores.

    Smoke, water, glass and blur may make monocular boundaries unreliable.
    Plan text annotates that limitation; it cannot establish clean geometry.
    The size-proxy branch remains for compatibility with existing consumers.
    """
    m = ms.get("physical.penetration")
    if m is not None and m.verdict is Verdict.FAIL:
        spans = _spans_for(plan_spans, m)
        shots = (m.detail or {}).get("flagged_shots") or []
        rows, all_fx = [], bool(shots)
        for s in shots:
            beats = _beats_at(spans, s.get("start"), s.get("end"))
            text = " / ".join(b["text"] for b in beats)
            fx = sorted({x.group(0).lower() for x in FX_RE.finditer(text)})
            rows.append({"shot": s.get("shot"), "start": s.get("start"), "end": s.get("end"),
                         "n_flagged": s.get("n_flagged"), "plan": text[:120], "effects": fx})
            if not fx:
                all_fx = False
        if rows and spans:
            _annotate(ms, m.key, planned_beats=rows)
            if all_fx:
                fx_names = sorted({e for r in rows for e in r["effects"]})[:4]
                _replace(ms, m.key, Verdict.FAIL,
                         (m.reason or "") + f"；flag 全部落在计划里有 {'/'.join(fx_names)} 的镜头："
                         "单目深度在此类内容上边界不可靠，读数存疑",
                         detail_extra={"depth_unreliable": True})
                effect, note = "reinterpret", (
                    f"{len(rows)} 个带 flag 的镜头，计划文字全部含 {'/'.join(fx_names)}："
                    "深度边界不可靠，穿模读数归为仪器存疑")
            else:
                n_fx = sum(1 for r in rows if r["effects"])
                effect, note = "annotate", (
                    f"{len(rows)} 个带 flag 的镜头已对到计划的拍（{n_fx} 个含特效/介质描述）")
            ledger.add("plan_explains_physical", ["continuity.shot_plan", "physical.per_shot"],
                       m.key, effect, note)
    d = ms.get("visual.scale_consistency")
    if d is not None and d.verdict is Verdict.FAIL:
        spans = _spans_for(plan_spans, d)
        beats = []
        for span in ((d.detail or {}).get("worst_pair_spans") or []):
            if isinstance(span, (list, tuple)) and len(span) == 2:
                beats.append([b["text"][:100] for b in _beats_at(spans, span[0], span[1])])
        if beats and any(beats):
            _annotate(ms, d.key, worst_pair_planned_beats=beats)
            ledger.add("plan_explains_physical", ["continuity.shot_plan", "physical.scale_drift"],
                       d.key, "annotate", "漂移最大的类别对所在镜头已对到计划的拍")


def cast_sizes_identities(ms: dict, n_cast, ledger: Ledger) -> None:
    """Annotate surplus recurring face clusters; never alter identity scores.

    Clustering can split a drifting face, but a cluster count alone does not
    identify a named character. The historical cast + 2 hint is metadata only.
    """
    m = ms.get("visual.identity_consistency")
    if m is None or not n_cast:
        return
    n_cross = (m.detail or {}).get("n_cross_shot_identities")
    if n_cross is None:
        return
    limit = int(n_cast) + 2
    if int(n_cross) > limit:
        _annotate(ms, m.key, identity_surplus=int(n_cross) - limit, expected_cast=int(n_cast),
                  identity_limit=limit)
        ledger.add("cast_sizes_identities", ["brief.characters", "visual.identities"], m.key,
                   "annotate",
                   f"跨镜头复现的人脸身份 {n_cross} 个，任务书 {n_cast} 个角色（容忍到 {limit}）："
                   "簇内一致性不会因拆簇而降分，多出的身份在此点名")


def handles_attribution(ms: dict, ledger: Ledger) -> None:
    """Still time after cuts (pixels) -> the source clips' slack (produced)
    -> whose debt it is. `seam.cut_continuity` charges the broken cuts; this
    entry says how the still time behind them splits, so a still that no trim
    could have removed reads as the material's, not the editor's."""
    cc = ms.get("seam.cut_continuity")
    if cc is None or cc.verdict in (Verdict.ABSTAIN, Verdict.NOT_APPLICABLE):
        return
    d = cc.detail
    if "shipped_s" not in d:
        return
    ship, trim, mat = float(d.get("shipped_s") or 0), float(d.get("trimmable_s") or 0), float(d.get("material_s") or 0)
    if ship <= 0:
        ledger.add("handles_attribution", ["seam.cut_continuity", "produced"], cc.key, "annotate",
                   "切点后没有静止手柄，无需裁剪")
        return
    ledger.add("handles_attribution", ["seam.cut_continuity", "produced"], cc.key, "annotate",
               f"切点后静止 {ship:.1f}s：{trim:.1f}s 源片段留有余量、agent 未裁；"
               f"{mat:.1f}s 余量不够、只能重生成或多要手柄（不记 agent）")


def apply(ms: dict, facts, ledger: Ledger, *, splice_times: dict | None = None,
          plan_spans: dict | None = None, voiceover_allowed=None, n_cast=None) -> None:
    """All trajectory x delivery rules, in dependency order."""
    handles_attribution(ms, ledger)
    pixels_corroborate_duration(ms, ledger)
    drift_corroborates_missing_conditioning(ms, ledger)
    contract_explains_consistency(ms, ledger)
    self_checks_contradicted(ms, facts, ledger)
    audio_triangulation(ms, facts, ledger)
    requested_vs_delivered(ms, ledger)
    seam_alignment(ms, facts, splice_times or {}, ledger)
    deviation_attribution(ms, facts, ledger)
    # plan text over module timestamps
    plan_vs_brief_voiceover(ms, plan_spans or {}, voiceover_allowed, ledger)
    plan_explains_physical(ms, plan_spans or {}, ledger)
    cast_sizes_identities(ms, n_cast, ledger)


# ------------------------------------------------------- checklist use ----
def checklist_usage(facts, ms: dict, ledger: Ledger) -> dict:
    """Per checklist field: was it observable on this run, who consumes it,
    and which of those consumers actually fired. "Fully used" is a property
    you can read off a report, not a promise in a docstring."""
    fired_rules = ledger.rules
    live = {k for k, m in ms.items() if m.verdict is not Verdict.ABSTAIN}
    rows: dict[str, dict] = {}
    for name, value in _flatten(facts):
        consumers = CONSUMERS.get(name, [])
        fired = []
        for kind, target in consumers:
            if kind == "check" and target in live:
                fired.append(f"check:{target}")
            elif kind in ("flow", "config") and target in fired_rules:
                fired.append(f"{kind}:{target}")
            elif kind == "context":
                fired.append(f"context:{target}")
        rows[name] = {"present": value is not None,
                      "consumers": [f"{k}:{t}" for k, t in consumers],
                      "fired": fired}
    present = [k for k, r in rows.items() if r["present"]]
    used = [k for k in present if rows[k]["fired"]]
    return {"fields": rows, "n_present": len(present), "n_present_used": len(used),
            "unused_present": [k for k in present if k not in used]}
