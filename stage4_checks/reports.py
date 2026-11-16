"""Module reports -> Measurements, on the two shared scales.

This is where EVAL_DESIGN.md 6.1 becomes code. Each module already reports rich
continuous values; the job here is to convert them onto one of two scales so a
single pair of cut points can serve all of them:

    RATIO   how many times its own noise floor a quantity is
    RATE    a defect fraction in [0, 1]

For the similarity metrics the noise floor is already being computed and thrown
away. visual_consistency reports `intra_shot` (agreement *within* one shot,
where content cannot have changed) next to `inter_shot`. Their dissimilarities
divide:

    r = (1 - inter_shot) / (1 - intra_shot)

On the reference episode that is (1-0.353)/(1-0.901) = 6.5 -- cross-shot
inconsistency is six and a half times the within-shot measurement noise. That
number is readable before any calibration set exists, which the raw 0.353
cosine is not.

Every extractor is defensive about field names. Module schemas drift (the
stored ep1_full.json predates the current background block), and an extractor
that raises on a renamed key would take a whole case down with it. A field that
cannot be found yields no Measurement, and the item abstains -- which is the
correct answer for "we could not measure this".
"""
from __future__ import annotations

import re as _re

from shared.types import Measurement
from shared.types import Scale
from shared.types import Verdict
from stage3_evidence.intervals import repeat_evidence
from stage3_evidence.intervals import union_seconds
from stage4_checks.penalties import dependent


def dig(d, *paths, default=None):
    """First dotted path that resolves to a non-None value."""
    for path in paths:
        cur = d
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                cur = None
                break
        if cur is not None:
            return cur
    return default


def ratio_from_similarity(inter: float | None, intra: float | None) -> float | None:
    """Dissimilarity over its own noise floor.

    Returns None when the floor is not estimable -- a perfect intra-shot score
    means the estimator reports no noise at all, and dividing by that would
    manufacture an enormous ratio out of rounding. Same reasoning as
    av_consistency refusing to call a signal noisy when it cannot see a floor.
    """
    if inter is None or intra is None:
        return None
    noise = 1.0 - float(intra)
    if noise <= 1e-3:
        return None
    return max(0.0, 1.0 - float(inter)) / noise


def _m(key, verdict, scale=Scale.BOOLEAN, value=None, reason="", **detail):
    return Measurement(key=key, verdict=verdict, scale=scale, value=value,
                       reason=reason, detail=detail)


def _abstain(key, reason):
    return _m(key, Verdict.ABSTAIN, reason=reason)


# AudioSet labels the av module routes through the face check
# (presence.SPEECH_LABELS). `startswith("speech")` used to miss "Male speech,
# man speaking", "Conversation" and "Narration, monologue", so those windows
# never reached the voice-to-face denominator.
_SPEECH_LABEL = _re.compile(r"speech|conversation|narration|monologue", _re.I)


def is_speech_label(label) -> bool:
    return bool(_SPEECH_LABEL.search(str(label or "")))


def _shot_spans(report: dict) -> dict[int, tuple[float | None, float | None]]:
    """shot_id -> (start, end) from a module's own `shots` list. This is what
    lets a per-shot finding be read against the plan's time spans without
    handing the module a shots file."""
    out = {}
    for i, s in enumerate(report.get("shots") or []):
        if isinstance(s, dict):
            out[int(s.get("shot_id", i))] = (s.get("start"), s.get("end"))
    return out


# ---------------------------------------------------------------- spec ----
_SPEC_KEYS = {
    "duration": "spec.duration",
    "resolution": "spec.resolution",
    "aspect": "spec.aspect",
    "audio": "spec.audio_track",
    "retiming": "spec.retiming",
}


def from_spec(report: dict | None) -> list[Measurement]:
    """spec_checks: exact predicates, so BOOLEAN throughout.

    `retiming` is the one that reads VOID rather than FAIL. A run that slowed
    1060 frames from 44.21 s to 59.97 s satisfies a 60 s brief on paper while
    containing no additional generated content -- scoring that as a mere
    duration miss would let the trick pay (EVAL_DESIGN.md 6.2).
    """
    if not report:
        return []
    checks = dig(report, "checks", default={}) or {}
    out: list[Measurement] = []
    for name, key in _SPEC_KEYS.items():
        c = checks.get(name)
        if not isinstance(c, dict):
            continue
        msg = str(c.get("detail") or c.get("message") or "")
        # spec_checks reports {ok, undetermined, detail}. `undetermined` is
        # checked first because it ships with ok=True -- the module's own rule
        # is that an undetermined check is never counted as a pass, and reading
        # `ok` alone would silently promote "no source clips to compare
        # against" into a clean retiming result.
        if c.get("undetermined"):
            out.append(_abstain(key, msg or f"{name} 未能测量"))
        elif c.get("ok"):
            out.append(_m(key, Verdict.PASS))
        else:
            verdict = Verdict.VOID if name == "retiming" else Verdict.FAIL
            out.append(_m(key, verdict, reason=msg or f"{name} 未达标"))

    return out


def spec_from_probe(info, target) -> list[Measurement]:
    """Fallback when spec_checks could not run: the subset ffprobe can answer.

    Deliberately narrow. It covers geometry and audio presence and stops there;
    `retiming` needs the source clips and is left to the module, because a
    weaker stand-in reported under the same key would quietly downgrade the
    strongest evidence the pipeline has.
    """
    out: list[Measurement] = []
    if target.seconds_per_episode and info.duration:
        err = abs(info.duration - target.seconds_per_episode) / target.seconds_per_episode
        out.append(_m("spec.duration",
                      Verdict.PASS if err <= 0.05 else Verdict.FAIL,
                      reason="" if err <= 0.05 else
                      f"时长 {info.duration:.1f}s，目标 {target.seconds_per_episode:.0f}s",
                      value=None))
    if target.resolution and info.width:
        want = str(target.resolution).lower().replace("×", "x")
        got = f"{info.width}x{info.height}"
        ok = got == want
        out.append(_m("spec.resolution", Verdict.PASS if ok else Verdict.FAIL,
                      reason="" if ok else f"分辨率 {got}，要求 {want}"))
    if target.aspect_ratio and info.aspect:
        ok = info.aspect == str(target.aspect_ratio).replace("：", ":")
        out.append(_m("spec.aspect", Verdict.PASS if ok else Verdict.FAIL,
                      reason="" if ok else f"画幅 {info.aspect}，要求 {target.aspect_ratio}"))
    out.append(_m("spec.audio_track",
                  Verdict.PASS if info.has_audio else Verdict.FAIL,
                  reason="" if info.has_audio else "没有音轨"))
    return out


# ------------------------------------------------------------ artifact ----
# A black or solid run entirely inside the first/last second is a fade, not a
# defect: claudecode ep2 carried four black frames at 59.83-60.00 s and was
# scoring 0.5 on black_frames, solid_frames *and* join_artifacts for one
# fade-out. Reported, never counted.
EDGE_SECONDS = 1.0

_ARTIFACT_RATES = {
    "black_frames": "artifact.black_frames",
    "solid_frames": "artifact.solid_frames",
    "frozen_runs": "artifact.frozen_runs",
}


def from_artifact(report: dict | None) -> list[Measurement]:
    """artifact_checks: frame-count phenomena become defect rates.

    The module already divides by the frame count and reports `fraction`, so
    these arrive on the RATE scale with no conversion. `flicker` is the
    exception -- it counts events, not frames -- and gets its own denominator.
    """
    if not report:
        return []
    total = dig(report, "meta.n_frames", "video_meta.n_frames", "n_frames")
    duration = dig(report, "meta.duration", "video_meta.duration", "duration")
    findings = dig(report, "findings", "phenomena", default={}) or {}
    out: list[Measurement] = []

    for name, key in _ARTIFACT_RATES.items():
        f = findings.get(name)
        if not isinstance(f, dict):
            continue
        rate = f.get("fraction")
        n = f.get("n_frames")
        if rate is None:
            if n is None or not total:
                out.append(_abstain(key, f"{name} 无可用分母"))
                continue
            rate = float(n) / float(total)
        rate = float(rate)
        exempt = 0
        segs = f.get("segments") or []
        if duration and segs and total:
            for sg in segs:
                a, b = sg.get("start_s"), sg.get("end_s")
                if a is None or b is None:
                    continue
                if float(b) <= EDGE_SECONDS or float(a) >= float(duration) - EDGE_SECONDS:
                    exempt += int(sg.get("n_frames") or 0)
            if exempt:
                rate = max(0.0, (float(n or 0) - exempt) / float(total))
        # The module's `present` is its verdict on whether the phenomenon
        # occurred (a frozen *run* needs consecutive frames; twelve scattered
        # identical frames are not one). Reading the raw fraction over that
        # verdict scored claudecode ep3 0.5 on frozen_runs the module had
        # marked absent. Older reports without the flag fall back to rate > 0.
        module_present = f.get("present") if "present" in f else (rate > 0)
        present = bool(module_present) and rate > 0
        detail = f"影响 {n} 帧（{rate:.1%}）"
        # HARD_RATE, not RATE. These are frames a viewer sees: on the shared
        # statistical band, 5% of a 1440-frame episode -- three seconds of black
        # -- scored a clean 1.0. Any visible occurrence is a defect here.
        out.append(_m(key, Verdict.FAIL if present else Verdict.PASS,
                      Scale.HARD_RATE, rate,
                      reason=f"{name}：{detail}" if present else "",
                      n_frames=total, edge_exempt_frames=exempt,
                      module_present=bool(module_present)))

    # Seams that show. artifact_checks reported this all along and the first
    # version never read it. claudecode ep1's sixth clip came back as six
    # slivers (17, 4, 4, 25, 4, 186 frames): a 4-frame segment is not content,
    # it is a join glitch. Rate is per join, so one bad seam in a 6-clip
    # episode is visible, not diluted by 1440 frames.
    ja = findings.get("join_artifacts")
    sp_n = dig(report, "splices.n_splices")
    if isinstance(ja, dict) and sp_n is not None:
        items = ja.get("join_artifacts") or []
        n = ja.get("n_join_artifacts")
        if n is None:
            n = len(items)
        # A "seam" inside the first/last second is the fade, not a join.
        if duration and items:
            edge = [j for j in items if isinstance(j, dict) and j.get("t_s") is not None
                    and (float(j["t_s"]) <= EDGE_SECONDS
                         or float(j["t_s"]) >= float(duration) - EDGE_SECONDS)]
            n = max(0, int(n) - len(edge))
        if sp_n > 0:
            rate = float(n) / float(sp_n)
            # Statistical band, not HARD_RATE: the denominator is seams, not
            # frames. On the frame-tuned hard band one bad seam in six (17%)
            # scored 0 -- identical to six bad seams. One visible glitch in a
            # six-clip episode is a flaw (0.5), not a collapse.
            out.append(_m("artifact.join_artifacts",
                          Verdict.FAIL if n else Verdict.PASS, Scale.RATE, rate,
                          reason=f"{n}/{sp_n} 个接缝处有可见伪影" if n else "",
                          n_join_artifacts=n, n_splices=sp_n))
        else:
            out.append(_abstain("artifact.join_artifacts", "没有拼接点，无接缝可评"))

    tail = findings.get("tail_freeze")
    if isinstance(tail, dict):
        hit = bool(tail.get("present"))
        held = tail.get("held_seconds") or tail.get("held_frames")
        # A frozen tail is runtime padding: the file reaches its target length
        # without generating anything for it. Same family as retiming, so the
        # same verdict -- scoring it as a mild defect would let padding pay.
        out.append(_m("artifact.tail_freeze",
                      Verdict.VOID if hit else Verdict.PASS,
                      reason=f"结尾冻结 {held}，用静止画面凑时长" if hit else ""))

    # The splice structure itself (`splices.segment_frames`, `splices.splices`)
    # is read straight off the report by run_eval for claims.structure_consistent
    # and the seam-alignment flow. It used to be echoed here as a diagnostic
    # measurement, `provenance.segments_recovered`; deleted 2026-09-22.
    return out


# -------------------------------------------------------------- visual ----
def _identity_consistency(report: dict) -> Measurement:
    """1.2.2: validate face evidence, then express drift over its own noise."""
    key = "visual.identity_consistency"
    not_measured = report.get("not_measured") or {}
    if "character_identity" not in report and "character_identity" in not_measured:
        return _abstain(key, f"人物身份：{not_measured['character_identity']}")
    # Padding/repetition belongs to C and cannot void face evidence.
    # Both statistics must refer to the same recurring identity and encoder.
    inter = dig(report, "character_identity.per_identity_consistency.worst",
                "character_identity.per_identity_consistency.p10", "face_identity.inter_shot.p10")
    intra = dig(report, "character_identity.per_identity_consistency.intra_shot",
                "character_identity.intra_identity.mean")
    r = ratio_from_similarity(inter, intra)
    if r is None:
        why = ("模块未报告同镜头内噪声底" if intra is None else
               "跨镜头值缺失" if inter is None else "噪声底为零，不可估")
        return _abstain(key, f"人物身份：{why}")
    if dig(report, "character_identity.per_identity_consistency.comparison_scope") != "cross_shot":
        return _abstain(key, "旧版人脸统计混入同镜头配对，需要重跑检测")
    return _m(key, Verdict.PASS if r < 1.0 else Verdict.FAIL, Scale.RATIO, r,
              reason=f"人物身份跨镜头不一致是同镜头内噪声的 {r:.1f} 倍" if r >= 1.0 else "",
              inter=inter, intra=intra)


def from_visual(report: dict | None) -> list[Measurement]:
    """Face evidence for 1.2.2; repeated footage for C; deferred scene proxy.

    This adapter emits measurements only. Scope exclusions and score mapping
    belong to rubric/capabilities.py and rubric/node.py respectively.
    """
    if not report:
        return []
    identity = _identity_consistency(report)
    # Counts remain metadata; they cannot bind a detected face to a named role.
    ids = dig(report, "character_identity.identities", default=[]) or []
    if isinstance(ids, list):
        identity.detail.update(
            n_identities=len(ids),
            n_major_identities=sum(1 for i in ids if (i.get("n_faces") or 0) >= 2),
            n_cross_shot_identities=sum(1 for i in ids if i.get("cross_shot")))
    out = [identity]

    # Preserved for existing consumers, uniformly excluded from Q.
    scene = dig(report, "scene_reference_check")
    if isinstance(scene, dict) and scene.get("n_shots"):
        n = int(scene["n_shots"])
        ok = int(scene.get("n_shots_matching_own_ref") or 0)
        out.append(_m("visual.scene_reference",
                      Verdict.PASS if ok >= n else Verdict.FAIL,
                      Scale.RATE, (n - ok) / float(n),
                      reason="" if ok >= n else
                      f"{n-ok}/{n} 个镜头与 agent 自己的场景条件图几何上对不上"))

    duration = dig(report, 'meta.duration', 'duration')
    if duration is None:
        duration = max((float(x.get('end') or 0) for x in report.get('shots', [])), default=0)
    repeat = repeat_evidence(report, duration or None)
    if repeat['abstained'] or not duration:
        out.append(_abstain('visual.near_duplicate_shots', repeat['abstained'] or '时长未知'))
    else:
        seconds = repeat['seconds']
        out.append(_m('visual.near_duplicate_shots', Verdict.FAIL if seconds else Verdict.PASS,
                      Scale.RATE, seconds / duration,
                      reason=f'时间轴证明重复 {seconds:.2f}s' if seconds else '',
                      duplicate_seconds=seconds, candidate_pairs=repeat['n_shot_pairs'],
                      evidence_scope='temporal_intervals'))
    return out


# ----------------------------------------------------- 1.2.2 cut evidence --
def from_seam(report: dict | None, charged_spans=None) -> list[Measurement]:
    """seam_checks: one small check on the cuts themselves, `seam.cut_continuity`.

    A cut breaks when it lands on the still head every image-to-video clip
    starts from (`p_head` below alpha among the film's own interior windows),
    when an unusual motion step also has a sustained still head, or when
    it is a cut inside a delivered clip that the plan never had (a hidden
    cut, as strong as the agent's own). The value is the share of broken cuts
    over all cuts seen, on the RATE scale: k broken of n = seams + hidden.

    The regret split (seconds of still after a cut the source clip's slack
    would have let the agent trim, against seconds no trim could reach) rides
    in `detail` for the ledger (`flows.handles_attribution`); it is not a
    second score.

    A single-segment episode has no cut to judge: NOT_APPLICABLE, it leaves
    the denominator. No summary at all is an ABSTAIN.
    """
    if not report:
        return []
    key = "seam.cut_continuity"
    if report.get("status") == "no_seams":
        return [_m(key, Verdict.NOT_APPLICABLE, reason="单段成片，没有拼接点可检")]
    s = report.get("summary") or {}
    n_seams = s.get("n_seams")
    if not n_seams:
        return [_abstain(key, "seam 模块无摘要")]
    alpha = float(report.get("alpha", 0.05))
    seams = [x for x in (report.get("seams") or []) if isinstance(x, dict)]
    dead = [x["t_s"] for x in seams if x.get("p_head") is not None and float(x["p_head"]) < alpha]
    pace = [x["t_s"] for x in seams if x.get("p_pace") is not None and float(x["p_pace"]) < alpha
            and float(x.get("excess_dead_s") or 0) > 0]
    broken = sorted({*dead, *pace})
    # A still head already wholly removed from genuine footage cannot also
    # count as a bad cut. Independent pace/hidden cuts keep their own evidence.
    covered = []
    for row in seams:
        t, length = row.get("t_s"), row.get("dead_s")
        if t in broken and length and float(length) > 0 and charged_spans:
            t, length = float(t), float(length)
            overlap = [(max(t, a), min(t + length, b)) for a, b in charged_spans
                       if a < t + length and b > t]
            if union_seconds(overlap) >= length - 1e-6:
                covered.append(t)
    broken = [t for t in broken if t not in covered]
    dead = [t for t in dead if t not in covered]
    pace = [t for t in pace if t not in covered]
    hidden = [h.get("t_s") for h in (report.get("hidden_cuts") or []) if isinstance(h, dict)]
    # hidden cuts need a visible seam as reference; without one they are unknown, not zero
    if not s.get("n_visible_seams"):
        hidden = []
    hidden = sorted(set(hidden) - {x.get("t_s") for x in seams})
    k, n = len(broken) + len(hidden), int(n_seams) + len(hidden)
    detail = dict(k=k, n=n, n_seams=int(n_seams), dead_heads=len(dead), pace_breaks=len(pace),
                  hidden_cuts=len(hidden), broken_at=broken[:20], hidden_at=hidden[:20],
                  total_dead_s=s.get("total_dead_s"), calibration=report.get("calibration"))
    detail["already_charged_in_C"] = covered
    rg = report.get("regret")
    if isinstance(rg, dict) and rg.get("status") == "ok":
        detail.update(shipped_s=float(rg.get("shipped_s") or 0), trimmable_s=float(rg.get("trimmable_s") or 0),
                      material_s=float(rg.get("untrimmable_s") or 0), slow_clips=rg.get("slow_clips"))
    parts = []
    if dead:
        parts.append(f"{len(dead)} 个切在静止片头上")
    if pace:
        parts.append(f"{len(pace)} 个跨切点节奏断裂")
    if hidden:
        parts.append(f"片段内部另有 {len(hidden)} 处与 agent 自己的切点同强度的切换")
    reason = f"{n_seams} 个拼接点：" + "，".join(parts) if parts else ""
    return [_m(key, Verdict.PASS if k == 0 else Verdict.FAIL, Scale.RATE, k / n, reason=reason, **detail)]


# ------------------------------------------------ 1.2.3 physical evidence --
# Evidence gates, separate from the score bands in rubric/node.py.
MIN_CONTACT_FRAME_RATE = 0.2
MIN_RESOLVED_PAIRS = 10


def from_physical(report: dict | None) -> list[Measurement]:
    """Resolved nonattachment contacts for 1.2.3.a; deferred size proxy."""
    if not report:
        return []
    out: list[Measurement] = []
    spans = _shot_spans(report)

    pen = dig(report, "penetration", default={}) or {}
    evaluated = dig(pen, "coverage.pairs_evaluated")
    flagged = dig(pen, "flagged.n")
    contact_rate = dig(pen, "coverage.frame_contact_rate")
    judged = dig(pen, "coverage.pairs_judged")
    unresolved = dig(pen, "unresolvable.n")
    resolved = (int(judged) - int(unresolved)) if (judged is not None and unresolved is not None) else None
    if not evaluated:
        out.append(_abstain("physical.penetration",
                            "没有互相接触的实例对，穿模检查是空的而不是干净的"))
    elif contact_rate is not None and float(contact_rate) < MIN_CONTACT_FRAME_RATE:
        out.append(_abstain("physical.penetration",
                            f"只有 {float(contact_rate):.0%} 的帧有接触对（模块阈值 20%）："
                            "指示性读数，不作为案例分"))
    elif dig(pen, "flagged.by_reason.no_depth_step", default=0):
        out.append(_abstain("physical.penetration", "旧版检测把无深度差当成穿模，需要重跑检测"))
    elif resolved is None or flagged is None:
        out.append(_abstain("physical.penetration", "缺少有效接触对或穿模计数，不能把未测量当成干净"))
    elif not (0 <= int(flagged) <= resolved <= int(judged) <= int(evaluated)):
        out.append(_abstain("physical.penetration", "接触、可解析与穿模计数不一致"))
    elif resolved < MIN_RESOLVED_PAIRS:
        out.append(_abstain("physical.penetration",
                            f"{judged} 组接触里只有 {resolved} 组深度可解析（{unresolved} 组两侧深度"
                            "过近），单目深度在这段素材上给不出结论"))
    else:
        # Attachments and unresolved boundaries are never clean-contact credit.
        rate = float(flagged) / resolved
        # Which shots the flags fell in, with the module's own shot times, so
        # flows.plan_explains_physical can read them against the plan's beats.
        flagged_shots = []
        for sid, row in sorted((dig(pen, "per_shot", default={}) or {}).items(),
                               key=lambda kv: int(kv[0])):
            if isinstance(row, dict) and (row.get("n_flagged") or 0) > 0:
                st, en = spans.get(int(sid), (None, None))
                flagged_shots.append({"shot": int(sid), "start": st, "end": en,
                                      "n_flagged": int(row["n_flagged"]),
                                      "n_judged": row.get("n_judged")})
        out.append(_m("physical.penetration",
                      Verdict.PASS if not flagged else Verdict.FAIL,
                      Scale.RATE, rate,
                      reason="" if not flagged else
                      f"{flagged}/{resolved} 组可解析接触被判为穿模或互相遮挡",
                      **{"by_reason": dig(pen, "flagged.by_reason", default={}),
                         "flagged_shots": flagged_shots,
                         "pairs_judged": judged, "pairs_unresolvable": unresolved,
                         "pairs_resolved": resolved, "pairs_evaluated": evaluated,
                         "frame_contact_rate": contact_rate}))

    # The detector pairs CLASSES, not instances. A different chair in a new
    # shot is a legitimate edit; the ratio alone cannot prove size drift of
    # the same object. Keep its readings for inspection, never score them.
    drift = dig(report, "scale_drift", default={}) or {}
    out.append(_m("visual.scale_consistency", Verdict.ABSTAIN,
                  reason="尺寸检测只关联物体类别，未确认跨镜头为同一实例",
                  measurement_scope="class_pairs", requires="cross_shot_instance_correspondence",
                  drift_over_noise=dig(drift, "drift_over_noise.value"),
                  coverage=drift.get("coverage"), pairs=drift.get("pairs", []),
                  skipped=drift.get("skipped")))
    return out


# --------------------------------------------------- 1.2.3 audio evidence --
DYNAMIC_RANGE_DB = 12.0     # av_consistency's floor for a healthy dialogue mix


def _audio_defects(clean: dict) -> list[Measurement]:
    """1.2.3.b clipping and dynamic range; silence is handled separately."""
    out: list[Measurement] = []
    runs = dig(clean, "clipped_runs")
    if runs is not None:
        n = len(runs) if isinstance(runs, list) else int(runs)
        # The detector counts flat extrema, not merely near-full-scale peaks.
        out.append(_m("av.clipping", Verdict.PASS if n == 0 else Verdict.FAIL,
                      reason="" if n == 0 else f"{n} 段连续削波"))

    pause = dig(clean, "pause_fraction")
    dr = dig(clean, "dynamic_range_db")
    if dr is None or pause is None:
        out.append(_abstain("av.dynamic_range", "动态范围或停顿占比缺失"))
    elif float(pause) < 0.05:
        # Steady-state audio collapses the 10th percentile onto the median, so
        # the number stops being a proxy for noise floor at all.
        out.append(_abstain("av.dynamic_range",
                            "停顿太少，底噪不可估（floor_not_estimable）"))
    else:
        # Graded, not boolean: 11 dB against a 12 dB floor is a near miss, 7 dB
        # is a flattened mix. The shortfall fraction rides the RATE band.
        short = max(0.0, (DYNAMIC_RANGE_DB - float(dr)) / DYNAMIC_RANGE_DB)
        out.append(_m("av.dynamic_range", Verdict.PASS if short == 0.0 else Verdict.FAIL,
                      Scale.RATE, short,
                      reason="" if short == 0.0 else
                      f"动态范围仅 {float(dr):.1f}dB（下限 {DYNAMIC_RANGE_DB:.0f}dB）",
                      dynamic_range_db=float(dr)))

    return out


def from_av(report: dict | None, expected_speakers: int | None = None) -> list[Measurement]:
    """Gate silent tracks, measure audio defects, retain deferred voice proxies.

    Long silence comes from from_silence; subtitle/ASR correspondence comes
    from expectations.subtitles_aligned. Neither needs the voice proxies.
    """
    if not report:
        return []
    out: list[Measurement] = []
    clean = dig(report, "cleanliness", default={}) or {}

    # A stream that carries nothing is not an audio track. All three openmontage
    # episodes had rms=0, peak=0, signal_db=-200 and were scoring PASS on
    # `spec.audio_track` ("a stream exists") and PASS on O5 ("zero clipping
    # runs" -- on silence). Zero defects on nothing is not a clean result; it
    # excludes dependent audio numbers and fails the track itself. Worst-wins merging
    # lets this FAIL override spec_checks' stream-presence PASS.
    rms, peak = dig(clean, "rms"), dig(clean, "peak")
    sig = dig(clean, "signal_db")
    silent = ((rms is not None and float(rms) < 1e-4)
              or (peak is not None and float(peak) < 1e-3)
              or (sig is not None and float(sig) < -90))
    no_audio = dig(report, "audio_stream.present") is False
    if silent or no_audio:
        why = "视频没有音轨" if no_audio else "音轨存在但全程静音（rms=0）"
        out.append(_m("spec.audio_track", Verdict.FAIL, reason=why))
        for k in ("av.clipping", "av.dynamic_range", "av.voice_face_match",
                  "av.long_silence"):
            out.append(dependent(k, "spec.audio_track", f"{why}，仅在音轨合规扣分"))
        # A track with no voices says nothing about voice consistency.
        # The missing track has already been charged in specification.
        out.append(dependent("av.speaker_consistency", "spec.audio_track", f"{why}，无说话人可比"))
        return out

    # Voice-to-face attribution. presence.py judges each speech window that
    # has faces on screen: `plausible` when the voice type matches a visible
    # face, `implausible` when it matches none, `unknown` when unresolvable,
    # `offscreen_voice` when nobody is on screen (narration, judged by the
    # brief's rule elsewhere). Only the resolved windows form the denominator.
    wins = dig(report, "presence.windows", default=[]) or []
    counts = {"plausible": 0, "implausible": 0, "unknown": 0, "offscreen_voice": 0}
    for w in wins:
        for vd in (w.get("verdicts") or []):
            if is_speech_label(vd.get("sound")) and vd.get("verdict") in counts:
                counts[vd["verdict"]] += 1
    # A compatible voice type somewhere in the picture is not a binding to
    # the speaking face. Reverse shots and offscreen dialogue also make the
    # opposite verdict unreliable. Preserve the proxy without scoring it.
    out.append(_m("av.voice_face_match", Verdict.ABSTAIN,
                  reason="仅有声线类型与可见人脸的粗匹配，未定位实际说话者",
                  measurement_scope="voice_type_presence",
                  requires="active_speaker_attribution", **counts))

    # Active audio checks. Voice proxies above/below do not enter Q.
    out.extend(_audio_defects(clean))

    # Cast size is not the number of speakers in this episode. Neither few
    # clusters nor many clusters establishes whether the same role keeps its
    # voice. No role/window bindings or cross-episode identities are emitted
    # by the current detector, so counts remain diagnostic evidence only.
    out.append(_m("av.speaker_consistency", Verdict.ABSTAIN,
                  reason="只有声音簇数量，缺少角色与声音、跨片段说话人的对应证据",
                  n_clusters=dig(clean, "n_speaker_clusters"),
                  n_windows=dig(clean, "n_speech_windows"),
                  expected_speakers=expected_speakers,
                  requires="role_voice_correspondence"))

    return out


LONG_SILENCE_S = 5.0


def from_silence(runs: list[dict] | None, duration: float | None) -> Measurement:
    """av.long_silence from ffmpeg silencedetect runs (evidence bus).

    Keep detected low-level intervals lasting at least five seconds and
    divide their total duration by the track duration. The evidence bus
    supplies the amplitude threshold; this adapter applies the duration gate.
    """
    key = "av.long_silence"
    if runs is None:
        return _abstain(key, "ffmpeg 不可用，静音段未测")
    if not duration:
        return _abstain(key, "时长未知")
    long = [r for r in runs if float(r.get("duration") or 0.0) >= LONG_SILENCE_S]
    if not long:
        return _m(key, Verdict.PASS, Scale.RATE, 0.0, n_runs=len(runs))
    covered = sum(float(r["duration"]) for r in long)
    rate = min(1.0, covered / float(duration))
    return _m(key, Verdict.FAIL, Scale.RATE, rate,
              reason=f"{len(long)} 段 ≥{LONG_SILENCE_S:.0f}s 的完全静音，合计 {covered:.1f}s"
                     f"（最长 {max(float(r['duration']) for r in long):.1f}s，"
                     f"从 {min(float(r['start']) for r in long):.0f}s 起）",
              n_long=len(long), seconds=round(covered, 2),
              runs=[{k: r[k] for k in ("start", "end", "duration")} for r in long][:10])


# --------------------------------------------------------------- trace ----
def from_facts(facts, target=None) -> list[Measurement]:
    """TrajectoryFacts -> Measurements.

    Reads the checklist, not a trajectory format. Whether the facts came from a
    regex over `trace.jsonl`, a cost ledger, or a language model reading prose
    is not this function's concern -- which is the point of having a checklist.
    A None field means the backend could not observe it, and becomes ABSTAIN.
    """
    out: list[Measurement] = []
    why = "; ".join(facts.notes) if facts.notes else "轨迹中不可观测"

    e = facts.errors

    # 2.1.a, scored: attempts that never succeeded. A failure the agent retried
    # into a success is resilience and costs nothing -- across `data/` only 21
    # of 536 agent-caused failures are attempts that stuck, so charging all of
    # them scored the retry loop rather than the agent. The raw call-level
    # rate (`trace.error_rate`) was reported beside this as a diagnostic until
    # 2026-09-22 and is gone with its three checklist fields: it counted
    # retries as failures and described the instrument as much as the agent.
    if e.n_unrecovered is None or not e.n_attempts:
        unknown = facts.reliability_audit.get("unknown") or []
        out.append(_m("trace.unrecovered_failures", Verdict.ABSTAIN,
                      reason="生产任务身份、终态或责任归属不完整" if unknown else "没有可核验的生产任务终态",
                      evidence=facts.reliability_audit))
    else:
        rec = e.n_recovered or 0
        out.append(_m("trace.unrecovered_failures",
                      Verdict.PASS if not e.n_unrecovered else Verdict.FAIL,
                      Scale.RATE, e.n_unrecovered / float(e.n_attempts),
                      reason="" if not e.n_unrecovered else
                      f"{e.n_unrecovered}/{e.n_attempts} 次尝试最终没有成功"
                      + (f"（另有 {rec} 次失败后重试成功，不计分）" if rec else ""),
                      recovered=rec,
                      # Keep uncertainty context separate from the point score.
                      k=int(e.n_unrecovered), n=int(e.n_attempts),
                      jeffreys_error_mean=(e.n_unrecovered + 0.5) / (e.n_attempts + 1),
                      evidence=facts.reliability_audit))

    # Declared intent vs the brief. The pixel channel cannot supply this: a run
    # that *asks* for 16:9 against a 9:16 brief is wrong before a file exists,
    # and vimax does exactly that. Pure string comparison over parameters the
    # agent itself put on the wire.
    out.extend(_requested_vs_target(facts.requested, target))

    # Conditioning edges: did the agent feed earlier assets back in as reference.
    # Absent from every HTTP trace by construction -- a request record carries
    # `body_bytes`, not the body -- so this abstains rather than reading the
    # silence as "no references were used".
    if facts.conditioning is None:
        out.append(_abstain("provenance.conditioning_edges",
                            "轨迹未记录条件资产的引用关系"))
    else:
        n = len(facts.conditioning)
        out.append(_m("provenance.conditioning_edges",
                      Verdict.PASS if n else Verdict.FAIL,
                      reason="" if n else "轨迹显示生成时未使用任何一致性参考资产",
                      n_edges=n))
    return out


# Content checks in the delivery dimension. When no episode exists these are
# not unmeasurable -- they are failed, and must say so rather than abstain.
_NO_DELIVERY_VOIDS = (
    "spec.duration", "spec.resolution", "spec.aspect", "spec.audio_track",
    "spec.retiming", "artifact.tail_freeze", "provenance.duration_provenance",
)


def from_delivery(n_delivered: int, target, delivered_seconds: float | None,
                  facts, disagreement: list[str] | None = None, *, root=None) -> list[Measurement]:
    """Trajectory x delivery: the closed-loop checks.

    Everything here compares two channels that the first version extracted
    and never put side by side -- what the agent said, what it did, what it
    shipped. Each contradiction is scored, because a contradiction between
    channels is evidence about the run, not noise in the measurement.
    """
    out: list[Measurement] = []
    want = getattr(target, "episodes", None)

    # ---- episode count: the first gate -----------------------------------
    if want:
        short = max(0.0, 1.0 - n_delivered / float(want))
        out.append(_m("o2.episode_count",
                      Verdict.PASS if n_delivered >= want else Verdict.FAIL,
                      Scale.RATE, short,
                      reason="" if n_delivered >= want else
                      f"要求 {want} 集，定位到 {n_delivered} 集",
                      delivered=n_delivered, required=want))
    else:
        out.append(_abstain("o2.episode_count", "任务未给出集数"))

    # ---- no delivery at all: determinate failure, not missing evidence ---
    if n_delivered == 0:
        for k in _NO_DELIVERY_VOIDS:
            out.append(_m(k, Verdict.VOID, reason="没有交付任何成片"))

    # Evaluate the final extracted agent claims, not the locator's mixed
    # harness/declaration notes. LLM-added paths are checked here too.
    from stage4_checks.process.honesty import check_deliverables
    from stage4_checks.process.honesty import check_status
    out.append(check_status(facts, n_delivered, want))
    out.append(check_deliverables(facts, root))

    # ---- generated seconds vs delivered seconds ----------------------------
    gen = facts.requested.generated_seconds
    if n_delivered == 0:
        pass                                   # already VOID above
    elif gen is None or not delivered_seconds:
        why = "轨迹未记录生成时长" if gen is None else "交付时长未知"
        out.append(_abstain("provenance.duration_provenance", why))
    else:
        out.append(_m("provenance.duration_provenance", Verdict.ABSTAIN,
                      reason="请求或自述秒数不证明实际成功产出；交付与源素材的差距由逐集实测账核验",
                      source=facts.requested.generated_seconds_source or "unknown",
                      reported_seconds=gen, delivered_seconds=delivered_seconds))
    return out


MIN_SEGMENT_FRAMES = 24     # about one second; anything shorter is a sliver


def _norm_aspect(s) -> str | None:
    if not s:
        return None
    t = str(s).lower().replace("：", ":").replace("x", ":").replace("×", ":")
    if ":" not in t:
        return None
    a, _, b = t.partition(":")
    try:
        from math import gcd
        ai, bi = int(float(a)), int(float(b))
        g = gcd(ai, bi) or 1
        return f"{ai // g}:{bi // g}"
    except (TypeError, ValueError):
        return None


def _short_side(s) -> int | None:
    """720p / 1080x1920 / 1920 -> the short side in pixels, when derivable."""
    if not s:
        return None
    t = str(s).lower().replace("×", "x")
    m = _re.fullmatch(r"(\d+)p", t)
    if m:
        return int(m.group(1))
    m = _re.fullmatch(r"(\d+)\s*x\s*(\d+)", t)
    if m:
        return min(int(m.group(1)), int(m.group(2)))
    return None


def _requested_vs_target(req, target) -> list[Measurement]:
    """o2.requested_params: what the agent asked the generator for vs the brief."""
    key = "o2.requested_params"
    if target is None:
        return [_abstain(key, "任务未给出规格目标")]
    want_a = _norm_aspect(getattr(target, "aspect_ratio", None))
    want_px = _short_side(getattr(target, "resolution", None))
    if want_a is None and want_px is None:
        return [_abstain(key, "任务未给出画幅或分辨率目标")]
    got_a = [x for x in (_norm_aspect(v) for v in (req.aspect_ratios or [])) if x]
    got_px = [x for x in (_short_side(v) for v in (req.resolutions or [])) if x]
    if not got_a and not got_px:
        return [_abstain(key, "轨迹未记录生成请求的画幅或分辨率")]

    bad: list[str] = []
    if want_a and got_a:
        wrong = sorted({a for a in got_a if a != want_a})
        if wrong:
            bad.append(f"请求画幅 {', '.join(wrong)}，任务要求 {want_a}")
    if want_px and got_px:
        low = min(got_px)
        if low < want_px:
            bad.append(f"请求短边最低 {low}px，任务要求 {want_px}px")
    return [_m(key, Verdict.FAIL if bad else Verdict.PASS,
               reason="；".join(bad), requested_aspects=got_a,
               requested_short_sides=sorted(set(got_px)))]
