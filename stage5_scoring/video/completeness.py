"""1.1 完成度 Completeness -- C, a ledger of seconds, not a rubric item.

Per episode, the requested interval R = [0, T] and the delivered footage D
are sets on a timeline. The genuine part of D is what is left after the
deductions the modules can prove:

    clean = d - union(frozen, black/solid, repeats, seams, no-picture)
    g = clean * source_picture / delivered_picture   (only for proved slowdown)

    e_i = 1 - |D ∩ R| / |D ∪ R| = 1 - min(g, T) / (d + T - min(g, T))
    C_i = min(band(e_i), g/d) - a_i                    missing episode: C_i = 0
    C   = mean_i C_i

The shortfall e is the ordinary relative duration error whenever the delivery
is honest -- 57 s against a 60 s ask is e = 0.05 -- and it is more than that
when it is not: the deductions are inside it, so 40 s padded to 59 s reads
e = 0.49 rather than the 0.017 the container claims. `band` then reads it as
three states, on spec / short / off spec, at 5 % and 20 % (see `band`).

`a_i` is the assembly point: 0 when this episode exists as a cut-together
file, `assembly_deduction` (0.1) when only its material does. Two separate
questions are being asked -- how much real footage is there, and was it made
into an episode -- so they are two separate terms, and the second is small
because it is the smaller of the two failures. An agent that generated the
footage and stopped before the final concatenation has done most of the work;
an agent that concatenated six copies of one clip has not, and no amount of
assembling earns that back. A flat subtraction keeps that ordering: it never
lifts a padded delivery, and it never reduces assembly to a coin flip worth
half the layer.

The ordering this produces is the one a grader wants and none of it is a rule:
an honest 40 s for 60 s scores 0.42, the same 40 s padded to 59 s scores 0.32,
a 10 s clip looped six times scores 0.06, 57 s of real footage scores 1.0, and
the same 57 s left as loose scenes scores 0.90.

Nothing here is user input. Every quantity comes from a module report the
orchestrator already ran, or from the probe; a module that did not run just
contributes no deduction, and the ledger says so.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from dataclasses import asdict
import math
from stage3_evidence.intervals import union_seconds
from stage3_evidence.intervals import union_intervals
from stage3_evidence.intervals import repeat_evidence

EDGE_SECONDS = 1.0        # a fade inside the first/last second is not padding


@dataclass
class EpisodeLedger:
    video: str
    episode: int | None = None
    target_s: float | None = None
    delivered_s: float | None = None
    # False when this file is material -- a scene, a shot, a render -- rather
    # than an episode cut together from it. Set by the locator, which already
    # decides this to choose what to hand the modules; the ledger only reads it.
    assembled: bool = True
    # Per-detector seconds, for the report. These may overlap each other.
    frozen_s: float = 0.0
    black_s: float = 0.0
    duplicate_s: float = 0.0
    seam_s: float = 0.0
    no_picture_s: float = 0.0
    retime_excess_s: float = 0.0
    # How many times each detector fired, and over what denominator. The
    # seconds say how much of the file is not footage; these say how broken
    # it is -- a hundred scattered black frames and one long fade cost the
    # same seconds and are not the same defect. Nothing else scores them now
    # that 1.2.3.a is physical-only, so the ledger has to carry them.
    counts: dict = field(default_factory=dict)
    # The intervals behind the pixel figures above, so the deduction can be
    # their union. Empty when a module did not run.
    spans: list[tuple[float, float]] = field(default_factory=list)
    sources: dict = field(default_factory=dict)     # deduction -> module that proved it
    notes: list[str] = field(default_factory=list)
    retime_picture_s: float | None = None
    source_comparison: dict = field(default_factory=dict)
    evidence_gaps: list[str] = field(default_factory=list)

    @property
    def deductions_s(self) -> float:
        """Union of the pixel spans, plus retiming, never past what was delivered."""
        # Unlocated legacy counts do not prove disjoint intervals. Retain the
        # strongest lower bound instead of adding unknown overlaps together.
        pixel = max(union_seconds(self.spans, self.delivered_s), self.frozen_s,
                    self.black_s, self.duplicate_s, self.seam_s, self.no_picture_s)
        # Pixel intervals are on the delivery clock. A uniform slowdown
        # expands defective frames too; charge retiming only on the clean
        # picture left after the interval union, not on those same frames.
        if self.retime_picture_s and self.retime_picture_s > 0:
            clean = max(0.0, (self.delivered_s or 0.0) - pixel)
            retime = min(clean, self.retime_picture_s) * min(
                1.0, self.retime_excess_s / self.retime_picture_s)
        else:
            # Legacy callers have no temporal alignment. Both are lower
            # bounds on the defect, not proof of disjoint seconds.
            retime = max(0.0, self.retime_excess_s - pixel)
        total = pixel + retime
        return min(total, max(0.0, self.delivered_s)) if self.delivered_s is not None else total


@dataclass
class EpisodeScore:
    video: str
    episode: int | None
    target_s: float | None
    delivered_s: float | None
    genuine_s: float | None
    score: float | None
    within_tolerance: bool = False
    deductions: dict = field(default_factory=dict)
    capped_by_generation: bool = False
    notes: list[str] = field(default_factory=list)
    shortfall: float | None = None       # e = 1 - Jaccard
    pass_band: float | None = None       # the e this episode had to stay under
    band: str = ""                       # on_spec / short / off_spec
    counts: dict = field(default_factory=dict)   # how often each defect fired
    source_comparison: dict = field(default_factory=dict)
    evidence_gaps: list[str] = field(default_factory=list)
    assembled: bool = True
    assembly_deduction: float = 0.0      # what the assembly point actually cost


@dataclass
class CompletenessResult:
    score: float | None                  # C; None when a delivery/target cannot be timed
    n_required: int | None
    n_found: int
    episodes: list[EpisodeScore] = field(default_factory=list)
    delivered_s: float = 0.0
    genuine_s: float = 0.0
    deductions_s: float = 0.0
    tolerance_s: float = 0.0
    tolerance_source: str = ""
    generated_cap_s: float | None = None
    generation: dict = field(default_factory=dict)   # generated vs delivered
    assembly: dict = field(default_factory=dict)     # the assembly point, per episode
    reason: str = ""
    is_upper_bound: bool = False
    evidence_gaps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ----------------------------------------------------------- the ledger --
def _f(x) -> float | None:
    try:
        v = None if x is None else float(x)
        return v if v is None or math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _segment_spans(finding: dict | None, duration: float | None, *, exempt_edges=True) -> list[tuple[float, float]]:
    """The same segments as intervals, for the union."""
    out: list[tuple[float, float]] = []
    if not isinstance(finding, dict) or finding.get("present") is False:
        return out
    for sg in finding.get("segments") or []:
        a, b = _f(sg.get("start_s")), _f(sg.get("end_s"))
        if a is None or b is None or b <= a:
            continue
        if exempt_edges and duration and (b <= EDGE_SECONDS or a >= duration - EDGE_SECONDS):
            continue
        out.append((a, b))
    return union_intervals(out, duration)


def ledger_from_reports(video: str, episode: int | None, target_s: float | None,
                        probe_duration: float | None,
                        spec: dict | None = None, artifact: dict | None = None,
                        visual: dict | None = None, *,
                        assembled: bool = True,
                        probe_picture_s: float | None = None) -> EpisodeLedger:
    """Read one episode's seconds off the reports the orchestrator already has.

    `probe_picture_s` is the picture stream's duration from the evaluator's
    own ffprobe pass. It is used only when neither the spec nor the artifact
    report supplies one, so the container-longer-than-picture deduction does
    not depend on a detector module being installed.
    """
    led = EpisodeLedger(video=video, episode=episode, target_s=target_s,
                        assembled=assembled)

    # Delivered seconds: the spec module's probe, else ours.
    d = None
    if isinstance(spec, dict):
        d = _f((spec.get("meta") or {}).get("duration"))
        if d is None:
            d = _f(((spec.get("checks") or {}).get("duration") or {}).get("duration"))
    if d is None:
        d = _f(probe_duration)
    if d is None and isinstance(artifact, dict):
        d = _f((artifact.get("meta") or {}).get("duration"))
    led.delivered_s = d
    picture_s = _f((spec.get("meta") or {}).get("video_duration")) if isinstance(spec, dict) else None

    # Frozen / black / solid seconds from the artifact module.
    if isinstance(artifact, dict):
        f = artifact.get("findings") or artifact.get("phenomena") or {}
        if any(k not in f for k in ("tail_freeze", "frozen_runs", "black_frames", "solid_frames", "join_artifacts")):
            led.evidence_gaps.append("artifact_incomplete")
        dur = _f((artifact.get("meta") or {}).get("duration")) or d
        if picture_s is None:
            picture_s = dur
        tail = f.get("tail_freeze") or {}
        held = _f(tail.get("held_seconds")) if isinstance(tail, dict) and tail.get("present") else None
        frozen = _segment_spans(f.get("frozen_runs"), dur, exempt_edges=False)
        black = (_segment_spans(f.get("black_frames"), dur)
                 + _segment_spans(f.get("solid_frames"), dur))
        # A held tail has no segment list of its own; place it at the end.
        if held and picture_s:
            frozen.append((max(0.0, picture_s - held), picture_s))
        led.frozen_s = union_seconds(frozen, d)
        led.black_s = union_seconds(black, d)
        led.spans += union_intervals(frozen + black, d)
        led.sources.update({"frozen_s": "artifact", "black_s": "artifact"})

        # Container longer than the picture. `d` is the container duration --
        # what a player's scrubber shows -- and an audio track that outlasts
        # the last frame carries the file past it. Those seconds run, but
        # nothing new is in them, so they are not footage. Measured on one
        # real delivery: a 60.04 s picture inside an 82.79 s container, the
        # extra 22.75 s carried by the narration. The artifact module reports
        # the picture duration (n_frames / fps) in its own meta, so the ledger
        # already holds both numbers.
        if dur is not None and d is not None and d - dur > EDGE_SECONDS:
            led.no_picture_s = d - dur
            led.spans.append((dur, d))
            led.sources["no_picture_s"] = "artifact"
            led.notes.append(f"容器 {d:.2f}s 比画面 {dur:.2f}s 长 "
                             f"{d - dur:.2f}s：多出的部分没有画面")

        # Seams. A join artifact is a dark or flat frame at a splice, one or
        # two frames wide -- too short to form a black-frame segment, which is
        # why it has its own detector. Its frames are charged like any other
        # non-footage frames; its *rate over splices* is the part that says
        # how badly the episode was assembled, and that is a count, not
        # seconds, so the counts block carries it.
        ja, sp = f.get("join_artifacts"), artifact.get("splices") or {}
        if isinstance(ja, dict):
            fps = _f((artifact.get("meta") or {}).get("fps"))
            width = float(ja.get("window_frames") or 1)
            items = [j for j in (ja.get("join_artifacts") or []) if isinstance(j, dict)]
            seam_spans = []
            for j in items:
                t = _f(j.get("t_s"))
                if t is None or not fps or fps <= 0:
                    continue
                half = (width / fps) / 2.0
                if dur and (t + half <= EDGE_SECONDS or t - half >= dur - EDGE_SECONDS):
                    continue                       # the fade, not a join
                seam_spans.append((max(0.0, t - half), t + half))
            led.seam_s = union_seconds(seam_spans, d)
            led.spans += union_intervals(seam_spans, d)
            n_sp = sp.get("n_splices") if isinstance(sp, dict) else None
            led.counts["join_artifacts"] = {
                "n": int(ja.get("n_join_artifacts") or len(items)),
                "n_splices": int(n_sp) if n_sp is not None else None,
                "issues": sorted({s for j in items for s in (j.get("issues") or [])}),
            }
            if seam_spans:
                led.sources["seam_s"] = "artifact"
        for name in ("frozen_runs", "black_frames", "solid_frames"):
            fi = f.get(name)
            if isinstance(fi, dict):
                led.counts[name] = {
                    "n_frames": fi.get("n_frames"),
                    "n_segments": len(fi.get("segments") or []),
                    "fraction": fi.get("fraction"),
                    "present": bool(fi.get("present")) if "present" in fi else None,
                }
    else:
        led.notes.append("artifact 未运行：冻帧/黑场/接缝未扣")
        led.evidence_gaps.append("artifact_unmeasured")

    # Stream metadata is sufficient for this deduction even without a pixel
    # module; otherwise disabling artifact would hide the audio overhang.
    picture_source = "video_stream"
    if picture_s is None and _f(probe_picture_s) is not None:
        picture_s, picture_source = _f(probe_picture_s), "ffprobe"
    if picture_s is not None and d is not None and d - picture_s > EDGE_SECONDS:
        led.no_picture_s = d - picture_s
        led.spans.append((picture_s, d))
        led.sources.setdefault("no_picture_s", picture_source)
        if picture_source == "ffprobe":
            led.notes.append(f"容器 {d:.2f}s 比画面流 {picture_s:.2f}s 长 "
                             f"{d - picture_s:.2f}s：多出的部分没有画面（ffprobe 元数据）")

    # Similar shot pairs are candidates, never proof of whole-shot reuse.
    repeat = repeat_evidence(visual, d)
    led.duplicate_s = repeat['seconds']
    led.spans += repeat['spans']
    led.sources['duplicate_s'] = 'visual.timeline'
    led.counts['repeats'] = {
        'n_shot_pairs': repeat['n_shot_pairs'],
        'n_shots': len((visual or {}).get('shots') or []),
        'n_runs': repeat['n_runs'],
        'scan': 'abstained' if repeat['abstained'] else 'ran',
        'shot_pairs_deducted': False,
    }
    if repeat['abstained']:
        led.evidence_gaps.append('timeline_repeats_unmeasured')
        led.notes.append('时间轴重复检测缺测：' + repeat['abstained'])
    if repeat['n_shot_pairs']:
        led.notes.append('近似镜头仅作候选；重复秒数以时间轴匹配区间为准')

    # Retime excess: delivered seconds beyond what the source clips hold.
    if isinstance(spec, dict):
        rt = (spec.get("checks") or {}).get("retiming") or {}
        led.source_comparison = {
            "source_files": list(spec.get("source_clips") or []),
            "source_video_s": _f(rt.get("source_duration")),
            "delivered_video_s": picture_s,
            "frame_count_preserved": rt.get("frame_count_preserved"),
            "status": "compared" if rt and not rt.get("undetermined") else "unmeasured",
            "correspondence_verified": rt.get("correspondence_verified") is True,
        }
        source_meta = spec.get("source_meta") or []
        unique_sources = {s.get("path"): s for s in source_meta if s.get("path")}
        durations = [_f(s.get("video_duration")) for s in unique_sources.values()]
        if durations and all(x is not None and x > 0 for x in durations):
            unique_s = sum(durations)
            led.source_comparison.update(
                unique_source_video_s=round(unique_s, 3),
                n_unique_sources=len(unique_sources),
                n_source_uses=len(source_meta),
                # This is a comparison, not a deduction: a manifest or file
                # name alone does not prove which frames survived the edit.
                picture_minus_unique_source_s=(round(picture_s - unique_s, 3)
                                               if picture_s is not None else None))
        # Equal frame counts are timing evidence, not content identity. This
        # metadata-only comparison does not certify full source provenance.
        if rt.get("correspondence_verified") is not True:
            led.evidence_gaps.append("source_correspondence_unverified")
        if (isinstance(rt, dict) and not rt.get("undetermined") and rt.get("ok") is False
                and rt.get("correspondence_verified") is True):
            src, fin = _f(rt.get("source_duration")), _f(rt.get("final_duration"))
            # Old cached spec reports used container time. The picture clock
            # prevents an audio overhang from being mistaken for slowdown.
            if fin is not None and picture_s is not None:
                fin = min(fin, picture_s)
            if src is not None and fin is not None and fin > src:
                led.retime_excess_s = fin - src
                led.retime_picture_s = fin
                led.sources["retime_excess_s"] = "spec"
    else:
        led.evidence_gaps.append("source_correspondence_unverified")
    return led


# ------------------------------------------------------------- the score --
def jaccard(genuine: float, delivered: float, target: float) -> float:
    inter = min(max(genuine, 0.0), target)
    union = delivered + target - inter
    return (inter / union) if union > 0 else 0.0


def band(e: float, *, pass_e: float, fail_e: float, fail_score: float) -> float:
    """The shortfall e = 1 - Jaccard, read as on-spec / short / off-spec.

    For an honest delivery e *is* the relative duration error: 57 s against a
    60 s ask gives J = 0.95, e = 0.05. What it adds over a plain |d - T| / T is
    that padding cannot hide in it -- the deductions are already inside J, so
    40 s stretched to 59 s reads e = 0.49, not e = 0.017.

        e <= pass_e              1.0
        pass_e < e <= fail_e     1.0 down to fail_score
        e >  fail_e              fail_score down to 0 at e = 1

    Off-spec slides rather than dropping to zero, and that is not softness. A
    hard zero past fail_e would score an honest 40 s the same as delivering
    nothing, and once both are zero, padding is a free bet: stretch to 59 s and
    any second the repeat detector misses lands the run back in a scoring band.
    The slide keeps the ordering that makes padding never pay.
    """
    e = min(max(e, 0.0), 1.0)
    if e <= pass_e:
        return 1.0
    if e <= fail_e:
        span = fail_e - pass_e
        return 1.0 - (1.0 - fail_score) * ((e - pass_e) / span) if span > 0 else fail_score
    return fail_score * (1.0 - e) / (1.0 - fail_e) if fail_e < 1.0 else 0.0


def score_completeness(ledgers: list[EpisodeLedger], n_required: int | None,
                       cuts: dict, *, generated_cap_s: float | None = None,
                       generated_s: float | None = None,
                       generated_source: str = "",
                       tolerance_s: float | None = None,
                       tolerance_source: str = "") -> CompletenessResult:
    """Ledgers -> C. Missing episodes count zero; untimed deliveries abstain.

    `generated_s` is a legacy trajectory field: requested or stated seconds,
    not measured successful output. No value of `generated_source` promotes
    it to a hard cap. Real source files and the picture clock are compared per
    episode. The explicit `generated_cap_s` argument remains for callers with
    an independently verified complete inventory; production does not supply
    it. Missing detectors/correspondence are reported as an upper bound.
    """
    cc = cuts.get("completeness") or {}
    tol = float(tolerance_s if tolerance_s is not None else cc.get("tolerance_s", 2.0))
    src = tolerance_source or ("given" if tolerance_s is not None else "default")
    pass_e = float(cc.get("band_pass", 0.05))
    fail_e = float(cc.get("band_fail", 0.20))
    fail_score = float(cc.get("band_fail_score", 0.5))
    assembly_ded = max(0.0, float(cc.get("assembly_deduction", 0.1)))

    eps: list[EpisodeScore] = []
    raw_genuine: list[float | None] = []
    for led in ledgers:
        d = led.delivered_s
        if d is None:
            raw_genuine.append(None)
            continue
        g = max(0.0, d - led.deductions_s)
        raw_genuine.append(g)

    # Optional independently verified inventory, never inferred from requests.
    cap = generated_cap_s
    # `per_request` counts requested seconds, often HTTP 202s. Only an
    # explicitly supplied, independently verified inventory may bound C.
    factor, capped = 1.0, False
    if cap is not None and cap >= 0:
        tot = sum(g for g in raw_genuine if g is not None)
        if tot > cap and tot > 0:
            factor, capped = cap / tot, True

    for led, g in zip(ledgers, raw_genuine):
        d, T = led.delivered_s, led.target_s
        # Per detector (may overlap) plus the union that was actually applied,
        # so a reader can see both which detector fired and what it cost.
        ded = {"frozen_s": round(led.frozen_s, 2), "black_s": round(led.black_s, 2),
               "duplicate_s": round(float(led.duplicate_s), 2),
               "seam_s": round(led.seam_s, 2),
               "no_picture_s": round(led.no_picture_s, 2),
               "retime_excess_s": round(led.retime_excess_s, 2),
               "pixel_union_s": round(union_seconds(led.spans, d), 2),
               "deducted_s": round(led.deductions_s, 2)}
        if d is None or T is None or T <= 0:
            eps.append(EpisodeScore(led.video, led.episode, T, d, None, None,
                                    deductions=ded, counts=dict(led.counts),
                                    notes=led.notes + ["时长或目标未知"],
                                    source_comparison=dict(led.source_comparison),
                                    evidence_gaps=led.evidence_gaps + ["duration_or_target_unknown"],
                                    assembled=led.assembled))
            continue
        gg = g * factor if g is not None else 0.0
        # The pass band is relative, floored by the absolute tolerance: 5 % of
        # a 20 s episode is a second, finer than any generation granularity,
        # and an episode must not be marked short for a rounding its model
        # cannot avoid.
        # Duration tolerance is bounded below the failing band. It must
        # never turn a zero-content episode into an on-spec delivery.
        free = min(max(pass_e, max(0.0, tol) / T), math.nextafter(fail_e, 0.0))
        e = 1.0 - jaccard(gg, d, T)
        s = band(e, pass_e=free, fail_e=fail_e, fail_score=fail_score)
        # The duration band exempts small edits/rounding, not non-footage.
        # Every proved padded second must reduce the delivery's value.
        s = min(s, max(0.0, min(1.0, gg / d)) if d > 0 else 0.0)
        # The assembly point, charged per episode. Seconds of real footage are
        # what the lines above measure; this asks the separate question of
        # whether they were cut into the episode that was asked for. Subtracted
        # rather than multiplied in, so it stays a small fixed cost instead of
        # rescaling a score that has already been earned on the pixels.
        a = 0.0 if led.assembled else min(assembly_ded, s)
        if a:
            s -= a
            notes_i = list(led.notes) + [f"只有素材，没有合成分集：扣 {a:.2f}"]
        else:
            notes_i = list(led.notes)
        eps.append(EpisodeScore(led.video, led.episode, T, d, round(gg, 2), round(s, 4),
                                e <= free, ded, capped, notes_i,
                                shortfall=round(e, 4), pass_band=round(free, 4),
                                band=("on_spec" if e <= free else
                                      "short" if e <= fail_e else "off_spec"),
                                counts=dict(led.counts),
                                source_comparison=dict(led.source_comparison),
                                evidence_gaps=list(led.evidence_gaps),
                                assembled=led.assembled,
                                assembly_deduction=round(a, 4)))

    n_found = len(ledgers)
    scored = [e.score for e in eps if e.score is not None]
    if len(scored) < len(eps):
        C, reason = None, "存在时长或目标未知的交付集，不能据其余集推断完整完成度"
    elif n_required:
        n_missing = max(0, n_required - n_found)
        denom = len(scored) + n_missing
        C = (sum(scored) / denom) if denom else None
        reason = "" if C is not None else "没有一集能测出时长"
        if n_missing:
            reason = f"要求 {n_required} 集，定位到 {n_found} 集"
    elif scored:
        C, reason = sum(scored) / len(scored), "任务未给出集数，按找到的集平均"
    else:
        C, reason = None, "任务未给出集数，且没有可测的集"

    delivered_total = round(sum(e.delivered_s or 0.0 for e in eps), 2)
    gen = {
        "generated_s": generated_s,
        "quantity_kind": "requested_or_reported_seconds",
        "source": generated_source or ("given" if generated_cap_s is not None
                                       else "unstated"),
        "delivered_s": delivered_total,
        "stated_by_run": generated_s is not None,
        "bounds_genuine_seconds": cap is not None,
        "applied": capped,
    }
    if generated_s and delivered_total:
        gen["delivered_over_generated"] = round(delivered_total / generated_s, 3)
    if generated_s is None:
        gen["note"] = "轨迹没写生成了多少秒；交付时长无从对照，不按等同处理"
    elif cap is None:
        gen["note"] = (f"轨迹秒数来源为{gen['source']}，只报告不封顶："
                       "请求获受理、请求片长或自述总量均不证明实际成功生成的素材时长")

    gaps = sorted({gap for ep in eps for gap in ep.evidence_gaps})

    n_unassembled = sum(1 for e in eps if not e.assembled)
    asm = {
        "deduction_per_episode": assembly_ded,
        "n_episodes": len(eps),
        "n_assembled": len(eps) - n_unassembled,
        "n_material_only": n_unassembled,
        "charged": round(sum(e.assembly_deduction for e in eps), 4),
    }
    if n_unassembled:
        asm["note"] = (f"{n_unassembled} 集只有素材、没有合成成片："
                       f"每集扣 {assembly_ded}，素材本身的秒数照常计分")

    return CompletenessResult(
        None if C is None else round(C, 4), n_required, n_found, eps,
        delivered_total,
        round(sum(e.genuine_s or 0.0 for e in eps), 2),
        round(sum(led.deductions_s for led in ledgers), 2),
        tol, src, cap, gen, asm, reason,
        bool(gaps) and C is not None and C > 0.0, gaps)
