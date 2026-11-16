#!/usr/bin/env python
"""Evaluate saved agent deliveries and traces with VAHBench.

Read briefs, locate delivered episodes, collect and cross-check evidence, and
compute C (completeness), Q (quality), P (process), and S (aggregate score).
Use --discover-only to inspect delivery selection without scoring or LLM calls.
Runtime settings and extraction credentials are configured in config/default.yaml.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from shared.paths import REPO_ROOT, CONFIG_ROOT

from shared.settings import Settings
from shared.settings import ALL_MODULES
import stage1_inputs.brief as brief
import stage4_checks.video.brief as expectations
import stage4_checks.reports as extract
import stage1_inputs.extractors as extractors
import stage4_checks.cross_checks as flows
import stage1_inputs.client as llm
import shared.preflight as preflight
import stage4_checks.dispatch as dispatch
# HTTP trace extraction is owned by checklist.extractors.
import stage1_inputs.schema as trajectory_schema
from stage3_evidence.collector import EvidenceBus                                 # noqa: E402
from stage2_delivery.discover import find_runs
from stage2_delivery.discover import locate_delivery                  # noqa: E402
from stage4_checks.cross_checks import Ledger                                         # noqa: E402
from stage5_scoring.registry import completeness_of_rubric as rubric_completeness                            # noqa: E402
from stage2_delivery.probe import ProbeCache                                     # noqa: E402
from stage1_inputs.contract import summary as preprocessing_summary
from stage5_scoring import DEFAULT_CUT_POINTS
from stage5_scoring import check_score
from stage5_scoring import score_case
from stage5_scoring import ledger_from_reports
from stage5_scoring import tree_spec

HERE = REPO_ROOT
EVALUATION_VERSION = "2026-09-26-ranking-v15-cfloor2"


# ----------------------------------------------------------------- config --
def load_cut_points(path: Path | None) -> dict:
    path = path or CONFIG_ROOT / "thresholds.yaml"
    if not path.is_absolute():
        path = CONFIG_ROOT.joinpath(*path.parts[1:]) if path.parts[0] == "config" else HERE / path
    if not path.is_file():
        raise ValueError(f"Threshold file does not exist: {path}")
    try:
        import yaml
        overrides = yaml.safe_load(path.read_text("utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError("Threshold file must be valid YAML") from exc
    overrides = {} if overrides is None else overrides
    if not isinstance(overrides, dict):
        raise ValueError("Threshold file must be a mapping")
    if set(overrides) - set(DEFAULT_CUT_POINTS):
        raise ValueError("Threshold file contains unsupported fields")
    return {**DEFAULT_CUT_POINTS, **overrides}


def cut_points_hash(cuts: dict) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(cuts, sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()[:12]


def install_key(cfg) -> None:
    """Load credentials into the selected provider's environment. Never printed."""
    if cfg.lm.provider not in llm.PROVIDERS:
        raise SystemExit("lm.provider 必须是 openrouter、deepseek 或 openai")
    target_env = llm.PROVIDERS[cfg.lm.provider][1]
    source_env = cfg.lm.api_key_env or target_env
    provider_envs = {env for _, env in llm.PROVIDERS.values()}
    if source_env in provider_envs and source_env != target_env:
        raise SystemExit("lm.api_key_env 与 lm.provider 不匹配；设为 null 使用该服务商的标准变量，或指定自定义变量名")
    if cfg.lm.api_key_file:
        p = Path(cfg.lm.api_key_file).expanduser()
        if not p.is_file():
            raise SystemExit(f"lm.api_key_file 不存在: {p}")
        val = p.read_text(encoding="utf-8").strip()
        if not val:
            raise SystemExit("lm.api_key_file 为空")
        if not os.environ.get(target_env, "").strip():
            os.environ[target_env] = val
    elif source_env != target_env:
        val = os.environ.get(source_env, "").strip()
        if val and not os.environ.get(target_env, "").strip():
            os.environ[target_env] = val


# ------------------------------------------------------------- evaluation --
def gather_facts(run, *, model: str, provider: str | None,
                 max_chars: int, max_tokens: int, call_log,
                 base_facts=None, discover_only: bool = False):
    """Build one checklist: deterministic draft first, then LLM coordination.

    The model is the coordination step, not a second scoring path. Order is not
    arbitrary: on a machine-readable log the regex reader is exact and free,
    so letting a model re-derive those numbers would trade certainty for
    variance. The model earns its place on what the rules could not read --
    and measured over `data/` that is 75 of 80 runs (`conditioning` alone is
    unreadable from every HTTP trace by construction, `body_bytes` and no
    body), so a sweep without it never filled the checklist the tree was
    designed against. The 2026-09 caches were exactly that sweep.

    `base_facts` is the deterministic pass when main() already ran it over
    every run to decide, before any work, whether the model is needed for
    this data. `discover_only` is the one mode that skips the model, because
    it does not score; the report then says which gaps were left.
    """
    facts = base_facts if base_facts is not None else \
        extractors.deterministic_extract(run.root, run.root.parent)
    gaps = extractors.needs_lm(facts)
    if not gaps:
        facts.notes.append("规则抽取已填满清单，未调用 LM")
        return facts, None
    if discover_only:
        facts.notes.append("--discover-only：未调用 LM，清单缺口未补填：" + "、".join(gaps))
        return facts, None
    fn = llm.make_call_fn(model, provider=provider, log=call_log,
                          max_tokens=max_tokens)
    if fn is None:
        # main() refuses to start in this state; reaching it here means the
        # key went away mid-sweep. Loud, not a quiet deterministic score: a
        # checklist with holes the model was meant to fill must not be
        # reported as if the rules had read everything.
        raise RuntimeError("LM 后端不可用，清单缺口无法补填：" + "、".join(gaps))
    lm_facts = extractors.lm_extract(run.root, model=model,
                                     revision=call_log.model_served or "unpinned",
                                     call_fn=fn, max_chars=max_chars,
                                     call_log=call_log, focus_fields=gaps,
                                     base_facts=facts)
    merged = _merge_facts(facts, lm_facts)
    # Before and after, so the report shows what the model was asked for and
    # what it could not read either -- the audit trail for "the checklist
    # extracted everything the tree needs".
    left = extractors.needs_lm(merged)
    merged.notes.append("LLM 协调字段：" + ("、".join(gaps) if gaps else "无"))
    merged.notes.append("LM 补填前缺口：" + "、".join(gaps))
    merged.notes.append("LM 补填后仍缺：" + ("、".join(left) if left else "无"))
    if left:
        merged.notes.append("未填字段不等于漏检：原轨迹无充分证据时保持未知；只有核实原轨迹存在可提取证据却未提取时才计漏检。")
    return merged, lm_facts


def lm_gate(gaps: dict[str, list[str]], lm_ready: bool) -> str | None:
    """The refusal main() adds after discovery, as a message or None.

    Whether the model is needed is a property of the data, measured by running
    the deterministic pass over every run first; it is never a flag. With no
    model reachable the sweep may proceed only if no run has a gap left.
    """
    need = {k: v for k, v in gaps.items() if v}
    if lm_ready or not need:
        return None
    lines = [f"LM 抽取不可用，而 {len(need)}/{len(gaps)} 个 run 的清单在规则抽取后仍有缺口："]
    for rid, g in sorted(need.items())[:12]:
        lines.append(f"  {rid}: {'、'.join(g)}")
    if len(need) > 12:
        lines.append(f"  … 另 {len(need) - 12} 个")
    lines.append("LM 是清单抽取的一部分，不是可选项：配置 API key"
                 "（config/default.yaml 的 lm.api_key_env / api_key_file）后重跑。")
    return "\n".join(lines)


def _merge_facts(base, lm):
    """Deterministic wins every field it filled; the LM fills only the Nones.

    One field is re-derived rather than copied: a `generated_seconds` the LM
    *derived* used the LM's own `video_gen` count, and the merged checklist
    keeps the deterministic count. Copying the figure across would report a
    total inconsistent with the count beside it, so the derivation is redone
    over the merged facts.
    """
    from dataclasses import fields as dfields
    # Keep declarations even when a legacy scalar slot is already filled by
    # rules or cannot represent several stages/subjects.
    for statement in lm.statements:
        if statement not in base.statements:
            base.statements.append(statement)
    lm_req = lm.requested
    filled_by_lm = set()
    rederive = (base.requested.generated_seconds is None
                and lm_req.generated_seconds is not None
                and lm_req.generated_seconds_source == "derived")
    for grp in ("calls", "errors", "requested", "claims", "continuity"):
        a, b = getattr(base, grp), getattr(lm, grp)
        for f in dfields(a):
            if getattr(a, f.name) is None and getattr(b, f.name) is not None:
                setattr(a, f.name, getattr(b, f.name))
                key = f"{grp}.{f.name}"
                filled_by_lm.add(key)
                if key in lm.lm_evidence:
                    base.lm_evidence[key] = lm.lm_evidence[key]
    for grp in ("produced", "conditioning"):
        if getattr(base, grp) is None and getattr(lm, grp) is not None:
            setattr(base, grp, getattr(lm, grp))
            if grp in lm.lm_evidence:
                base.lm_evidence[grp] = lm.lm_evidence[grp]
    base.lm_audit = lm.lm_audit
    for grp, total, parts in (
        ("calls", "total", ("image_gen", "video_gen", "audio_gen")),
        ("errors", "n_attempts", ("n_recovered", "n_unrecovered")),
    ):
        values = getattr(base, grp)
        ceiling = getattr(values, total)
        if ceiling is not None and sum(getattr(values, k) or 0 for k in parts) > ceiling:
            rejected = [k for k in (total, *parts) if f"{grp}.{k}" in filled_by_lm]
            for k in rejected:
                setattr(values, k, None)
                base.lm_evidence.pop(f"{grp}.{k}", None)
            if rejected:
                base.notes.append(f"LM 补全与规则计数矛盾，已丢弃：{grp}." + f"、{grp}.".join(rejected))
    if rederive:
        base.requested.generated_seconds = None
        base.requested.generated_seconds_source = None
        base.notes = [n for n in base.notes if "推得" not in n]
        lm.notes = [n for n in lm.notes if "推得" not in n]
        extractors.derive_generated_seconds(base)
    elif base.requested.generated_seconds is None:
        # The coordinator may recover only durations while rules already own
        # the video count. Derive once from the combined checklist, not from
        # the model's intentionally sparse view (10 s vs 3 x 10 s).
        extractors.derive_generated_seconds(base)
    base.backend = f"{base.backend}+{lm.backend}"
    # Which files were sent to the model has to be visible in the report --
    # this is the record of what left the machine.
    base.formats_seen.extend(f for f in lm.formats_seen
                             if f not in base.formats_seen)
    base.source_files.extend(s for s in lm.source_files
                             if s not in base.source_files)
    base.notes.extend(n for n in lm.notes if n not in base.notes)
    return base


def _video_key(video: Path, delivery) -> str:
    """Keep readable names, disambiguating repeated names throughout evidence."""
    return (str(video) if sum(Path(c.info.path).name == video.name for c in delivery.videos) > 1
            else video.name)


def evaluate_run(run, delivery, out_dir: Path, cuts: dict, *,
                 modules: tuple[str, ...], device: str, lm_opts: dict,
                 briefs: dict | None = None, asr_model: str = "base",
                 matched_brief: tuple | None = None,
                 facts_cache: dict | None = None) -> dict:
    """One run, three channels, one ledger.

    The brief is compiled to expectations, the trajectory to a checklist, the
    delivery to module reports -- and every place two of them are read
    together is written to the flow ledger, so the report can show which
    upstream fact reconfigured which downstream module and which downstream
    measurement re-read which upstream claim (EVAL_DESIGN 4.5).
    """
    call_log = llm.CallLog()
    facts, lm_facts = gather_facts(run, call_log=call_log,
                                   base_facts=(facts_cache or {}).get(run.run_id),
                                   **lm_opts)
    if not lm_opts.get("discover_only", False):
        from stage1_inputs.observability import recheck_facts
        recheck_facts(run.root, facts)
    if matched_brief is not None:
        bf, brief_source = matched_brief
    else:
        bf = (briefs or {}).get(brief.prompt_number(run.case))
        brief_source = "case_name" if bf is not None else "none"
    n_cast = len(bf.characters) if bf is not None and bf.characters else None

    if lm_opts.get("discover_only", False):
        from dataclasses import asdict
        return {
            "run_id": run.run_id, "harness": run.harness, "case": run.case,
            "model": run.model, "root": str(run.root), "target": asdict(run.target),
            "evaluation_version": EVALUATION_VERSION,
            "harness_reported_status": run.status, "evaluation_status": "discovery_only",
            "S": None, "C": None, "Q": None, "P": None, "tier": None,
            "coverage": None, "sufficient_evidence": False, "score_is_upper_bound": False,
            "delivery": {"selection": delivery.selection, "assembled": delivery.assembled,
                         "n_videos": len(delivery.videos), "n_candidates": delivery.n_candidates,
                         "material_seconds": delivery.material_seconds,
                         "required_seconds": delivery.required_seconds,
                         "videos": [str(c.info.path) for c in delivery.videos],
                         "disagreement": delivery.disagreement, "reason": delivery.reason},
            "trajectory": {"backend": facts.backend, "field_coverage": facts.coverage,
                           "facts": facts.to_dict()},
            "notes": ["仅发现输入与交付物；未运行评分、检测器或 LM。"],
        }

    ledger = Ledger()
    bus = EvidenceBus(run, delivery, out_dir, device=device, facts=facts,
                      ledger=ledger, asr_model=asr_model)
    if delivery.found and delivery.selection == "declared":
        ledger.add("delivery_location", ["claims.declared_deliverables"], "o2.episode_count",
                   "configure", f"成片由轨迹声明的 {len(delivery.videos)} 个文件定位"
                                "（声明通道，已与内容/名字通道交叉核对）")

    measurements: dict[str, object] = {}
    for m in extract.from_facts(facts, run.target):
        measurements[m.key] = m

    module_status: dict[str, dict] = {}
    per_video: list[dict] = []
    episode_evidence: list[dict] = []
    measured_segments: list[list[int]] = []       # per episode, from pixels
    splice_times: dict[str, list[float]] = {}     # per episode, seam timestamps
    episode_ledgers = []                          # per episode, seconds for C

    readers = {"spec": extract.from_spec, "artifact": extract.from_artifact,
               "visual": extract.from_visual, "physical": extract.from_physical,
               "av": extract.from_av, "seam": extract.from_seam}
    if delivery.found:
        for cand in delivery.videos:
            video = cand.info.path
            video_key = _video_key(video, delivery)
            got: dict[str, object] = {}
            raw: dict[str, dict | None] = {}      # module reports, for the ledger
            for name in modules:
                # The bus runs the module and remembers it. spec gets the
                # episode's source clips when the trajectory left them (the
                # only way `retiming` is ever determined); av gets the brief's
                # cast size; the rest run plain. Cache files are unchanged.
                # spec / artifact / visual / physical / av: each bus method
                # attaches what the run itself supplies (source clips, the
                # pixel-recovered shot boundaries) and records the cache tag.
                rep = getattr(bus, name)(video)
                raw[name] = rep
                module_status.setdefault(name, {})[video_key] = bus.status(name, video, None)
                ms_new = (extract.from_av(rep, expected_speakers=n_cast) if name == "av"
                          else readers[name](rep))
                for m in ms_new:
                    # Which episode a measurement came from survives the
                    # worst-of merge, so the plan-text flows can look up
                    # that episode's beats for its timestamps.
                    m.detail.setdefault("episode", video_key)
                    got[m.key] = m
                if name == "artifact" and rep is not None:
                    segs = extract.dig(rep, "splices.segment_frames")
                    if isinstance(segs, list):
                        measured_segments.append([int(x) for x in segs])
                    sp = extract.dig(rep, "splices.splices")
                    if isinstance(sp, list):
                        splice_times[str(video)] = [
                            float(x["t_s"]) for x in sp
                            if isinstance(x, dict) and x.get("t_s") is not None]
            if module_status.get("spec", {}).get(video_key) != "ok":
                for m in extract.spec_from_probe(cand.info, run.target):
                    got.setdefault(m.key, m)
            if "av" in modules:
                # Runs of nothing on the track, from ffmpeg itself -- no
                # module, no venv. Merged worst-of like everything else, so a
                # silent-track VOID from av still wins.
                sm = extract.from_silence(bus.silence_runs(video), cand.info.duration)
                if sm.key not in got or got[sm.key].verdict is not extract.Verdict.VOID:
                    got[sm.key] = sm
            from stage4_checks.penalties import normalize_episode
            episode_ledger = ledger_from_reports(
                str(video), cand.episode, run.target.seconds_per_episode,
                cand.info.duration, raw.get("spec"), raw.get("artifact"),
                raw.get("visual"), assembled=delivery.assembled,
                probe_picture_s=getattr(cand.info, "video_duration", None))
            if raw.get("seam"):
                seam_m = extract.from_seam(raw["seam"], episode_ledger.spans)[0]
                seam_m.detail.setdefault("episode", video_key)
                got[seam_m.key] = seam_m
                if seam_m.detail.get("already_charged_in_C"):
                    ledger.add("penalty_interval_union", ["completeness.spans"], seam_m.key,
                               "annotate", "已完整计入 C 的冻结片头不再扣切点连续性")
            normalize_episode(got, ledger)
            per_video.append({"video": str(video), "episode": cand.episode,
                              "duration": cand.info.duration})
            episode_evidence.append({"video": str(video), "episode": cand.episode,
                                     "duration": cand.info.duration, "measurements": dict(got),
                                     "assembly": bus.assembly_evidence(video)})
            # Completeness reads seconds, not verdicts: what this episode
            # delivered and what the modules proved was not real content.
            episode_ledgers.append(episode_ledger)
            _merge_worst(measurements, got, cuts)
        _preserve_quality_episode_gaps(measurements, episode_evidence)
        if n_cast and "av" in modules:
            ledger.add("brief_cast_sizes_speakers", ["brief.characters"],
                       "av.speaker_consistency", "configure",
                       f"任务书 {n_cast} 个角色：仅记录背景信息；声音簇数量不能证明角色声音一致")

    # Closed-loop checks: what the trajectory says x what was delivered. These
    # come last and merge worst-of, so a silent-track FAIL from av or a
    # no-delivery VOID overrides any stream-presence PASS from spec.
    delivered_secs = sum((v.get("duration") or 0.0) for v in per_video) or None
    delivery_checks = extract.from_delivery(len(delivery.videos), run.target, delivered_secs, facts,
                                             delivery.disagreement, root=run.root)
    _merge_worst(measurements, {m.key: m for m in delivery_checks}, cuts)
    claim_checks = [m for m in delivery_checks if m.key.startswith("claims.")]

    from stage4_checks.process.honesty import check_episode_claims
    for claim_check in check_episode_claims(facts, run.root, episode_evidence):
        claim_checks.append(claim_check)
        _merge_worst(measurements, {claim_check.key: claim_check}, cuts)
        if claim_check.detail.get("scoped_checks") or claim_check.detail.get("assemblies"):
            ledger.add("scoped_claim_verification", ["agent_claims", "episode_measurements", "executed_assembly"],
                       claim_check.key, "fail" if claim_check.verdict is extract.Verdict.FAIL else "annotate",
                       claim_check.reason or "按同一集/文件核验声明；源片段数不以像素切点代替")

    # 2.3.b/c/d: brief x trajectory, no pixel involved. They run whatever the
    # modules did and whatever shipped -- a run that delivered nothing still
    # planned or failed to plan, and that is the run worth knowing it about.
    _merge_worst(measurements,
                 {m.key: m for m in expectations.evaluate_text(bf, facts)}, cuts)

    # The rest pull evidence through the bus -- a transcript, a seeded visual
    # pass, a subtitle file -- and are where brief, trajectory and delivery
    # are read together.
    if bf is not None and delivery.found and ("av" in modules or "visual" in modules):
        exp = {m.key: m for m in expectations.evaluate(bf, bus, facts, cuts)}
        # These checks cover the delivered series and actual reference files.
        # A plain per-video match must not override a missing reference in a
        # later episode; the broader check is authoritative even on ABSTAIN.
        for key in ("provenance.conditioning_edges", "brief.characters_anchored",
                    "visual.scene_reference", "brief.subtitles_aligned"):
            if key in exp:
                measurements[key] = exp.pop(key)
        _merge_worst(measurements, exp, cuts)
    # The plan on the pixel clock, per episode: what every module's timed
    # finding is read against. No module receives it as a flag.
    plan_spans = ({_video_key(Path(c.info.path), delivery): bus.plan_spans(Path(c.info.path))
                   for c in delivery.videos} if delivery.found else {})
    flows.apply(measurements, facts, ledger, splice_times=splice_times,
                plan_spans=plan_spans,
                voiceover_allowed=bf.voiceover_allowed if bf is not None else None,
                n_cast=n_cast)

    # Request durations (including accepted asynchronous jobs) are not an
    # inventory of successful output footage. Report them, never cap C with
    # them. Editing can trim clips: requested clip length must not relax the
    # delivery tolerance or exempt frozen/repeated seconds.
    gen_s = facts.requested.generated_seconds
    gen_src = facts.requested.generated_seconds_source or ""
    # Did the agent generate any video at all? Either channel is enough: a
    # log may record the calls without their durations, or the total without
    # the calls. Only a run where neither says yes can score a completeness
    # of zero -- see rubric.score_case.
    video_generated = bool(facts.calls.video_gen) or bool(gen_s)
    card = score_case(measurements, episode_ledgers,
                      n_required=run.target.episodes, delivered=delivery.found,
                      assembled=delivery.assembled,
                      generated_s=gen_s, generated_source=gen_src,
                      video_generated=video_generated,
                      cuts=cuts)
    rep = _assemble(run, delivery, card, facts, call_log,
                    module_status, per_video, cuts)
    rep["brief"] = ({"prompt_no": bf.prompt_no, "title": bf.title, "source": brief_source,
                     "characters": [c.name for c in bf.characters],
                     "declared_character_count": bf.declared_character_count,
                     "parse_warnings": bf.parse_warnings,
                     "locations": bf.locations,
                     "word_cap": bf.word_cap, "voiceover_allowed": bf.voiceover_allowed,
                     "mandated_lines": bf.mandated_lines,
                     "timing_bands": [[b.start, b.end, b.label] for b in bf.timing_bands],
                     "deliverables": [d.key for d in bf.deliverables]}
                    if bf else None)
    rep["evidence_demanded"] = bus.log
    rep["execution_failures"] = [
        {"module": "asr", "video": str(key[1]), "reason": "transcription unavailable"}
        for key, value in getattr(bus, "_memo", {}).items()
        if key[0] == "asr" and value is None]
    rep["seeds"] = bus.seeds
    rep["flows"] = ledger.to_list()
    def measured_record(m):
        return {"key": m.key, "verdict": m.verdict.value, "value": m.value,
                "scale": m.scale.value, "reason": m.reason, "detail": m.detail}
    rep["process_evidence"] = {k: measured_record(m) for k, m in measurements.items()
                               if k == "trace.unrecovered_failures" or k.startswith("claims.")}
    # Keep also the checks before worst-of: a known file existence must not
    # erase the report of an unresolved self-check on that file.
    rep["claim_verification"] = [measured_record(m) for m in claim_checks]
    rep["measurements"] = {k: measured_record(m) for k, m in measurements.items()}
    from stage5_scoring.capabilities import METHOD_DIAGNOSTIC_SOURCES
    rep["method_diagnostics"] = {
        k: {"scored": False, "reason": reason, "measurement": rep["measurements"].get(k)}
        for k, reason in METHOD_DIAGNOSTIC_SOURCES.items()}
    rep["observability_audit"] = facts.lm_audit.get("observability_recheck", {})
    rep["episode_measurements"] = [
        {**{k: v for k, v in ep.items() if k != "measurements"},
         "measurements": {k: measured_record(m) for k, m in ep["measurements"].items()}}
        for ep in episode_evidence]
    rep["checklist_usage"] = flows.checklist_usage(facts, measurements, ledger)
    _finalize_evaluation_status(rep, facts, discover_only=lm_opts.get("discover_only", False))
    return rep


def _finalize_evaluation_status(rep, facts, *, discover_only=False):
    """Separate a computed score from discovery or failed extraction."""
    if discover_only:
        rep.update(evaluation_status="discovery_only", S=None, C=None, Q=None, P=None,
                   sufficient_evidence=False, score_is_upper_bound=False)
        return
    audit = facts.lm_audit
    total = audit.get("chunks_total", 0)
    failed_extraction = total and (audit.get("chunks_valid", 0) != total
                                   or not audit.get("input_complete", False))
    recheck = audit.get("observability_recheck", {})
    failed_extraction = failed_extraction or recheck.get("passed") is False
    failures = list(rep.get("execution_failures", []))
    failures.extend({"module": module, "video": video, "reason": status}
                    for module, videos in rep.get("modules", {}).items()
                    for video, status in videos.items() if status != "ok")
    rep["execution_failures"] = failures
    if failed_extraction:
        rep.update(evaluation_status="extraction_failed", S=None,
                   sufficient_evidence=False, score_is_upper_bound=False)
        rep.setdefault("notes", []).append(
            f"抽取或原始输入复核失败：{audit.get('chunks_valid', 0)}/{total} 块有效，"
            f"{recheck.get('errors', [])}；不能把抽取失败当成未采用某种方法")
    elif failures and rep.get("C") != 0:
        rep.update(evaluation_status="incomplete_evidence",
                   sufficient_evidence=False, score_is_upper_bound=False,
                   score_is_partial=True)
        rep.setdefault("notes", []).append(
            "检测或转写执行失败：仅按已测有效项给条件分；保留 execution_failures，"
            "不把评测失败当 Agent 缺陷，不宣称全部检测成功")
    else:
        rep["evaluation_status"] = "completed" if rep.get("S") is not None else "incomplete_evidence"


def _preserve_quality_episode_gaps(acc: dict, episodes: list[dict]) -> None:
    """A successful episode must not hide a detector failure on another."""
    from stage5_scoring.video.quality.consistency import CONSISTENCY
    from stage5_scoring.video.quality.plausibility import PLAUSIBILITY
    from copy import deepcopy
    series_keys = {"brief.characters_anchored", "visual.scene_reference", "brief.subtitles_aligned"}
    for key in (*CONSISTENCY.source_keys, *PLAUSIBILITY.source_keys):
        if key in series_keys:
            continue
        rows, gaps = [], []
        for ep in episodes:
            m = ep["measurements"].get(key)
            rows.append({"episode": ep["video"], "verdict": m.verdict.value if m else "ABSTAIN",
                         "value": m.value if m else None, "reason": m.reason if m else "未产出测量",
                         "detail": deepcopy(m.detail) if m else {}})
            if m is None or m.verdict is extract.Verdict.ABSTAIN:
                gaps.append(ep["video"])
        if gaps:
            old = acc.get(key)
            if old is not None and old.measurable:
                # A missing episode cannot erase a proved defect elsewhere.
                old.detail.update(episode_results=rows, missing_episodes=gaps)
            else:
                acc[key] = extract._m(key, extract.Verdict.ABSTAIN,
                                      reason=f"{len(gaps)}/{len(episodes)} 集缺少 {key} 证据",
                                      episode_results=rows, missing_episodes=gaps)


def _merge_worst(acc: dict, new: dict, cuts: dict) -> None:
    """Worst episode wins per check; a measured value always beats an abstention."""
    from shared.types import Verdict
    from stage5_scoring.node import measurement_rank
    from stage5_scoring.registry import SOURCE_MAPPINGS
    for k, m in new.items():
        old = acc.get(k)
        if old is None:
            acc[k] = m
            continue
        mapping = SOURCE_MAPPINGS.get(k, "band")
        if measurement_rank(m, cuts, mapping) < measurement_rank(old, cuts, mapping):
            acc[k] = m


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def _category(c) -> dict:
    """One quality dimension or trajectory category, leaves inside."""
    return {"id": c.id, "name": c.name, "score": c.score,
            "observed_score": c.observed_score, "reason": c.reason,
            "source_coverage": round(c.source_coverage, 4),
            "n_applicable": c.n_applicable, "missing_sources": c.missing_sources,
            "coverage": round(c.coverage, 4),
            "n_leaves": c.n_leaves, "n_measured": c.n_measured,
            "leaves": [{"id": r.id, "key": r.key, "name": r.name, "score": r.score,
                        "abstained": r.abstained, "abstain_reason": r.abstain_reason,
                        "n_sources": r.n_sources, "n_measured": r.n_measured,
                        "n_applicable": r.n_applicable, "not_applicable": r.not_applicable,
                        "missing_sources": r.missing_sources,
                        "sources": r.per_source, "reasons": r.reasons}
                       for r in c.leaves]}


def _assemble(run, delivery, card, facts, call_log, module_status,
              per_video, cuts) -> dict:
    return {
        "run_id": run.run_id, "harness": run.harness, "case": run.case,
        "model": run.model, "root": str(run.root),
        "target": {"episodes": run.target.episodes,
                   "seconds_per_episode": run.target.seconds_per_episode,
                   "aspect_ratio": run.target.aspect_ratio,
                   "resolution": run.target.resolution,
                   "source": run.target.source},
        "harness_reported_status": run.status,
        "delivery": {"selection": delivery.selection,
                     "assembled": delivery.assembled,
                     "n_videos": len(delivery.videos),
                     "n_candidates": delivery.n_candidates,
                     "material_seconds": round(delivery.material_seconds, 1),
                     "required_seconds": round(delivery.required_seconds, 1),
                     "videos": [str(c.info.path) for c in delivery.videos],
                     "disagreement": delivery.disagreement,
                     "reason": delivery.reason},
        # ---- the tree: S at the root, C / Q / P underneath ----------------
        "S": card.S, "C": card.C, "Q": card.Q, "P": card.P, "tier": card.tier,
        "scoring_version": cuts.get("version", "legacy"),
        "evaluation_version": EVALUATION_VERSION,
        "coverage": card.coverage,               # measured scored leaves / leaves
        "sufficient_evidence": card.sufficient,
        "process_unmeasured": card.process_unmeasured,
        "honesty_violation": card.honesty_violation,
        "completeness": card.completeness.to_dict(),
        "score_is_upper_bound": card.score_is_upper_bound,
        "score_is_partial": card.score_is_partial,
        "aggregation": card.aggregation,
        "completeness_inputs": card.completeness_inputs,
        "quality": {"score": card.quality.score,
                    "coverage": round(card.quality.coverage, 4),
                    "reason": card.quality.reason,
                    "dimensions": {k: _category(d) for k, d in card.quality.dimensions.items()}},
        "process": {"score": card.process.score,
                    "coverage": round(card.process.coverage, 4),
                    "categories": {k: _category(c) for k, c in card.process.categories.items()}},
        "abstained": card.abstained, "voided": card.voided,
        "diagnostics": card.diagnostics,         # measured, reported, not scored
        "rubric_completeness": rubric_completeness(),
        "tree": tree_spec(),
        "negative_feedback": card.negative_feedback, "notes": card.notes,
        # The checklist twice: `facts` grouped the way a log shows it, which
        # is the order a parser fills, and `by_node` grouped the way the tree
        # reads it (docs/structure.md B.2), which is the order a grader asks.
        # The second is derived from the first, never stored separately.
        "trajectory": {"backend": facts.backend,
                       "preprocessing": preprocessing_summary(facts.lm_audit.get('preprocessing')),
                       "formats_seen": sorted(set(facts.formats_seen)),
                       "field_coverage": round(facts.coverage, 4),
                       "by_node": trajectory_schema.by_node(facts),
                       "facts": facts.to_dict()},
        "lm": {"provider": call_log.provider,
               "model_requested": call_log.model_requested,
               "model_served": call_log.model_served,
               "model_drifted": call_log.model_drifted,
               "n_calls": call_log.n_calls,
               "prompt_chars": call_log.total_prompt_chars,
               "errors": call_log.errors} if call_log.n_calls or call_log.errors else None,
        "modules": module_status, "per_video": per_video,
        "cut_points": cuts, "cut_points_hash": cut_points_hash(cuts),
    }


# --------------------------------------------------------------------- cli --
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="轨迹根目录")
    ap.add_argument("--run", default=None,
                    help="run_id 子串过滤，如 harness/codex 或 prompt_01；不给则跑全部")
    ap.add_argument("--out-dir", default="_out")
    ap.add_argument("--discover-only", action="store_true",
                    help="只定位成片，不跑检查也不跑 LM（预检用，输出的 coverage 不是评分覆盖率）")
    ap.add_argument("--config", default=None,
                    help="YAML 配置文件；默认使用随代码分发的 config/default.yaml")
    ap.add_argument("--allow-partial", action="store_true",
                    help="允许缺少检测器或主动关闭模块的部分评测；不能作为完整论文复现")
    args = ap.parse_args(argv)
    try:
        cfg = Settings.load(Path(args.config) if args.config else None)
        cuts = load_cut_points(Path(cfg.cut_points) if cfg.cut_points else None)
    except (ValueError, OSError) as exc:
        ap.error(str(exc))

    # Windows: a redirected stdout defaults to the console code page (gbk),
    # which cannot encode the preflight's tick marks or any reason string.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass

    if not args.discover_only:
        install_key(cfg)
    dispatch.USE_REFERENCE = cfg.reference_detectors
    want_modules = cfg.modules
    if args.discover_only:
        want_modules = ()
    want_lm = not args.discover_only

    pf = preflight.check(Path(args.data), want_modules=want_modules,
                         want_lm=want_lm, provider=cfg.lm.provider,
                         model=cfg.lm.model)
    print("预检：")
    print(pf.render())
    if pf.fatal:
        print("\n有致命项，终止。修复上述环境问题后重跑。")
        return 2
    unavailable = set(want_modules) - set(pf.runnable_modules)
    origins = {m: pf.module_origins.get(m, "missing") for m in want_modules}
    # From the resolved origin, not from what preflight thinks can run: a
    # module preflight doubts is still attempted, and whatever it measures
    # must be attributed.
    reference = sorted(m for m in want_modules if origins.get(m) == "reference")
    # "full" is the paper's configuration: all six original detectors. The
    # bundled reference reimplementation measures real media, but it is a
    # different instrument and covers less, so it never makes a run full.
    is_full = (not unavailable and not reference
               and set(want_modules) == set(ALL_MODULES))
    if reference and not args.discover_only:
        print("\n使用参考重实现的模块：" + "、".join(reference)
              + "。它们不是论文所用检测器，阈值未经校准；结果不能与论文分数直接比较。")
    if not args.discover_only and not args.allow_partial and not is_full:
        print("\n完整评测需要全部六个原始检测模块（当前缺少："
              + ("、".join(sorted((set(ALL_MODULES) - set(pf.runnable_modules)) | set(reference))) or "无")
              + "）。如使用参考重实现或仅检查已有功能，显式使用 --allow-partial。")
        return 2
    # Retain requested modules in partial runs: their failure must remain in
    # evidence records, and built-in FFmpeg/ASR checks must still be reachable.
    modules = tuple(want_modules)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = ProbeCache(out_dir / "probe_cache.json")

    runs = find_runs(Path(args.data), None)
    if args.run:
        runs = [r for r in runs if args.run in r.run_id.replace("\\", "/")]
    if not runs:
        print("\n没有匹配的 run，未执行评测。请核对数据目录和 --run 筛选。")
        return 2

    # The deterministic pass over every run, before any work. It is cheap, it
    # is needed anyway, and it is what decides whether the model is required
    # for *this* data: the rules read what they can, and the model is owed
    # whatever is still None. Nothing about that is a flag.
    pre_facts: dict[str, object] = {}
    if want_lm:
        gaps: dict[str, list[str]] = {}
        for r in runs:
            f = extractors.deterministic_extract(r.root, r.root.parent)
            pre_facts[r.run_id] = f
            gaps[r.run_id] = extractors.needs_lm(f)
        refusal = lm_gate(gaps, pf.lm_ready)
        if refusal:
            print("\n" + refusal)
            return 2
        n_need = sum(1 for g in gaps.values() if g)
        if n_need:
            print(f"\n清单：{n_need}/{len(runs)} 个 run 在规则抽取后仍有缺口，由 LM 补填。"
                  f"注意：轨迹文件内容将发送到 {pf.lm_provider}"
                  f"（每块约 {cfg.lm.max_chars} 字符，处理全部分块）", flush=True)
        else:
            print("\n清单：规则抽取已填满全部 run，本次不调用 LM", flush=True)

    # The brief: prompt.md beside the data when there is one, else the task
    # text the trajectory itself carries. It also fills a target no record
    # supplied -- episodes, seconds and aspect are properties of the prompt.
    briefs = brief.load_briefs(Path(args.data))
    matched = {r.run_id: brief.match_brief(briefs, r) for r in runs}
    n_from_brief = brief.apply_brief_targets(runs, matched)
    hows = {}
    for bf, how in matched.values():
        hows[how] = hows.get(how, 0) + 1
    print(f"任务书：prompt.md 解析到 {len(briefs)} 条；run 的任务书来源 "
          + "、".join(f"{k} {v}" for k, v in sorted(hows.items()))
          + (f"；{n_from_brief} 个 run 的目标由任务书补齐" if n_from_brief else ""), flush=True)

    print(f"\n发现 {len(runs)} 个 run，正在定位成片…", flush=True)
    deliveries = [(r, locate_delivery(r, cache)) for r in runs]
    cache.save()
    n_ok = sum(1 for _, d in deliveries if d.assembled)
    n_material = sum(1 for _, d in deliveries if d.found and not d.assembled)
    print(f"{n_ok} 个定位到分集成片，{n_material} 个仅有未合成素材，"
          f"{len(deliveries)-n_ok-n_material} 个未定位到可评估交付\n", flush=True)

    lm_opts = {"model": cfg.lm.model,
               "max_tokens": cfg.lm.max_tokens,
               "provider": pf.lm_provider or cfg.lm.provider,
               "max_chars": cfg.lm.max_chars,
               "discover_only": args.discover_only}
    def work(item):
        r, d = item
        try:
            return evaluate_run(r, d, out_dir, cuts, modules=modules,
                                device=cfg.device, lm_opts=lm_opts, briefs=briefs,
                                asr_model=cfg.asr_model,
                                matched_brief=matched.get(r.run_id),
                                facts_cache=pre_facts)
        except Exception as e:            # one run must never kill the sweep
            return {"run_id": r.run_id, "harness": r.harness, "case": r.case,
                    "model": r.model, "S": None, "C": None, "Q": None, "P": None,
                    "tier": None, "coverage": 0.0, "sufficient_evidence": False,
                    "evaluation_status": "failed", "error": f"{type(e).__name__}: {e}"}

    if cfg.jobs > 1:
        with ThreadPoolExecutor(max_workers=cfg.jobs) as ex:
            reports = list(ex.map(work, deliveries))
    else:
        reports = [work(it) for it in deliveries]

    (out_dir / "cases").mkdir(exist_ok=True)
    for rep in reports:
        rep["execution_mode"] = ("discovery" if args.discover_only else
                                 "full" if is_full else "partial")
        rep["requested_modules"] = list(want_modules)
        rep["unavailable_modules"] = sorted(unavailable)
        rep["detector_origins"] = dict(origins)
        rep["reference_modules"] = reference
        (out_dir / "cases" / f"{_safe(rep['run_id'])}.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    # A filtered invocation (--run) re-scores a few runs; it must not replace
    # the sweep's summary with those few rows. Merge by run_id into what is
    # already there -- summary.json, and cases/ as the fuller record when the
    # summary was itself clobbered earlier.
    summary = list(reports)
    if args.run:
        prior: dict[str, dict] = {}
        for src in [out_dir / "summary.json", *sorted((out_dir / "cases").glob("*.json"))]:
            try:
                got = json.loads(src.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for r in (got if isinstance(got, list) else [got]):
                if isinstance(r, dict) and r.get("run_id"):
                    prior.setdefault(r["run_id"], r)
        for rep in reports:
            prior[rep["run_id"]] = rep
        summary = sorted(prior.values(), key=lambda r: r["run_id"])
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(out_dir / "summary.csv", summary)
    _print_table(reports)
    return 1 if any(r.get("error") or r.get("evaluation_status") == "extraction_failed"
                    for r in reports) else 0


Q_DIMS = ("specification", "consistency", "plausibility")
P_CATS = ("reliability", "honesty", "planning")


def _write_csv(path: Path, reports: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "harness", "case", "model", "n_videos",
                    "S", "C", "Q", "P", "tier",
                    *[f"q_{k}" for k in Q_DIMS], *[f"p_{k}" for k in P_CATS],
                    "coverage", "sufficient", "process_unmeasured",
                    "honesty_violation", "delivered_s", "genuine_s",
                    "traj_backend", "selection", "reason", "score_is_upper_bound", "score_is_partial"])
        for r in reports:
            d = r.get("delivery") or {}
            qd = ((r.get("quality") or {}).get("dimensions")) or {}
            pc = ((r.get("process") or {}).get("categories")) or {}
            comp = r.get("completeness") or {}
            t = r.get("trajectory") or {}
            w.writerow([r.get("run_id"), r.get("harness"), r.get("case"),
                        r.get("model"), d.get("n_videos", 0),
                        r.get("S"), r.get("C"), r.get("Q"), r.get("P"), r.get("tier"),
                        *[(qd.get(k) or {}).get("score") for k in Q_DIMS],
                        *[(pc.get(k) or {}).get("score") for k in P_CATS],
                        r.get("coverage"), r.get("sufficient_evidence"),
                        r.get("process_unmeasured"), r.get("honesty_violation"),
                        comp.get("delivered_s"), comp.get("genuine_s"),
                        t.get("backend"), d.get("selection"),
                        (d.get("reason") or r.get("error") or "")[:120],
                        r.get("score_is_upper_bound"), r.get("score_is_partial")])


def _print_table(reports: list[dict]) -> None:
    """One user-facing total per case; measurement details stay internal."""
    for report in reports:
        score = report.get("S")
        value = f"{score:.2f}" if score is not None else "评测未完成"
        print(value if len(reports) == 1 else f"{report['run_id']}\t{value}")


if __name__ == "__main__":
    raise SystemExit(main())
