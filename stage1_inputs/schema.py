"""The trajectory checklist: what the benchmark needs out of an execution log.

**This is supporting evidence, not the rubric.** The rubric is the tree in `shared/rubric/`.
Every field here exists to feed a check or to cross-examine a claim, and each
one says which below. A field that does neither is dead weight and was removed:
the first version carried `errors.n_retries` (never filled by any of 45 runs),
`resources.n_processes` (fed nothing), `ordering.first_*` (diagnostic only) and
`claims.plan_text_present` (duplicated `planning_artifact`). 2026-09-22 removed
the last seven that fed neither the tree nor a flow -- `calls.reasoning`,
`calls.polls`, `errors.status_counts`, `errors.examples`, `requested.models`
and both `resources` fields (cost and wall clock, and with them the report's
`efficiency` block): they were report decoration, and a checklist field that
decorates is a field nobody keeps correct.

One schema, many source formats. Measured across `data/`, agents leave at least
four mutually unreadable kinds of trajectory:

    trace.jsonl      HTTP-level records      openmontage · univa · videoclaw · vimax
    costs.jsonl      a cost ledger           claudecode · codex
    events.jsonl     tool calls + results    openclaw · hermes · opensquilla
                     (three record shapes)
    TRAJECTORY.md    human-written prose     claudecode
    run_video.log    semi-structured text    claudecode

A parser per format does not scale and a generic reader cannot bridge shapes
that differ this much, so the *requirement* is declared here and each backend
fills what it can see. **Every field is optional.** A backend that cannot
determine a field leaves it None, and None propagates to ABSTAIN -- never to a
zero, and never to a guess.

Two stages build one checklist: `deterministic` (regex/JSON, exact, free)
creates the draft, and a small pinned language model coordinates only the
fields the deterministic stage could not read. On the 23 trajectories both can
read, `compare()` makes the model's agreement with the exact parser a measured
number rather than an assumption.
"""
from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from dataclasses import is_dataclass

# What every field is for, indexed by the tree node it feeds (SCORING_SYSTEM.md
# B.2). This is the reading order that matters: "what does 2.4 need out of a
# trajectory" is the question, and the dataclasses below are grouped the other
# way -- by what one log can show -- because that is the order a parser sees.
# `by_node()` turns a filled checklist into this shape for the report, computed
# from CONSUMERS rather than written twice.
#
# Nothing in the checklist is here for decoration. A field feeds a tree node, or
# it feeds a cross-channel rule in flows.py; there is no third kind any more.
ROLES = {
    "1.1  完成度": "requested.generated_seconds / _source / clip_durations —— "
                   "请求/自述量只作对照；完成度以逐集源素材、画面时长和缺陷区间为据",
    "1.2.1.c 请求参数": "requested.aspect_ratios / resolutions —— 像素侧供不出的一项："
                        "对着 9:16 的任务书请求 16:9，文件出生前就错了",
    "2.1.a 未恢复错误": "errors.n_attempts / n_unrecovered —— 只算 agent 自己的错（环境错剔除），"
                        "且只算没救回来的：失败后重试成功不扣分",
    "2.2.a 自述一致": "claims.self_reported_status / declared_deliverables / "
                      "declared_clips_per_episode —— 逐条对照实测。**只收 agent 自己写的**："
                      "run.json 是跑批脚本在 agent 退出后写的（status 来自 returncode，"
                      "videos 是一次 rglob），不是自述",
    "2.3.a 角色资产": "produced + conditioning —— 角色图、关键帧、源片段进没进后续请求",
    "2.3.b 连续性契约": "continuity.source / text —— agent 生成前把外观写下来没有",
    "2.3.c 规格进 prompt": "requested.prompts —— 写下的外观有没有进自己的 prompt",
    "2.3.d 分镜计划": "continuity.shot_plan —— 计划的镜头与地点，也是 evidence 的时间轴锚",
    "flows": "claims.planning_artifact / deviations / self_checks、calls.image_gen / audio_gen"
             " —— 不直接计分，供 flows.py 改判或加注（自检对实测、音轨三角、偏差归因）",
}

# Every field's consumers, by kind:
#   check    a rubric check reads it directly
#   flow     a cross-channel rule in flows.py / expectations.py re-reads a
#            measurement in its light
#   config   it decides how a downstream module is run (evidence.py)
#   context  reported, never scored
# tests/test_flows.py asserts no field is an orphan, and
# flows.checklist_usage() reports, per run, which consumers actually fired --
# so "the checklist is fully used" is a number in every report, not a claim.
CONSUMERS: dict[str, list[tuple[str, str]]] = {
    # No observed video call is not proof of zero successfully produced footage.
    "calls.total": [("check", "provenance.duration_provenance")],
    "calls.image_gen": [("flow", "image_triangulation")],
    # feeds 1.1 through `derive_generated_seconds` (typical clip x call count)
    "calls.video_gen": [("check", "provenance.duration_provenance"),
                        ("flow", "seam_alignment")],
    "calls.audio_gen": [("flow", "audio_triangulation")],
    "errors.n_attempts": [("check", "trace.unrecovered_failures")],
    "errors.n_unrecovered": [("check", "trace.unrecovered_failures")],
    "errors.n_recovered": [("context", "errors")],
    "requested.aspect_ratios": [("check", "o2.requested_params"),
                                ("flow", "requested_vs_delivered")],
    "requested.resolutions": [("check", "o2.requested_params"),
                              ("flow", "requested_vs_delivered")],
    "requested.clip_durations": [("check", "provenance.duration_provenance"),
                                 ("flow", "seam_alignment")],
    "requested.generated_seconds": [("check", "provenance.duration_provenance")],
    "requested.generated_seconds_source": [("check", "provenance.duration_provenance"),
                                           ("flow", "pixels_corroborate_duration")],
    "requested.prompts": [("check", "brief.specs_in_prompts"),
                          ("flow", "contract_explains_consistency")],
    "claims.declared_deliverables": [("check", "claims.deliverables_consistent"),
                                     ("config", "delivery_location")],
    "claims.self_reported_status": [("check", "claims.status_consistent")],
    "claims.planning_artifact": [("flow", "planned_vs_delivered")],
    "claims.deviations": [("flow", "deviation_attribution")],
    "claims.declared_clips_per_episode": [("check", "claims.structure_consistent"),
                                          ("flow", "seam_alignment")],
    "claims.self_checks": [("flow", "self_checks_contradicted")],
    "produced": [("config", "sheets_seed_o3"), ("config", "keyframes_seed_scene"),
                 ("config", "clips_seed_retiming"), ("config", "clips_seed_seam"),
                 ("flow", "audio_triangulation"),
                 ("check", "provenance.conditioning_edges")],
    "conditioning": [("config", "sheets_seed_o3"),
                     ("check", "provenance.conditioning_edges")],
    "continuity.source": [("check", "brief.continuity_contract")],
    "continuity.text": [("check", "brief.continuity_contract")],

    # Two consumers, and the config one is the point. The plan is the only
    # channel that can assert two shots are in *different* rooms: scene_links'
    # ORB verification is asymmetric by construction (a match proves sameness,
    # silence proves nothing) and the CLIP clustering it replaced scored
    # AUC~0.5. `plan_seeds_locations` maps the plan's spans onto the pixel-
    # recovered cuts and hands the result to the visual module, which reports
    # it beside the verified links rather than in place of them.
    "continuity.shot_plan": [("check", "brief.scene_plan"),
                             # and, once the plan is on the pixel clock
                             # (evidence.plan_spans), the modules' timed
                             # findings are read against its beats:
                             ("flow", "plan_vs_brief_voiceover"),    # plan V.O. x brief ban
                             ("flow", "plan_explains_physical")],    # plan effects x depth flags
}


@dataclass
class CallInventory:
    """How much work of each kind the run did. Counts, not content.

    `total` counts every call the log records, polls included. It is read by
    `derive_generated_seconds` can summarise requested volume, but neither a
    complete call list nor zero observed video requests proves output volume.
    Shell-generated clips and asynchronous failures require artifact evidence.
    """

    total: int | None = None
    image_gen: int | None = None
    video_gen: int | None = None        # generation requests only, never polls
    audio_gen: int | None = None        # TTS, ASR, music


@dataclass
class ErrorProfile:
    """Terminal outcomes of identifiable media-production tasks.

    One output/shot/job is one task across retries and fallback tools. Only
    image, video, audio and bound assembly operations count. Polls update their
    submission; reasoning and filesystem calls are not denominator events.
    Incomplete identity, outcome, chronology or attribution keeps all counts
    unknown. The deterministic reliability_audit records known tasks and gaps.
    These fields are not filled by model-generated scalar totals.
    """

    n_attempts: int | None = None        # denominator: distinct terminal production tasks
    n_unrecovered: int | None = None     # attempts whose last call still failed
    n_recovered: int | None = None       # failed then succeeded; reported only


@dataclass
class RequestedParams:
    """What the agent *asked the generator for* -- declared intent.

    Present on 685 media calls across `data/`. This is the one thing the pixel
    channel cannot supply: a run that requests 16:9 against a 9:16 brief, or
    720p against 720x1280, is wrong before the file exists. vimax does exactly
    that.
    """

    aspect_ratios: list[str] | None = None
    resolutions: list[str] | None = None
    # The text the agent sent an *image or video* generator, one entry per
    # generation request whose prompt was recoverable, in call order. The most
    # basic artifact a generative agent leaves and the only one that says what
    # it *meant* each shot to contain; it sits here with the other request
    # parameters rather than under the continuity contract it happens to be
    # checked against, because it feeds more than continuity. Music and speech
    # prompts are excluded: `brief.specs_in_prompts` counts wardrobe words in
    # here, and a score prompt can only dilute that.
    prompts: list[str] | None = None
    # One entry per generation request. NOT a set of distinct values: the
    # language-model backend once returned `[10.0]` meaning "each clip is ten
    # seconds", and summing that gave 10 s generated against 180 s delivered --
    # a phantom retiming finding. `generated_seconds` is the field to read.
    clip_durations: list[float] | None = None
    # Legacy name: requested or stated seconds, NOT measured successful output.
    # Report only. C must not cap real footage from this number.
    generated_seconds: float | None = None
    #   per_request  summed from all non-rejected POST durations (not outputs)
    #   partial_requests  only some POSTs recorded a duration; an incomplete sum
    #   stated       the log states the total outright
    #   derived      a typical clip length scaled by the generation count
    # None of these sources certifies an actual generated-footage inventory.
    generated_seconds_source: str | None = None


@dataclass
class ProducedArtifact:
    """One call -> one file. The edge the provenance graph wants.

    An HTTP trace cannot supply this: a request record carries `body_bytes`,
    not the body. A cost ledger can -- claudecode records `out` on 30 of 48
    rows -- which is why neither format dominates.
    """

    path: str
    kind: str | None = None              # image | video | audio | text
    call_position: int | None = None
    model: str | None = None


@dataclass
class ConditioningUse:
    """A generation that consumed earlier assets as reference.

    An HTTP trace never records it: a request row carries `body_bytes`, not
    the body, and the most it offers is a *rate* from request size (bodies
    split bimodally into text-only <20KB and image-attached >=200KB -- 593 of
    4746 video calls and 911 of 1477 image calls carry an image). Two other
    formats name the asset outright: a tool-call export puts it in the
    arguments (hermes: `video_generate(image_url=.../frame_ep1_hook.png)`),
    and a generation script the agent wrote lists it beside the prompt
    (claudecode: `refs: ['maya', 'victor']`). An earlier docstring here said
    no format records it, which was one more failed lookup generalised into a
    claim about the data. The stronger answer still belongs to the pixel
    channel, which can check whether the delivered face *is* the reference
    face.
    """

    target: str | None = None            # artifact being generated
    references: list[str] | None = None  # reference inputs fed in
    kind: str | None = None              # character | scene | previous_clip

    # `facts.conditioning` distinguishes three states, and the middle one is
    # the whole point of the field being a list rather than a count:
    #
    #   None   no channel to read. An HTTP trace records that a call happened,
    #          never what was in it, so it stays None and the check abstains.
    #   []     every generation call was read and none carried a reference.
    #          openclaw's 23 record their arguments in full and contain no
    #          image: that is a finding about the agent, and it fails.
    #   [...]  the edges themselves.
    #
    # Recording the middle case as None cost twice over: the report claimed
    # "the trajectory does not record references" about a trajectory that
    # does, and `needs_lm` sent the model after an answer already in hand --
    # conditioning was the largest gap across `data/`, 69 runs of 80.


@dataclass
class PlannedShot:
    """One shot the agent laid out before generating, with when it runs.

    The time span is what makes this usable rather than decorative: it maps
    onto the splice points the pixel side recovers, so "the agent said shots 1
    and 4 are the same room" becomes a statement about *delivered* shots.

    Three of five harnesses write this, in three shapes that share one thing --
    a time span followed by prose:

        hermes       SHOT 1 (0:00-0:03) INT. GLASS CONFERENCE ROOM / THROUGH GLASS
        openclaw     | Hook | 0-3s | Victor's applause under Maya's stolen slide
        opensquilla  - **S1 (0-4s, Hook)**: Victor applauded inside glass room

    `text` is kept raw. Deciding which brief location it names is expectations'
    job, because that is the layer holding the brief.
    """

    start: float
    end: float
    text: str
    # Which episode the span belongs to. Load-bearing, not decoration: a
    # package holding three episodes repeats the same spans three times
    # (0-4s Hook, 4-18s Setup, ...), so a plan keyed on the span alone merges
    # three episodes into one and hands episode 3's text to episode 1.
    episode: int | None = None
    source_line: int | None = None


@dataclass
class ContinuityContract:
    """The look the agent wrote down before generating, in its own words.

    Every other field in this checklist is a count, a parameter or a path.
    This one is *content*, and it exists because it is the only half of the
    benchmark's central information flow that the pixels cannot see:

        brief:      "Maya: navy officewear, silver pencil behind ear"
          -> did the agent write the spec down?      trajectory only, pure agent
          -> did its own prompts carry the spec?     trajectory only, pure agent
          -> did the delivered shot show it?         pixels, the model's job

    Measured over `data/harness`, prompt_01, the artifact exists in three
    shapes and one harness has none at all:

        opensquilla  creative_package.md "Characters (locked visual design)"
        openclaw     story/series_bible.md
        hermes       receipts_series/scripts/00_series_bible.md
        claudecode   embedded in series.js, the generation script it wrote
        codex        nothing -- no character appearance text anywhere

    That spread is why this is worth carrying: it separates agents before a
    single frame is decoded. Text is kept in full; matching it to the
    brief's cast belongs to expectations.py, which is the layer that has the
    brief. The checklist observes, it does not interpret.
    """

    source: str | None = None             # artifact the text was read from
    # Full selected document for existing consumers. Ordered source-bound
    # fragments are also stored in TrajectoryFacts.statements, without loss.
    text: str | None = None
    # One entry per generation request whose prompt was recoverable, in call
    # order. A bible nobody quotes into a prompt is an asset the agent built
    # and did not use -- the same failure `brief.characters_anchored` finds on
    # the image side, asked on the text side.
    # The agent's own shot breakdown, in time order. Its point is to replace a
    # guess: `visual_consistency` currently groups shots into locations by
    # clustering CLIP features at a fixed distance, because nothing told it
    # which shots share a room. The agent wrote that down before generating.
    shot_plan: list[PlannedShot] | None = None


@dataclass
class Claims:
    """What the agent said about its own run. Verifiable, never authoritative."""

    declared_deliverables: list[str] | None = None
    self_reported_status: str | None = None
    planning_artifact: str | None = None   # a plan file written before generation
    deviations: list[str] | None = None
    # Claimed source clips per episode, verified only against executed assembly
    # records bound to that episode; pixel cut counts are not source clip counts.
    declared_clips_per_episode: list[int] | None = None
    # The agent's own pass/fail verdicts on its delivery (verification.json's
    # durationOk / resolutionOk / audioOk). Claims, not facts: claudecode wrote
    # audioOk=true; whether the track carries signal is ours to measure.
    self_checks: dict[str, bool] | None = None


@dataclass
class TrajectoryFacts:
    """The full checklist. Everything optional; None means "not observable here"."""

    source_files: list[str] = field(default_factory=list)
    backend: str = ""                    # deterministic | lm:<model>@<revision>
    formats_seen: list[str] = field(default_factory=list)

    calls: CallInventory = field(default_factory=CallInventory)
    errors: ErrorProfile = field(default_factory=ErrorProfile)
    requested: RequestedParams = field(default_factory=RequestedParams)
    continuity: ContinuityContract = field(default_factory=ContinuityContract)
    claims: Claims = field(default_factory=Claims)
    produced: list[ProducedArtifact] | None = None
    conditioning: list[ConditioningUse] | None = None

    notes: list[str] = field(default_factory=list)
    # Audit metadata, excluded from scored fields by _flatten.
    lm_evidence: dict[str, list[dict]] = field(default_factory=dict)
    lm_audit: dict = field(default_factory=dict)
    # Deterministic evidence ledgers; never model-fillable checklist scalars.
    reliability_audit: dict = field(default_factory=dict)
    claim_evidence: list[dict] = field(default_factory=list)
    # Trajectory statements, with field/stage/subject/value/source. These are
    # declarations, not verified findings; retain intermediate and final ones.
    statements: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrajectoryFacts":
        """The inverse of to_dict, so a checklist stored in a report or a
        probe file can be compared again after the comparator changes."""
        def group(k, sub):
            v = d.get(k)
            if not isinstance(v, dict):
                return sub()
            allowed = {f.name for f in fields(sub)}
            return sub(**{a: b for a, b in v.items() if a in allowed})

        f = cls(source_files=list(d.get("source_files") or []),
                backend=str(d.get("backend") or ""),
                formats_seen=list(d.get("formats_seen") or []),
                calls=group("calls", CallInventory),
                errors=group("errors", ErrorProfile),
                requested=group("requested", RequestedParams),
                continuity=group("continuity", ContinuityContract),
                claims=group("claims", Claims),
                notes=list(d.get("notes") or []),
                lm_evidence=dict(d.get("lm_evidence") or {}),
                lm_audit=dict(d.get("lm_audit") or {}),
                reliability_audit=dict(d.get("reliability_audit") or {}),
                claim_evidence=list(d.get("claim_evidence") or []),
                statements=list(d.get("statements") or []))
        plan = f.continuity.shot_plan
        if isinstance(plan, list):
            f.continuity.shot_plan = [PlannedShot(**s) for s in plan
                                      if isinstance(s, dict)] or None
        prod = d.get("produced")
        if isinstance(prod, list):
            f.produced = [ProducedArtifact(**p) for p in prod if isinstance(p, dict)]
        cond = d.get("conditioning")
        if isinstance(cond, list):
            f.conditioning = [ConditioningUse(**c) for c in cond if isinstance(c, dict)]
        return f

    @property
    def observable_fields(self) -> int:
        """How much of the checklist this backend filled in.

        Travels with the facts for the same reason coverage travels with every
        score: a sparse extraction must announce itself rather than look clean.
        """
        return sum(1 for _, v in _flatten(self) if v is not None)

    @property
    def total_fields(self) -> int:
        return sum(1 for _ in _flatten(self))

    @property
    def coverage(self) -> float:
        """Non-null field fraction, not recall or an extraction success rate.

        Missing source evidence is not an extraction error. Recall needs an
        independently verified inventory of facts extractable from the source.
        """
        t = self.total_fields
        return (self.observable_fields / t) if t else 0.0


# Every dataclass group `_flatten` walks. Adding a group to TrajectoryFacts
# and forgetting it here is silent: the fields keep working, but they vanish
# from `checklist_usage` (so "the checklist is fully used" stops counting
# them), from `compare()` (so the LM backend is never scored on them), and
# from the orphan test, which checks fields have consumers and not the
# reverse. `ContinuityContract` was invisible that way for exactly one
# session. test_trajectory asserts this tuple covers TrajectoryFacts.
_GROUPS = (CallInventory, ErrorProfile, RequestedParams, Claims,
           ContinuityContract)


def _flatten(facts: TrajectoryFacts):
    for f in fields(facts):
        v = getattr(facts, f.name)
        if isinstance(v, _GROUPS):
            for sub in fields(v):
                yield f"{f.name}.{sub.name}", getattr(v, sub.name)
        elif f.name in ("produced", "conditioning"):
            yield f.name, v


def node_index() -> dict[str, list[str]]:
    """Which checklist fields each tree node draws on.

    Derived from CONSUMERS and the tree, never written out by hand: a leaf that
    gains a source key, or a field that gains a consumer, shows up here without
    anyone remembering to edit a second table. A field feeding two nodes
    appears under both, which is the honest shape and the reason the
    dataclasses above are not themselves tree-shaped.
    """
    from stage5_scoring import QUALITY
    from stage5_scoring import TRAJECTORY                     # local: no cycle
    from stage5_scoring.registry import COMPLETENESS_INPUTS

    of_key: dict[str, str] = {k: "1.1" for k in COMPLETENESS_INPUTS}
    for cat in (*QUALITY, *TRAJECTORY):
        for leaf in cat.leaves:
            for key in leaf.sources:
                of_key[key] = leaf.id

    out: dict[str, list[str]] = {}
    for field_name, consumers in CONSUMERS.items():
        nodes = sorted({of_key[c] for kind, c in consumers
                        if kind == "check" and c in of_key})
        # Every check a field feeds is scored or a completeness input, so a
        # field with no check consumer is flow-only. (There is no "diagnostic"
        # bucket to fall into: registry.DIAGNOSTIC is closed.)
        for node in nodes or ["flows"]:
            out.setdefault(node, []).append(field_name)
    return {k: out[k] for k in sorted(out)}


def by_node(facts: TrajectoryFacts) -> dict:
    """The filled checklist, laid out by tree node, for the report.

    The same values as `to_dict()`, read in the order the rubric reads them.
    A field holding objects rather than numbers -- `produced`, `conditioning`,
    `continuity.shot_plan` -- appears as its length here; the full lists stay
    in `facts`, which goes through `asdict`.

    The rule is by shape and not by field name on purpose: naming the two known
    lists left `shot_plan` handing raw `PlannedShot` instances to the report
    writer, and `json.dumps` then failed the whole sweep on every run whose
    agent wrote a shot plan.
    """
    values = dict(_flatten(facts))
    out: dict[str, dict] = {}
    for node, names in node_index().items():
        block = {}
        for name in names:
            v = values.get(name, getattr(facts, name, None))
            if isinstance(v, list) and v and is_dataclass(v[0]):
                v = len(v)
            block[name] = v
        out[node] = block
    return out


def compare(a: TrajectoryFacts, b: TrajectoryFacts) -> dict:
    """Field-by-field agreement between two backends.

    Run over the trajectories both can read, this is the language model's
    measured accuracy -- the number that has to exist before it is trusted on
    the formats only it can read.
    """
    da, db = dict(_flatten(a)), dict(_flatten(b))
    agree, differ, only_a, only_b, both_none = [], [], [], [], []
    approximate = []
    for k in sorted(set(da) | set(db)):
        va, vb = da.get(k), db.get(k)
        if va is None and vb is None:
            both_none.append(k)
        elif va is None:
            only_b.append(k)
        elif vb is None:
            only_a.append(k)
        elif not _exactly_comparable(va):
            # Free text and object lists are where the model earns its place
            # and where `==` is guaranteed to say "differ": 40 parsed shots
            # against 36 read ones, an 8000-char truncation against a
            # paraphrase. Scoring those as disagreements drove the headline
            # rate to 0.25 while the two backends were describing the same
            # run. They get a size comparison and stay out of the rate.
            approximate.append({"field": k, "a": _size(va), "b": _size(vb)})
        elif _same(k, va, vb):
            agree.append(k)
        else:
            differ.append({"field": k, "a": va, "b": vb})
    comparable = len(agree) + len(differ)
    return {
        "agreement_rate": (len(agree) / comparable) if comparable else None,
        "n_comparable": comparable,
        "agree": agree, "differ": differ, "approximate": approximate,
        "only_a": only_a, "only_b": only_b, "both_none": both_none,
    }


_PATH_SEP = ("/", "\\")


def _same(key: str, va, vb) -> bool:
    """Equal after normalisation; measured quantities equal within rounding.

    A summed ledger says 11.4223 and the run's own report says $11.42; a
    count of 18 against 17 is a real disagreement. So floats get a small
    tolerance and integers stay exact.
    """
    if isinstance(va, bool) or isinstance(vb, bool):
        return va == vb
    if isinstance(va, (int, float)) and isinstance(vb, (int, float)) \
            and (isinstance(va, float) or isinstance(vb, float)):
        return abs(va - vb) <= max(0.01, 0.005 * max(abs(va), abs(vb)))
    return _normalised(key, va) == _normalised(key, vb)


def _normalised(key: str, v):
    """The form in which two correct readings of one fact coincide.

    A relative path and a bare filename name the same bible; `image` and
    `image_gen` are one bucket; "OK" and "ok" are one status. The first
    probe scored hermes and openclaw at 0.00 agreement on exactly one
    comparable field -- `continuity.source`, `receipts_series/scripts/
    00_series_bible.md` against `00_series_bible.md` -- so the rate said the
    backends disagreed while they were pointing at the same file.
    """
    if isinstance(v, str):
        s = v.strip()
        if any(c in s for c in _PATH_SEP) or key.endswith((".source", ".planning_artifact")):
            return s.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].lower()
        return s.lower()
    return v


def _exactly_comparable(v) -> bool:
    """Is `==` a fair test for this value?

    True for the counts, flags and short identifiers the deterministic pass
    exists to get right. False for anything whose two renderings can be
    equally correct and textually different.
    """
    if isinstance(v, (int, float, bool)):
        return True
    if isinstance(v, str):
        return len(v) <= 120
    if isinstance(v, dict):
        return True                       # small maps: status counts, models
    return False                          # lists of objects, long text


def _size(v):
    if isinstance(v, (list, tuple, dict)):
        return len(v)
    if isinstance(v, str):
        return len(v)
    return v
