"""Execution-trace normalisation. Structure only -- never semantics.

The process dimension asks three questions of the trace, and all three are
answered from the *shape* of the event stream rather than from anything an
agent wrote about itself:

    error rate                4xx/5xx over all API calls, infrastructure excluded
    plan before generation    did a reasoning call precede the first media call
    output dir compliance     did the artefacts land in one place

Everything else the process rubric used to ask -- "was the specified skill
invoked", "were its steps complete", "was the input pre-processed" -- was
deleted (EVAL_DESIGN.md 5.1). Those needed semantics, and two of the three stop
existing once the composition-skill scaffolding is removed.

Trace formats in `data/` are HTTP-level (`trace.jsonl` from `_runner/tracehook`)
rather than tool-call-level, so classification keys off the endpoint. That is a
harness-independent signal: an image endpoint is an image endpoint regardless of
which agent framework called it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

# Endpoint -> what kind of work the call represents. Matched against the URL,
# so it survives any agent framework sitting on top.
REASONING = re.compile(r"/(chat/completions|responses|messages)\b", re.I)
IMAGE_GEN = re.compile(r"/(images|image|generations|seedream|edit)\b", re.I)
VIDEO_GEN = re.compile(r"/(videos|video|generate_video|seedance|kling)\b", re.I)
AUDIO_GEN = re.compile(r"/(audio|speech|tts|asr|transcriptions)\b", re.I)
TELEMETRY = re.compile(r"(telemetry|/system_stats|/uploads)\b", re.I)

# Failures the agent did not cause. Mirrors execution_error_analysis.py's
# is_ignorable_non_program_error: credentials and rate limits are the harness
# operator's problem, and counting them would rank agents by whose key expired.
from stage4_checks.process.reliability import INFRA_STATUS


@dataclass
class Call:
    seq: int
    url: str = ""
    status: int | None = None
    kind: str = "other"          # reasoning | image | video | audio | poll | telemetry | other
    elapsed_ms: float | None = None
    ts: float | None = None
    method: str = ""
    duration: float | None = None   # requested clip length, on generation POSTs

    @property
    def failed(self) -> bool:
        return self.status is not None and self.status >= 400

    @property
    def infra(self) -> bool:
        return self.status in INFRA_STATUS


@dataclass
class TraceSummary:
    found: bool = False
    source: str = ""
    n_calls: int = 0
    n_media: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    n_failed: int = 0
    n_failed_agent: int = 0      # failures excluding infrastructure
    n_failed_infra: int = 0
    error_rate: float | None = None
    plan_before_generation: bool | None = None
    first_media_seq: int | None = None
    first_reasoning_seq: int | None = None
    status_counts: dict[str, int] = field(default_factory=dict)
    note: str = ""


_POLL = re.compile(r"/(videos|video|jobs|tasks)/[A-Za-z0-9_-]{6,}(\?|$)", re.I)


def _classify(url: str, method: str = "") -> str:
    """Endpoint + method -> what kind of work the call is.

    Method matters. The first version classified on URL alone and counted
    every `GET /videos/<id>` status poll as a video generation: videoclaw's
    342 "generations" were 45 POSTs and 297 polls, vimax's 1214 were 111 and
    1103. Across `data/` there are 789 generation POSTs against 5944 polls --
    a 7.5x inflation that made every downstream ratio (yield, waste, calls
    per delivered second) meaningless.
    """
    if TELEMETRY.search(url):
        return "telemetry"
    if _POLL.search(url) or (method.upper() == "GET" and VIDEO_GEN.search(url)):
        return "poll"
    if VIDEO_GEN.search(url):
        return "video"
    if IMAGE_GEN.search(url):
        return "image"
    if AUDIO_GEN.search(url):
        return "audio"
    if REASONING.search(url):
        return "reasoning"
    return "other"


def _as_int(v) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _as_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_calls(run_root: Path) -> tuple[list[Call], str]:
    """Read whatever trace the harness left. Missing trace is not an error.

    Ordering is by wall-clock timestamp, with file line order as the fallback
    and tiebreaker. **Not** by the trace's own `seq`: the tracehook numbers per
    process, and a run that forks -- openmontage/prompt_01 has five pids --
    restarts the counter, so five different calls all claim `seq 2`. Sorting by
    seq interleaved those into an order that never happened, which flipped
    `plan_before_generation` to False for every openmontage run.
    """
    p = run_root / "trace.jsonl"
    if not p.is_file():
        return [], ""
    calls: list[Call] = []
    for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        url = str(d.get("url") or "")
        if not url:
            continue                      # tracehook_installed and friends
        method = str(d.get("method") or "")
        calls.append(Call(
            seq=i,                        # file line order: globally monotone
            url=url,
            status=_as_int(d.get("status")),
            kind=_classify(url, method),
            elapsed_ms=_as_float(d.get("elapsed_ms")),
            ts=_as_float(d.get("ts")),
            method=method,
            duration=_as_float(d.get("duration")),
        ))
    calls.sort(key=lambda c: (c.ts if c.ts is not None else float("inf"), c.seq))
    for order, c in enumerate(calls):     # renumber so seq means position again
        c.seq = order
    return calls, str(p)


def summarize(run_root: Path) -> TraceSummary:
    calls, src = load_calls(run_root)
    if not calls:
        return TraceSummary(found=False, note="没有 trace.jsonl，过程维度无法测量")

    counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for c in calls:
        counts[c.kind] = counts.get(c.kind, 0) + 1
        status_counts[str(c.status)] = status_counts.get(str(c.status), 0) + 1

    scored = [c for c in calls if c.kind != "telemetry" and c.status is not None]
    failed = [c for c in scored if c.failed]
    infra = [c for c in failed if c.infra]
    agent_failed = [c for c in failed if not c.infra]
    rate = (len(agent_failed) / len(scored)) if scored else None

    media = [c for c in calls if c.kind in ("image", "video", "audio")]
    reasoning = [c for c in calls if c.kind == "reasoning"]
    first_media = media[0].seq if media else None
    first_reason = reasoning[0].seq if reasoning else None
    plan_first: bool | None = None
    note = ""
    if first_media is None:
        note = "没有任何媒体生成调用"
    elif not reasoning:
        # Zero reasoning calls anywhere in the trace means this trace does not
        # capture the reasoning channel at all -- openmontage routes 391 of its
        # 416 calls to media endpoints and never hits a chat endpoint. Absence
        # of the whole category is an instrument limit, not a verdict on the
        # agent, so this abstains rather than scoring zero.
        note = "trace 中没有任何推理调用，规划步骤不可观测"
    else:
        plan_first = first_reason < first_media

    return TraceSummary(
        found=True, source=src, n_calls=len(calls), n_media=len(media),
        counts=counts, n_failed=len(failed), n_failed_agent=len(agent_failed),
        n_failed_infra=len(infra), error_rate=rate,
        plan_before_generation=plan_first,
        first_media_seq=first_media, first_reasoning_seq=first_reason,
        status_counts=status_counts, note=note,
    )


# --------------------------------------------------------------------------
# Output-directory compliance. Filesystem, not trace -- the question is where
# the artefacts landed, and the artefacts are the ground truth for that.
# --------------------------------------------------------------------------
MEDIA_EXT = frozenset({".mp4", ".mov", ".png", ".jpg", ".jpeg", ".wav", ".mp3",
                       ".m4a", ".aac", ".webm", ".srt", ".ass"})
