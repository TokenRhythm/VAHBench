"""Task-bound production outcomes, shared by HTTP and tool-event readers.

Transport acceptance is not completion. Only media-production tasks enter the
denominator; reasoning, reads, writes and polling never dilute a failed task.
An unlinked or unfinished production operation makes the inventory incomplete,
so its known subset is reported but never passed off as a complete failure rate.
"""
from __future__ import annotations

import re
import shlex
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field

INFRA_STATUS = frozenset({401, 402, 403, 407, 408, 429, *range(500, 600)})
PENDING = frozenset({"queued", "pending", "accepted", "started", "running", "processing", "in_progress"})
SUCCESS = frozenset({"ok", "success", "succeeded", "completed", "complete", "done", "finished"})
FAILED = frozenset({"error", "failed", "failure", "cancelled", "canceled", "unavailable", "not_available"})
MEDIA = re.compile(r"\.(mp4|mov|webm|mkv|png|jpe?g|webp|wav|mp3|m4a|aac)$", re.I)


def failure_owner(message: str = "", status: int | None = None) -> str:
    """One taxonomy for every instrument. Ambiguous causes stay unknown."""
    text = message.lower()
    match = re.search(r"\bhttp(?:\s+status)?\s*[:=]?\s*(\d{3})\b", text)
    code = int(match[1]) if match else status
    if code in INFRA_STATUS or re.search(
            r"rate.?limit|quota exceeded|insufficient (?:credits|balance)|"
            r"(?:api.?key|credential).*(?:missing|invalid|expired)|"
            r"provider.*(?:disabled|not configured)|command not found|"
            r"(?:connection|request|operation) timed out|content (?:filter|policy)|safety system|copyright restrictions",
            text):
        return "infrastructure"
    if code in (400, 422) or re.search(
            r"invalid (?:argument|parameter|url|option)|syntaxerror|typeerror|nameerror|"
            r"unrecognized option|unknown option|no such file|file not found|"
            r"missing required|invalid input|assertionerror", text):
        return "agent"
    return "unknown"


def result_state(payload: dict, *, transport: int | None = None,
                 explicit_ok: bool | None = None) -> str:
    status = str(payload.get("status") or payload.get("state") or "").lower()
    if status in PENDING:
        return "pending"
    if status in FAILED or payload.get("success") is False or payload.get("error"):
        return "failed"
    if transport is not None and transport >= 400:
        return "failed"
    if explicit_ok is False:
        return "failed"
    if status in SUCCESS or payload.get("success") is True:
        return "success"
    if status:
        return "pending"  # an unfamiliar state is not a terminal success
    code = payload.get("exit_code", payload.get("exitCode"))
    if isinstance(code, int) and not isinstance(code, bool):
        return "success" if code == 0 else "failed"
    if any(isinstance(payload.get(k), str) and payload[k]
           for k in ("video", "image", "audio", "output_path", "download_url")):
        return "success"
    if any(isinstance(payload.get(k), str) and MEDIA.search(payload[k]) for k in ("path", "output")):
        return "success"
    # Job handles prove submission only. An explicit synchronous tool result
    # may prove completion; an HTTP 2xx with no body cannot.
    if any(payload.get(k) for k in ("job_id", "task_id", "id")):
        return "pending"
    return "success" if explicit_ok is True and transport is None else "pending"


def output_target(args: dict) -> str | None:
    for k in ("output_path", "filename", "output_file", "save_path", "target_path", "target", "file", "out"):
        value = args.get(k)
        if isinstance(value, str) and value.strip():
            return "output:" + value.strip().replace("\\", "/").casefold()
    for k in ("shot_id", "clip_id"):
        value = args.get(k)
        if value is not None:
            ep = args.get("episode_id", args.get("episode", ""))
            return f"{k}:{ep}:{value}"
    return None


def assembly_target(args: dict) -> str | None:
    """Read a simple ffmpeg invocation, never execute or interpret shell code."""
    command = args.get("command", args.get("cmd", ""))
    if not isinstance(command, str) or not re.search(r"\bffmpeg(?:\.exe)?\b", command, re.I):
        return None
    if re.search(r"[;|\n]|&&|\$\(|`", command):
        return None
    try:
        tokens = shlex.split(command, posix=False)
    except ValueError:
        return None
    last = tokens[-1].strip("\"'") if tokens else ""
    return "output:" + last.replace("\\", "/").casefold() if MEDIA.search(last) else None


@dataclass
class Operation:
    kind: str
    target: str | None
    state: str
    owner: str = "unknown"
    aliases: list[str] = field(default_factory=list)
    source: str = ""
    poll: bool = False
    ambiguous: bool = False


def summarize_operations(operations: list[Operation]) -> dict:
    """Connect explicit task/output/job identities, including fallback tools."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    for op in operations:
        ids = ([op.target] if op.target else []) + op.aliases
        for ident in ids:
            find(ident)
        for ident in ids[1:]:
            parent[find(ident)] = find(ids[0])
    groups: dict[str, list[Operation]] = {}
    unknown = []
    for i, op in enumerate(operations):
        ids = ([op.target] if op.target else []) + op.aliases
        if not ids:
            if op.state == "failed" and op.owner == "infrastructure":
                continue
            if not op.poll:
                unknown.append({"reason": "production task has no target identity", **asdict(op)})
            continue
        groups.setdefault(find(ids[0]), []).append(op)
    tasks = []
    for identity, seq in groups.items():
        if any(x.ambiguous for x in seq):
            unknown.append({"reason": "conflicting or unordered operation evidence", "task": identity,
                            "events": [asdict(x) for x in seq]})
            continue
        if all(x.poll for x in seq):
            unknown.append({"reason": "terminal result has no submission", "task": identity})
            continue
        # A completed task is not made unfulfilled by an optional later retry
        # or a late failure from a concurrent job for the same target.
        successes = [i for i, x in enumerate(seq) if x.state == "success"]
        if successes:
            tasks.append({"task": identity, "unrecovered": False,
                          "recovered": any(x.state == "failed" and x.owner == "agent"
                                           for x in seq[:max(successes)]),
                          "events": [asdict(x) for x in seq]})
            continue
        final = seq[-1]
        if final.state == "failed" and final.owner == "infrastructure":
            if any(x.state == "failed" and x.owner == "agent" for x in seq):
                unknown.append({"reason": "infrastructure prevents terminal recovery attribution", "task": identity,
                                "events": [asdict(x) for x in seq]})
            continue
        if final.state == "pending" or (final.state == "failed" and final.owner == "unknown"):
            unknown.append({"reason": "terminal outcome or failure ownership unknown", "task": identity,
                            "events": [asdict(x) for x in seq]})
            continue
        failed = final.state == "failed"
        tasks.append({"task": identity, "unrecovered": failed,
                      "recovered": not failed and any(x.state == "failed" and x.owner == "agent" for x in seq),
                      "events": [asdict(x) for x in seq]})
    return {"version": "production-tasks-v1", "complete": not unknown,
            "n_attempts": len(tasks), "n_unrecovered": sum(t["unrecovered"] for t in tasks),
            "n_recovered": sum(t["recovered"] for t in tasks),
            "tasks": tasks, "unknown": unknown,
            "excluded_infrastructure": sum(o.state == "failed" and o.owner == "infrastructure" for o in operations)}
