"""Agent-authored claims checked against the same file/episode and property."""
from __future__ import annotations

from pathlib import Path
from pathlib import PureWindowsPath
import re
import shlex
import hashlib

from shared.types import Measurement
from shared.types import Verdict
from shared.types import Scale


def measurement(key, verdict, reason="", **detail):
    return Measurement(key=key, verdict=verdict, scale=Scale.BOOLEAN,
                       value=None, reason=reason, detail=detail)


def resolve_claim(root: Path, raw: str) -> tuple[Path | None, str]:
    """Run-local resolution, preserving explicit paths and ambiguity.

    Historical absolute paths can relocate by a unique multi-component suffix.
    An arbitrary same-basename file must not satisfy a different relative path.
    """
    root = root.resolve()
    raw = str(raw).replace("\\", "/").strip()
    if not raw or any(c in raw for c in "*?[]{}") or "://" in raw:
        return None, "unknown"
    migrated = PureWindowsPath(raw).is_absolute() or raw.startswith("/") or raw.startswith("$")
    path = Path(raw) if Path(raw).is_absolute() else root / raw
    try:
        resolved = path.resolve()
        if resolved.is_relative_to(root) and resolved.is_file():
            return resolved, "found"
        if not migrated and not resolved.is_relative_to(root):
            return None, "unknown"
        files = [p.resolve() for p in root.rglob("*") if p.is_file()]
    except OSError:
        return None, "unknown"
    parts = [p for p in raw.split("/") if p]
    if not migrated and len(parts) > 1:
        # Archiving may add a container directory (e.g. artifacts/). Preserve
        # the whole declared relative path, rather than matching its basename.
        relative_parts = parts
        roots = [i for i, part in enumerate(parts[:-1]) if part.casefold() == root.name.casefold()]
        if roots:
            relative_parts = parts[roots[-1] + 1:]
        suffix = '/'.join(relative_parts).casefold()
        if len(relative_parts) > 1:
            hits = [p for p in files if p.as_posix().casefold().endswith('/' + suffix)]
            if hits:
                return (hits[0], 'found') if len(hits) == 1 else (None, 'unknown')
        # An archive can replace the original container directory. A matching
        # parent + filename is evidence of possible relocation, not proof of
        # either delivery or absence. Never let basename-only matches rescue a
        # genuinely missing explicit path.
        if len(relative_parts) > 2:
            tail = '/'.join(relative_parts[-2:]).casefold()
            archive = root / 'artifacts'
            if any(p.is_relative_to(archive) and p.as_posix().casefold().endswith('/' + tail)
                   for p in files):
                return None, 'unknown'
        return None, 'missing'
    for n in range(len(parts), 0, -1):
        if migrated and n == 1 and len(parts) > 1:
            # Only a basename survived relocation: insufficient to prove which
            # of the old directories the file belonged to.
            return None, "unknown"
        suffix = "/".join(parts[-n:]).casefold()
        hits = [p for p in files if p.as_posix().casefold().endswith("/" + suffix)]
        if hits:
            return (hits[0], "found") if len(hits) == 1 else (None, "unknown")
    return None, "missing" if not migrated else "unknown"


def _literal_file_claim(raw: str) -> bool:
    """Absence only contradicts a concrete filename, not a delivery summary.

    Call after resolution: a real file with spaces, punctuation or a range in
    its name remains valid. Do not expand prose into invented file bindings.
    """
    raw = str(raw).replace("\\", "/").strip()
    name = raw.rsplit("/", 1)[-1]
    if not re.fullmatch(r".+\.[A-Za-z][A-Za-z0-9]{0,9}", name):
        return False
    if re.search(r"\||,\s|\s+(?:through|to|and|[+&…]|\.{2,})\s+", raw, re.I):
        return False
    # Episode range notation, e.g. ep1-6.mp4 or cover_ep01–05.png.
    if re.search(r"(?:^|[/_\s])(?:episode|ep|e)\d+\s*(?:[-–—…]|\.{2,})\s*(?:episode|ep|e)?\d+", raw, re.I):
        return False
    if re.search(r"(?:^|[/_\s])(?:episode|ep|e)0*[NX]+(?=[/_.-]|$)", raw, re.I):
        return False
    if re.search(r"(?:^|[/_\s])X{2,}(?=[/_.-]|$)", raw, re.I):
        return False
    return True


def check_deliverables(facts, root: Path | None) -> Measurement:
    key = "claims.deliverables_consistent"
    declared = facts.claims.declared_deliverables
    if declared is None:
        return measurement(key, Verdict.ABSTAIN, "轨迹未声明交付物")
    if root is None or not declared:
        return measurement(key, Verdict.ABSTAIN, "缺少声明文件的独立核验结果")
    evidence = []
    for raw in dict.fromkeys(declared):
        hit, state = resolve_claim(root, raw)
        if state == "missing" and not _literal_file_claim(raw):
            state = "unknown"
        evidence.append({"claim": raw, "state": state, "resolved": str(hit) if hit else None})
    missing = [e["claim"] for e in evidence if e["state"] == "missing"]
    if missing:
        return measurement(key, Verdict.FAIL, "声明交付的文件不存在：" + "、".join(missing), evidence=evidence)
    if any(e["state"] != "found" for e in evidence):
        return measurement(key, Verdict.ABSTAIN, "声明路径无法唯一定位，不能判为一致或虚报", evidence=evidence)
    return measurement(key, Verdict.PASS, evidence=evidence)


def check_status(facts, n_delivered: int, n_required: int | None) -> Measurement:
    key = "claims.status_consistent"
    status = (facts.claims.self_reported_status or "").lower().strip()
    if len({r["status"] for r in facts.claim_evidence if "status" in r}) > 1:
        return measurement(key, Verdict.ABSTAIN, "最终状态声明相互冲突，缺少明确更正顺序")
    if not status:
        return measurement(key, Verdict.ABSTAIN, "轨迹未自报最终全局状态")
    complete = status in {"ok", "completed", "complete", "success", "succeeded", "done", "finished", "0"}
    if complete:
        if n_delivered == 0 or (n_required is not None and n_delivered < n_required):
            return measurement(key, Verdict.FAIL,
                               f"自报全部完成，但只定位到 {n_delivered}/{n_required or '?'} 集")
        if n_required is None:
            return measurement(key, Verdict.ABSTAIN, "未知目标集数，无法验证全部完成")
        return measurement(key, Verdict.PASS)
    if status == "delivered":
        return measurement(key, Verdict.PASS if n_delivered else Verdict.FAIL,
                           "" if n_delivered else "声称已交付，但没有成片")
    if status in {"failed", "error", "timeout", "crashed"}:
        return measurement(key, Verdict.PASS, "未声称完成，不处罚保守自评")
    # 'Partial' may mean a script/asset is done; it does not assert a video.
    if status in {"partial", "partially", "incomplete"}:
        return measurement(key, Verdict.ABSTAIN, "部分完成未说明交付范围，不推断已交付成片")
    return measurement(key, Verdict.ABSTAIN, f"自报状态「{status}」无法归类")


def episode_matches(record: dict, item: dict, root: Path) -> bool:
    if record.get("file"):
        resolved, state = resolve_claim(root, record["file"])
        return state == "found" and resolved == Path(item["video"]).resolve()
    ep = record.get("episode")
    return ep is not None and item.get("episode") == ep


def executed_assemblies(root: Path, calls) -> dict[str, dict]:
    """Bind a successful ffmpeg concat to its output and recorded list contents.

    A list merely found on disk is a plan, not proof it was used. Its contents
    must occur in a preceding successful write event and still match on disk.
    Multiple writes to the same output invalidate an earlier binding unless the
    final write is another fully verified assembly.
    """
    from stage4_checks.process.reliability import assembly_target
    from stage4_checks.process.reliability import result_state
    from stage4_checks.process.reliability import output_target
    written, outputs = {}, {}
    for c in calls:
        produced = output_target(c.args)
        if produced and c.ok is True:
            hit, _ = resolve_claim(root, produced.removeprefix("output:"))
            if hit is not None:
                outputs.pop(str(hit).casefold(), None)
        if c.ok is True:
            raw = c.args.get("path", c.args.get("file_path"))
            content = c.args.get("content")
            if isinstance(raw, str) and isinstance(content, str) and re.search(r"write|create", c.name, re.I):
                p, state = resolve_claim(root, raw)
                if p is not None:
                    written[str(p)] = content
        target = assembly_target(c.args)
        if not target:
            continue
        out, state = resolve_claim(root, target.removeprefix("output:"))
        if out is None:
            continue
        outputs.pop(str(out).casefold(), None)
        if result_state(c.payload, explicit_ok=c.ok) != "success":
            continue
        command = c.args.get("command", c.args.get("cmd", ""))
        tokens = [x.strip("\"'") for x in shlex.split(command, posix=False)]
        if "-filter_complex" in tokens or "-ss" in tokens or "-t" in tokens:
            continue
        lists = [tokens[i + 3] for i in range(len(tokens) - 3)
                 if tokens[i:i + 3] == ["-f", "concat", "-i"]]
        # ffmpeg commonly places '-safe 0' between format and input.
        if not lists:
            lists = [tokens[i + 5] for i in range(len(tokens) - 5)
                     if tokens[i:i + 5] == ["-f", "concat", "-safe", "0", "-i"]]
        if len(lists) != 1:
            continue
        manifest, state = resolve_claim(root, lists[0])
        if manifest is None or str(manifest) not in written:
            continue
        snapshot = written[str(manifest)]
        try:
            if manifest.read_text(encoding="utf-8") != snapshot:
                continue
        except (OSError, UnicodeError):
            continue
        sources = []
        for line in snapshot.splitlines():
            if not line.strip() or line.startswith("#") or line == "ffconcat version 1.0":
                continue
            match = re.fullmatch(r"file\s+(['\"])(.*?)\1", line.strip())
            if not match:
                sources = []
                break
            raw = match[2]
            local = (manifest.parent / raw).resolve()
            if local.is_relative_to(root.resolve()) and local.is_file():
                sources.append(str(local))
            else:
                hit, state = resolve_claim(root, raw)
                if hit is None:
                    sources = []
                    break
                sources.append(str(hit))
        if sources:
            outputs[str(out).casefold()] = {"verified": True, "clip_count": len(sources),
                                 "sources": sources, "execution": c.call_id or c.position,
                                 "manifest": str(manifest),
                                 "manifest_sha256": hashlib.sha256(snapshot.encode()).hexdigest()}
    return outputs


def check_episode_claims(facts, root: Path, episodes: list[dict]) -> list[Measurement]:
    """Compare scoped claims before any cross-episode worst-of reduction.

    Clip counts require independently recorded successful assembly bindings;
    visual cuts are deliberately not accepted as a clip inventory.
    """
    checks = {"audioOk": "spec.audio_track", "durationOk": "spec.duration",
              "resolutionOk": "spec.resolution"}
    rows, contradictions, unknown = [], [], []
    structures, structure_unknown, structure_bad = [], [], []
    records = list(facts.claim_evidence)
    if not any("clips" in r for r in records) and facts.claims.declared_clips_per_episode:
        records += [{"episode": i + 1, "clips": n, "source": "claims.declared_clips_per_episode"}
                    for i, n in enumerate(facts.claims.declared_clips_per_episode)]
    values = {}
    for record in records:
        matches = [e for e in episodes if episode_matches(record, e, root)]
        if len(matches) != 1:
            continue
        props = {**(record.get("self_checks") or {}),
                 **{k: record[k] for k in ("clips", "duration") if k in record}}
        for key, value in props.items():
            values.setdefault((matches[0]["video"], key), set()).add(value)
    for record in records:
        if not (record.get("self_checks") or "clips" in record or "duration" in record):
            continue
        matches = [e for e in episodes if episode_matches(record, e, root)]
        if len(matches) != 1:
            unknown.append({"claim": record, "reason": "episode not uniquely measured"})
            if "clips" in record:
                structure_unknown.append(record)
            continue
        ep = matches[0]
        for claim, value in (record.get("self_checks") or {}).items():
            if type(value) is not bool or claim not in checks:
                continue
            m = ep.get("measurements", {}).get(checks[claim])
            row = {"source": record.get("source"), "video": ep["video"], "claim": claim,
                   "claimed": value, "measured": m.verdict.value if m else None}
            rows.append(row)
            if len(values.get((ep["video"], claim), set())) > 1:
                unknown.append({**row, "reason": "conflicting agent claims without revision order"})
                continue
            if value is False:
                continue
            if m is None or m.verdict in (Verdict.ABSTAIN, Verdict.NOT_APPLICABLE):
                unknown.append(row)
            elif m.verdict in (Verdict.FAIL, Verdict.VOID):
                contradictions.append(row)
        if "duration" in record:
            actual = ep.get("duration")
            row = {"source": record.get("source"), "video": ep["video"], "claim": "duration",
                   "claimed": record["duration"], "measured": actual}
            rows.append(row)
            if actual is None or len(values.get((ep["video"], "duration"), set())) > 1:
                unknown.append(row)
            elif record["duration"] > actual + 1.0:
                contradictions.append(row)
        if "clips" in record:
            binding = ep.get("assembly")
            if not binding or not binding.get("verified") or len(values.get((ep["video"], "clips"), set())) > 1:
                structure_unknown.append(record)
            else:
                row = {"video": ep["video"], "claimed": record["clips"],
                       "measured": binding["clip_count"], "source": record.get("source"),
                       "assembly": binding}
                structures.append(row)
                if row["claimed"] != row["measured"]:
                    structure_bad.append(row)
    result = []
    if rows or unknown:
        verdict = Verdict.FAIL if contradictions else Verdict.ABSTAIN if unknown else Verdict.PASS
        result.append(measurement("claims.deliverables_consistent", verdict,
                                  "逐集自述与该集实测矛盾" if contradictions else
                                  "部分逐集声明尚无法核验" if unknown else "",
                                  scoped_checks=rows, contradictions=contradictions, unknown=unknown))
    verdict = Verdict.FAIL if structure_bad else Verdict.ABSTAIN if structure_unknown or not structures else Verdict.PASS
    result.append(measurement("claims.structure_consistent", verdict,
                              "声明片段数与已执行装配记录矛盾" if structure_bad else
                              "缺少按集绑定的已执行装配证据；像素切点不证明源片段数" if verdict is Verdict.ABSTAIN else "",
                              assemblies=structures, unknown=structure_unknown))
    return result
