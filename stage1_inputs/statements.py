"""Store what the trajectory says, without testing whether it is true.

The statement list is the lossless extraction result. Existing scalar fields
are a compatibility view for consumers; they never replace the statements.
"""
from __future__ import annotations

import copy
import json
import re


STAGES = ("planning", "request", "generation", "assembly", "verification",
          "delivery", "unknown")


def document_statements(field, source, text, chunk_chars=8000):
    """Keep every character of a document in ordered, source-bound records.

    Offsets refer to the decoded source text and use an exclusive end. Split
    at line boundaries where possible; no stripping, summarising or grading.
    """
    if chunk_chars < 1:
        raise ValueError("chunk_chars must be positive")
    records, start = [], 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start, end)
            if boundary >= start:
                end = boundary + 1
        fragment = text[start:end]
        records.append({
            "field": field, "stage": "unknown", "subject": source,
            "value": fragment, "chunk": len(records) + 1,
            "evidence": [{"source": source, "quote": fragment,
                          "char_start": start, "char_end": end}],
        })
        start = end
    return records


def collect_statements(payload, focus, facts, chunk_index, body=None):
    """Check shape and source traceability, never the truth of a declaration.

Untraceable model output is retained in the audit, outside scoring facts.
``body=None`` supports callers that already own the statement provenance.
"""
    from stage1_inputs.schema import TrajectoryFacts
    from stage1_inputs.schema import _flatten
    known_fields = {key for key, _ in _flatten(TrajectoryFacts())}
    rows = payload.get("statements")
    if not isinstance(rows, list):
        facts.notes.append("LM statements 不是列表，无法保存记录")
        return False
    sources = {}
    if body is not None:
        for match in re.finditer(r'^### SOURCE: (.+)\n([\s\S]*?)(?=^### SOURCE: |\Z)', body, re.M):
            sources.setdefault(match[1], []).append(' '.join(match[2].split()))
    for row in rows:
        if not isinstance(row, dict) or row.get("field") not in known_fields:
            continue
        # Keep known-field declarations even if the rules already filled that
        # slot. Focus directs reading, not deletion of returned information.
        # Null means no extracted statement, not a negative finding.
        if row.get("value") is None:
            continue
        stage = row.get("stage", "unknown")
        evidence = row.get("evidence")
        refs = [copy.deepcopy(ref) for ref in evidence
                if isinstance(ref, dict) and isinstance(ref.get("source"), str)
                and isinstance(ref.get("quote"), str)] if isinstance(evidence, list) else []
        record = {
            "field": row["field"], "stage": stage if stage in STAGES else "unknown",
            "subject": row.get("subject") if isinstance(row.get("subject"), str) else None,
            "value": copy.deepcopy(row["value"]), "evidence": refs,
            "chunk": chunk_index,
        }
        if body is not None:
            invalid = []
            for ref in refs:
                quote = ref['quote']
                candidates = (quote, json.dumps(quote, ensure_ascii=False)[1:-1])
                if not quote.strip() or not any(' '.join(q.split()) in source
                        for q in candidates for source in sources.get(ref['source'], [])):
                    invalid.append(ref)
            if not refs or invalid:
                facts.lm_audit.setdefault('unverified_statements', []).append({
                    'statement': record, 'reason': 'missing_or_unmatched_source_quote',
                    'unmatched_evidence': invalid})
                facts.notes.append(f"LM 原文引用无法核验：{record['field']}，保留审计但不进入评分事实")
                continue
        # Repeated sightings in different chunks retain their provenance.
        if record not in facts.statements:
            facts.statements.append(record)
    return True


def statement_view(statements):
    """Project representable values for old consumers; never discard records.

    Stage selection is field mapping, not a verdict about a statement's truth.
    Ambiguous values and per-file observations remain available in statements.
    No sums, comparisons with the brief, file existence tests or video checks.
    """
    grouped = {}
    for row in statements:
        grouped.setdefault(row["field"], []).append(row)
    payload, evidence = {}, {}
    final_fields = {"claims.self_reported_status", "claims.declared_deliverables",
                    "claims.deviations", "claims.declared_clips_per_episode"}
    for key, rows in grouped.items():
        if key in final_fields:
            rows = [r for r in rows if r["stage"] == "delivery"]
        elif key == "claims.self_checks":
            # An episode-level claim must not become a whole-run boolean.
            rows = [r for r in rows if r["stage"] in ("verification", "delivery")
                    and r["subject"] is None]
        elif key == "claims.planning_artifact":
            rows = [r for r in rows if r["stage"] == "planning"]
        elif key.startswith("requested."):
            rows = [r for r in rows if r["stage"] in ("request", "generation")]
        if key in ("claims.self_reported_status", "claims.declared_clips_per_episode",
                   "requested.generated_seconds") or key.startswith(("calls.", "errors.")):
            rows = [r for r in rows if r["subject"] is None]
        if not rows:
            continue
        values = []
        for row in rows:
            value = row['value']
            # A per-file declaration is a single collection item, not a
            # conflicting scalar alternative to another chunk's inventory.
            # Keep the original statement untouched; normalize only this view.
            if key in ('produced', 'conditioning') and isinstance(value, dict):
                value = [value]
            if value not in values:
                values.append(value)
        if len(values) == 1:
            value = values[0]
        elif key == "continuity.text" and all(isinstance(v, str) for v in values):
            # Several extracted passages are content, not competing scalar
            # verdicts. Keep all passages visible to compatibility consumers;
            # the records retain each passage's original subject and source.
            value = "\n\n".join(values)
        elif (key not in ("requested.clip_durations", "claims.declared_clips_per_episode")
              and all(isinstance(v, list) for v in values)):
            value = []
            for items in values:
                for item in items:
                    if item not in value:
                        value.append(item)
        else:
            continue  # all versions remain in the statement list
        group, _, name = key.partition(".")
        if name:
            payload.setdefault(group, {})[name] = copy.deepcopy(value)
        else:
            payload[group] = copy.deepcopy(value)
        evidence[key] = [copy.deepcopy(ref) for row in rows for ref in row["evidence"]]
    return payload, evidence
