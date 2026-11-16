"""Versioned benchmark preprocessing contract and structural integrity checks.

This is a benchmark profile inspired by modular ETL and W3C PROV lineage, not
an implementation of OpenTelemetry/PROV serialization or a quality scorer.
No optional service, model, or user-provided configuration is required.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


VERSION = "checklist-input-v3"
SCHEMA_VERSION = "preprocessing-audit-1.0"
PROGRESS_TYPES = ("thinking_tokens", "token_usage", "usage_update", "heartbeat")
PROGRESS_KEYS = ("type", "subtype", "session_id", "sessionId", "uuid", "id", "ts",
                 "timestamp", "count", "tokens", "estimated_tokens", "estimated_tokens_delta",
                 "input_tokens", "output_tokens", "total_tokens", "elapsed", "elapsed_ms",
                 "sequence", "seq")
PROTECTED_KEYS = ("error", "errors", "result", "output", "message", "content", "arguments",
                  "tool_calls", "text", "data", "status", "toolName", "toolCall")
HISTORY_SHAPES = {"llm_call": ("input", "messages"),
                  "prompt.submitted": ("data", "messages"),
                  "model.completed": ("data", "messagesSnapshot")}
CONTEXT_TYPES = ("llm_call", "prompt.submitted", "model.completed", "trace.metadata", "trace.artifacts")
CONTEXT_MIN_CHARS = 1000
DATA_URI_PATTERN = r"data:(?:image|audio|video)/[\w.+-]+;base64,[A-Za-z0-9+/=]+"
CREDENTIAL_PATTERN = r"\bsk-(?:or-v1-)?[A-Za-z0-9_-]{20,}|(?i:Bearer)\s+[A-Za-z0-9._-]{25,}"
RULES = {
    "P001": {"action": "drop", "name": "pure_progress", "types": PROGRESS_TYPES,
             "allowed_keys": PROGRESS_KEYS, "protected_keys": PROTECTED_KEYS},
    "P002": {"action": "replace", "name": "encoded_media", "uri_pattern": DATA_URI_PATTERN,
             "keys": ["base64", "data when type/encoding=base64"]},
    "P003": {"action": "reference", "name": "identical_document", "match": "sha256_decoded_text"},
    "P004": {"action": "reference", "name": "input_history_replay", "shapes": HISTORY_SHAPES,
             "match": "exact_json_sorted_keys"},
    "P005": {"action": "reference", "name": "repeated_context", "types": CONTEXT_TYPES,
             "minimum_chars": CONTEXT_MIN_CHARS, "match": "exact_text"},
    "P006": {"action": "keep", "name": "unknown_or_malformed", "default": True},
    "P007": {"action": "replace", "name": "credential_redaction", "pattern": CREDENTIAL_PATTERN},
}
STAGES = ("select_sources", "decode_and_identify", "prepare_evidence", "validate_integrity",
          "chunk_all", "extract_checklist")


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def policy_manifest():
    policy = {"version": VERSION, "stages": STAGES, "rules": RULES,
              "scope": "checklist_input_only", "default_action": "keep",
              "summarize": False, "infer_facts": False, "merge_event_occurrences": False}
    # JSON round-trip gives callers an independent JSON-native manifest.
    return json.loads(json.dumps(policy, ensure_ascii=False, sort_keys=True))


def policy_hash():
    return sha256(json.dumps(policy_manifest(), ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def implementation_hash():
    root = Path(__file__).parent
    payload = b"".join(name.encode() + b"\0" + (root / name).read_bytes() for name in (
        "contract.py", "preprocess.py", "extractors.py",
        "schemas/preprocessing-audit.schema.json"))
    return hashlib.sha256(payload).hexdigest()


def summary(audit):
    """Small report view; the full manifest remains in the checklist audit."""
    if not audit:
        return {"status": "not_run"}
    return {"status": audit.get('validation', {}).get('status', 'not_validated'),
            **{k: audit.get(k) for k in ('version', 'schema_version', 'policy_sha256',
                'implementation_sha256', 'input_chars', 'output_chars', 'rule_hits',
                'source_inventory_complete')},
            "files": len(audit.get('files', [])), "read_errors": audit.get('read_errors', [])}


def validate_preparation(audit, parts):
    """Validate our own transformation, not the agent's data or declarations.

    Malformed JSON, absent task IDs and unknown content are valid inputs. The
    checks below detect implementation mistakes such as lost lines or dangling
    references. They do not assert extraction accuracy or benchmark compliance.
    """
    errors = []
    checks = []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})
        if not passed:
            errors.append(name)

    files = audit.get("files", [])
    sources = {f.get("source"): f for f in files}
    check("manifest_version", audit.get("schema_version") == SCHEMA_VERSION
          and audit.get("version") == VERSION and audit.get("policy_sha256") == policy_hash()
          and audit.get("policy") == policy_manifest())
    check("unique_source_names", len(sources) == len(files)
          and all(isinstance(s, str) and bool(s) for s in sources))
    counts = all(type(f.get(k)) is int and f[k] >= 0 for f in files
                 for k in ("input_chars", "output_chars", "input_lines"))
    check("size_accounting", counts and all(
        audit.get(k) == sum(f[k] for f in files) for k in ("input_chars", "output_chars")))
    prepared = {}
    for part in parts:
        header, _, body = part.partition("\n")
        if not header.startswith("### SOURCE: "):
            errors.append("invalid_source_header")
            continue
        source = header.removeprefix("### SOURCE: ")
        if source in prepared:
            errors.append("duplicate_prepared_source")
        prepared[source] = body
    check("output_hashes_and_sources", set(prepared) == {f['source'] for f in files if f['emitted']}
          and all(len(prepared.get(f['source'], '')) == f['output_chars']
                  and sha256(prepared.get(f['source'], '')) == f['output_sha256'] for f in files))
    coverage = True
    aliases = {}
    previous = set()
    for f in files:
        source = f['source']
        alias = f.get('duplicate_of')
        if alias is not None:
            aliases[source] = alias
            coverage &= alias in previous and sources[alias]['input_sha256'] == f['input_sha256']
        elif f['adapter'] != 'json_document':
            retained, removed = f['output_lines'], f['removed_progress']
            coverage &= (sorted(retained + removed) == list(range(1, f['input_lines'] + 1))
                         and retained == sorted(retained))
        previous.add(source)
    check("source_line_coverage_and_order", coverage)
    check("document_references", aliases == audit.get('duplicate_sources'))
    definitions = {d['id']: d for d in audit.get('history_definitions', [])}
    refs_ok = True
    for f in files:
        for ref in f['replayed_messages']:
            first = ref['first']
            refs_ok &= (definitions.get(first['id']) == first and first['source'] in sources)
        for ref in f['repeated_context']:
            refs_ok &= ref['first'] in audit.get('context_definitions', [])
    check("history_and_context_references", refs_ok)
    expected_hits = {k: 0 for k in RULES}
    for f in files:
        expected = {"P001": len(f['removed_progress']), "P002": len(f['media']),
                    "P003": int('duplicate_of' in f), "P004": len(f['replayed_messages']),
                    "P005": len(f['repeated_context']), "P006": len(f['unparsed_lines']),
                    "P007": len(f.get('credential_redactions', []))}
        if f.get('rule_hits') != expected:
            errors.append('file_rule_accounting')
        for key, value in expected.items():
            expected_hits[key] += value
    check("rule_accounting", audit.get('rule_hits') == expected_hits)
    return {"status": "passed" if not errors else "failed", "checks": checks, "errors": errors}
