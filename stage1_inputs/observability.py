"""Recheck archived input before interpreting checklist gaps.

A valid JSON response is not a recall certificate. Positive facts recovered
by the independent structured reader repair empty checklist slots; absence
after this recheck stays 'not observed', never proof of an unused method.
No model, media detector or network call is made here.
"""
from copy import deepcopy
from dataclasses import fields
from pathlib import Path

import stage1_inputs.extractors as extractors
from stage1_inputs.schema import TrajectoryFacts


METHOD_FIELDS = ("conditioning", "continuity.text", "continuity.shot_plan", "requested.prompts")


def _empty(value):
    return value is None or value == [] or value == "" or value == {}


def _slots(facts):
    for name in ("calls", "errors", "requested", "claims", "continuity"):
        group = getattr(facts, name)
        for f in fields(group):
            yield f"{name}.{f.name}", group, f.name
    for name in ("produced", "conditioning"):
        yield name, facts, name


def recheck_facts(root: Path, facts: TrajectoryFacts) -> dict:
    root = Path(root)
    audit = facts.lm_audit
    errors = []
    total = audit.get("chunks_total", 0)
    if total and (audit.get("chunks_valid") != total or not audit.get("input_complete")):
        errors.append("incomplete_extraction")
    prior = audit.get("preprocessing", {})
    if prior.get("source_inventory_complete") is False or prior.get("read_errors"):
        errors.append("incomplete_source_inventory")
    fresh = extractors.deterministic_extract(root, root.parent)
    prepared = TrajectoryFacts()
    extractors._lm_parts(root, prepared)
    current = prepared.lm_audit["preprocessing"]
    if not current.get("source_inventory_complete") or current.get("read_errors"):
        errors.append("source_read_failed")
    old_files = {f["source"]: f.get("raw_sha256") for f in prior.get("files", [])}
    new_files = {f["source"]: f.get("raw_sha256") for f in current.get("files", [])}
    changed = sorted(k for k in old_files.keys() | new_files.keys()
                     if old_files.get(k) != new_files.get(k)) if old_files else []
    if changed:
        errors.append("extraction_sources_changed")
    recovered = []
    values = {key: getattr(obj, name) for key, obj, name in _slots(fresh)}
    for key, obj, name in _slots(facts):
        old, new = getattr(obj, name), values[key]
        if _empty(old) and not _empty(new):
            # Never repair source changes using only one parser: the complete
            # extraction must first be refreshed for the new input.
            if not errors:
                setattr(obj, name, deepcopy(new))
                recovered.append(key)
    if recovered:
        for row in fresh.claim_evidence:
            if row not in facts.claim_evidence:
                facts.claim_evidence.append(deepcopy(row))
        for row in fresh.statements:
            if row not in facts.statements:
                facts.statements.append(deepcopy(row))
    vals = {key: getattr(obj, name) for key, obj, name in _slots(facts)}
    states = {key: ("recovered_extraction_omission" if key in recovered else
                    "observed" if not _empty(vals.get(key)) else
                    "extraction_unverified" if errors or not old_files else
                    "not_observed_after_source_recheck") for key in METHOD_FIELDS}
    result = {"passed": not errors, "errors": errors, "recovered_fields": recovered,
              "method_fields": states, "changed_sources": changed,
              "source_inventory_verified": bool(old_files) and not errors,
              "sources": [{"source": k, "sha256": v} for k, v in sorted(new_files.items())],
              "semantic_absence_proven": False,
              "scope": "Raw source inventory/hashes and structured reread. Unobserved is not proof of non-use; method diagnostics never affect scores."}
    facts.lm_audit["observability_recheck"] = result
    return result
