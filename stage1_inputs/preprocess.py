"""Lossless-for-checklist input preparation; no extraction or verdicts.

Keep unknown content. Recognised credentials are redacted; pure progress and
encoded media are removed. Identical documents share a body, but event occurrences are
never merged: repeated requests may be retries or separate production tasks.
"""
from __future__ import annotations

import hashlib
import json
import re

from stage1_inputs.contract import VERSION
from stage1_inputs.contract import SCHEMA_VERSION
from stage1_inputs.contract import PROGRESS_TYPES
from stage1_inputs.contract import PROGRESS_KEYS
from stage1_inputs.contract import PROTECTED_KEYS
from stage1_inputs.contract import HISTORY_SHAPES
from stage1_inputs.contract import CONTEXT_TYPES
from stage1_inputs.contract import CONTEXT_MIN_CHARS
from stage1_inputs.contract import DATA_URI_PATTERN
from stage1_inputs.contract import CREDENTIAL_PATTERN
from stage1_inputs.contract import RULES
from stage1_inputs.contract import policy_manifest
from stage1_inputs.contract import policy_hash
from stage1_inputs.contract import implementation_hash
from stage1_inputs.contract import validate_preparation


_DATA_URI = re.compile(DATA_URI_PATTERN)
_CREDENTIAL = re.compile(CREDENTIAL_PATTERN)
_PROGRESS = set(PROGRESS_TYPES)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class InputPreprocessor:
    def __init__(self):
        self.audit = {"version": VERSION, "schema_version": SCHEMA_VERSION,
                      "policy": policy_manifest(), "policy_sha256": policy_hash(),
                      "implementation_sha256": implementation_hash(),
                      "input_chars": 0, "output_chars": 0,
                      "files": [], "duplicate_sources": {}, "read_errors": [],
                      "source_inventory_complete": True}
        self._bodies = {}
        self._history = {}
        self._shared_text = {}

    def prepare(self, source, text, *, raw_bytes=None, decoding="utf-8"):
        """Return a source-labelled part and retain an audit for every file."""
        record = {"source": source, "input_chars": len(text),
                  "raw_sha256": hashlib.sha256(raw_bytes if raw_bytes is not None else text.encode('utf-8')).hexdigest(),
                  "raw_bytes": len(raw_bytes if raw_bytes is not None else text.encode('utf-8')),
                  "decoding": decoding, "input_lines": len(text.splitlines()),
                  "adapter": "lines", "unparsed_lines": [],
                  "input_sha256": digest(text), "removed_progress": [],
                  "media": [], "output_lines": [], "replayed_messages": [],
                  "repeated_context": [], "credential_redactions": []}
        self.audit["files"].append(record)
        self.audit["input_chars"] += len(text)
        previous = self._bodies.get(record["input_sha256"])
        if previous is not None:
            self.audit["duplicate_sources"][source] = previous
            record["duplicate_of"] = previous
            body = f"[IDENTICAL SOURCE: {previous}; same complete text, not an additional event inventory]"
        else:
            self._bodies[record["input_sha256"]] = source
            body = self._clean(text, record)
        record["output_chars"] = len(body)
        record["output_sha256"] = digest(body)
        record["emitted"] = bool(body.strip())
        record["rule_hits"] = {
            "P001": len(record['removed_progress']), "P002": len(record['media']),
            "P003": int('duplicate_of' in record), "P004": len(record['replayed_messages']),
            "P005": len(record['repeated_context']), "P006": len(record['unparsed_lines']),
            "P007": len(record['credential_redactions'])}
        self.audit["output_chars"] += len(body)
        return f"### SOURCE: {source}\n{body}" if body.strip() else ""

    def finalize(self, parts):
        """Called automatically before any prepared text is sent to the LLM."""
        self.audit['history_definitions'] = list(self._history.values())
        self.audit['context_definitions'] = list(self._shared_text.values())
        self.audit['rule_hits'] = {key: sum(f['rule_hits'][key] for f in self.audit['files'])
                                   for key in RULES}
        self.audit['validation'] = validate_preparation(self.audit, parts)
        if self.audit['validation']['status'] != 'passed':
            raise ValueError('Preprocessing integrity error: ' + ', '.join(self.audit['validation']['errors']))
        return self.audit

    @staticmethod
    def _noise(row):
        if not isinstance(row, dict):
            return False
        # A progress-shaped record carrying a result/error is still evidence.
        if any(k in row for k in PROTECTED_KEYS):
            return False
        kind = str(row.get("type", "")).lower()
        subtype = str(row.get("subtype", "")).lower()
        return (set(row) <= set(PROGRESS_KEYS) and
                (kind in _PROGRESS or (kind == "system" and subtype in _PROGRESS)))

    def _history_input(self, row, record, line):
        """Collapse only exact copies inside provider input-history arrays.

        Output events and repeated tool calls outside these replay envelopes
        are untouched. A unique message is kept even without a message event.
        """
        if not isinstance(row, dict):
            return row
        shape = HISTORY_SHAPES.get(row.get("type"))
        if shape is None:
            return row
        container, key = shape
        request = row.get(container)
        if not isinstance(request, dict) or not isinstance(request.get(key), list):
            return row
        messages, introduced = [], {}
        for index, message in enumerate(request[key]):
            body = json.dumps(message, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            sha = digest(body)
            location = f"line:{line}.{container}.{key}[{index}]"
            previous = self._history.get(sha)
            if previous is None:
                entry = {"id": f"h{len(self._history) + 1}", "source": record["source"],
                         "location": location, "sha256": sha}
                self._history[sha] = entry
                introduced[str(index)] = entry["id"]
                messages.append(message)
            else:
                messages.append({"_replayed_input_message": previous["id"]})
                record["replayed_messages"].append({"location": location,
                    "first": previous, "chars": len(body)})
        prepared = {**request, key: messages}
        if introduced:
            prepared["_input_history_ids"] = introduced
        return {**row, container: prepared}

    def _binary(self, text, record, location):
        sha = digest(text)
        record["media"].append({"location": location, "chars": len(text), "sha256": sha})
        return f"[ENCODED MEDIA OMITTED: chars={len(text)} sha256={sha}]"

    def _strings(self, text, record, location):
        def redact(match):
            record['credential_redactions'].append({'location': location})
            return '[CREDENTIAL REDACTED]'
        text = _CREDENTIAL.sub(redact, text)
        return _DATA_URI.sub(lambda m: m.group().split(",", 1)[0] + "," +
                            self._binary(m.group().split(",", 1)[1], record, location), text)

    def _value(self, value, record, location, share=False):
        if isinstance(value, dict):
            media_source = value.get("type") == "base64" or value.get("encoding") == "base64"
            return {k: self._binary(v, record, f"{location}.{k}")
                    if isinstance(v, str) and (k == "base64" or (k == "data" and media_source))
                    else self._value(v, record, f"{location}.{k}", share)
                    for k, v in value.items()}
        if isinstance(value, list):
            return [self._value(v, record, f"{location}[{i}]", share) for i, v in enumerate(value)]
        if isinstance(value, str):
            text = self._strings(value, record, location)
            if len(text) >= CONTEXT_MIN_CHARS:
                sha = digest(text)
                first = self._shared_text.get(sha)
                if first is None:
                    self._shared_text[sha] = {"source": record["source"], "location": location,
                                              "sha256": sha}
                elif share:
                    record["repeated_context"].append({"location": location, "first": first,
                                                         "chars": len(text)})
                    return (f"[REPEATED CONTEXT TEXT: source={first['source']} "
                            f"location={first['location']} sha256={sha}]")
            return text
        return value

    def _clean(self, text, record):
        # A pretty-printed JSON document is one document, not partial JSONL.
        if record["source"].lower().endswith(".json"):
            try:
                value = json.loads(text)
            except ValueError:
                pass  # malformed or mixed text remains readable evidence
            else:
                record['adapter'] = 'json_document'
                cleaned = self._value(value, record, "json")
                if not record["media"] and not record['credential_redactions']:
                    return text
                return json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"))
        output = []
        structured = record["source"].lower().endswith((".jsonl", ".log", ".txt"))
        for number, line in enumerate(text.splitlines(keepends=True), 1):
            row = None
            if structured and line.lstrip().startswith(("{", "[")):
                try:
                    row = json.loads(line)
                except ValueError:
                    record['unparsed_lines'].append(number)
            if self._noise(row):
                record["removed_progress"].append(number)
                continue
            if isinstance(row, (dict, list)):
                row = self._history_input(row, record, number)
                share = isinstance(row, dict) and row.get("type") in CONTEXT_TYPES
                cleaned = self._value(row, record, f"line:{number}", share)
                line = json.dumps(cleaned, ensure_ascii=False, separators=(",", ":")) + "\n"
            else:
                line = self._strings(line, record, f"line:{number}")
            output.append(line)
            record["output_lines"].append(number)
        return "".join(output)
