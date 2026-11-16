"""Read-only credential-pattern scan. Findings never contain matched values.

Includes hidden files and Git-ignored data. This is a heuristic working-tree
scan, not proof that credentials were never published or used elsewhere.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re


SKIP_DIRS = {".git", ".venv", ".venvs", "venv", "node_modules", "__pycache__"}
BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".mp4", ".mov",
                   ".webm", ".mkv", ".mp3", ".wav", ".aac", ".m4a", ".pdf",
                   ".zip", ".gz", ".bz2", ".xz", ".7z", ".woff", ".woff2"}
PATTERNS = {
    "provider_key": re.compile(
        r"\b(?:sk-(?:or-v1-|ant-api\d\d-)?[A-Za-z0-9_-]{20,}|"
        r"AIza[\w-]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16})"),
    "bearer": re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{20,}={0,2}"),
    "credential_assignment": re.compile(
        r'''(?i)["']?\b(?:[\w-]*(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret)|password)["']?\s*[:=]\s*["']?([A-Za-z0-9_./+~=-]{16,})'''),
    "url_credential": re.compile(
        r"(?i)[?&](?:api[_-]?key|access[_-]?token|token|secret|signature)=[A-Za-z0-9_%+./~-]{16,}"),
    "private_key": re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
}


def scan(root: Path) -> dict:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError("Scan root must be an existing directory")
    findings, errors = [], []
    skipped = Counter()
    files_scanned = 0

    def walk_error(_error):
        # OSError text may contain arbitrary input; do not serialize it.
        errors.append({"kind": "directory_read_error"})

    for directory, dirs, names in os.walk(root, onerror=walk_error, followlinks=False):
        for name in list(dirs):
            path = Path(directory) / name
            if name in SKIP_DIRS or path.is_symlink():
                skipped["dependency_cache_git_or_symlink_directory"] += 1
                dirs.remove(name)
        for name in sorted(names):
            path = Path(directory) / name
            rel = path.relative_to(root).as_posix()
            if path.is_symlink():
                skipped["symlink"] += 1
                continue
            try:
                with path.open("rb") as stream:
                    if path.suffix.lower() in BINARY_SUFFIXES or b"\0" in stream.read(8192):
                        skipped["binary_or_archive"] += 1
                        continue
                    stream.seek(0)
                    files_scanned += 1
                    for line_number, raw in enumerate(stream, 1):
                        line = raw.decode("utf-8", errors="replace")
                        # JSON-encoded shell commands can contain escaped quotes.
                        line = line.replace('\\"', '"').replace("\\'", "'")
                        for rule, pattern in PATTERNS.items():
                            count = sum(1 for _ in pattern.finditer(line))
                            if count:
                                findings.append({"file": rel, "line": line_number,
                                                 "rule": rule, "count": count})
                        if (line.strip() and path.suffix.lower() in {".key", ".pem"}
                                and line_number == 1):
                            findings.append({"file": rel, "line": 1,
                                             "rule": "credential_file", "count": 1})
            except OSError:
                errors.append({"file": rel, "kind": "file_read_error"})
    return {"scope": "working_tree_text_files_only", "root": str(root),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "files_scanned": files_scanned, "skipped": dict(skipped),
            "excluded_directories": sorted(SKIP_DIRS),
            "git_history": "not_scanned", "errors": errors,
            "status": "incomplete" if errors else ("review_required" if findings else "no_pattern_matches"),
            "matching_locations": len(findings),
            "findings": sorted(findings, key=lambda row: (row["file"], row["line"], row["rule"]))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error("--root must be an existing directory")
    report = scan(args.root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 2 if report["errors"] else (1 if report["findings"] else 0)


if __name__ == "__main__":
    raise SystemExit(main())
