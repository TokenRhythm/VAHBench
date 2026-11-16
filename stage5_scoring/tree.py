"""Walk the scoring tree and check every leaf against the code that feeds it.

    python -m stage5_scoring.tree

For each measurement key the tree reads, find which function emits it and from
which module's report. A leaf whose key nothing emits is a promise, not a
measurement, and this prints it as MISSING.
"""
from __future__ import annotations

import ast
import pathlib
import sys

from shared.paths import REPO_ROOT

import stage5_scoring as rubric
from stage5_scoring import registry

SOURCE_DIRS = [REPO_ROOT / name for name in (
    "stage1_inputs", "stage2_delivery", "stage3_evidence", "stage4_checks", "stage5_scoring", "shared")]
SRC = {p.relative_to(REPO_ROOT).as_posix(): p.read_text(encoding="utf-8")
       for directory in SOURCE_DIRS for p in directory.rglob("*.py")
       if not any(part in {"audit", "inspect", "batch"} for part in p.relative_to(directory).parts)}
ALL = set(registry.ALL_CHECKS)

# Which evidence each emitter reads. `extract.*` read one module report each;
# `expectations.*` read the brief against whatever the evidence bus supplies.
FEEDS = {"from_spec": "spec", "spec_from_probe": "ffprobe", "from_artifact": "artifact",
         "from_visual": "visual",
         "from_physical": "physical", "from_av": "av", "from_silence": "ffmpeg",
         "from_facts": "trace", "from_delivery": "trace+delivery",
         "_requested_vs_target": "trace",
         "check_status": "trace+delivery", "check_deliverables": "trace+delivery",
         "check_episode_claims": "trace+delivery",
         "from_seam": "seam",
         "word_cap": "brief+ASR", "mandated_lines": "brief+ASR",
         "no_voiceover": "brief+av", "deliverables": "brief+files",
         "subtitles_aligned": "brief+srt+ASR", "characters_anchored": "brief+visual",
         "scene_fidelity": "brief+visual", "asset_utilisation": "brief+trace",
         "continuity_contract": "brief+trace", "specs_in_prompts": "brief+trace",
         "scene_plan": "brief+trace"}

# Detection is over the AST, not the text. A key reaches its constructor by
# three different routes in this codebase -- literal (`_m("k", ...)`), a local
# (`key = "k"` ... `_m(key, ...)`), and a module-level name->key table -- and a
# text search that catches all three also catches every docstring that names a
# key. String *constants* in the tree catch all three and no prose.
CTORS = {"_m", "_abstain", "_void", "Measurement", "measurement"}
# flows.py rewrites verdicts on measurements that already exist; it originates
# none of them, so it is not an emitter (its rules are listed by `run_eval`'s
# flow ledger instead).
NOT_EMITTERS = {"stage4_checks.cross_checks"}


def _strings(node: ast.AST) -> set[str]:
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            out.add(n.value)
    return out


def _index() -> dict[str, list[str]]:
    """key -> ["module.function", ...] for every function that emits it."""
    idx: dict[str, list[str]] = {}
    for fname, src in SRC.items():
        mod = fname.removesuffix(".py").replace("/", ".")
        if mod in NOT_EMITTERS:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        # Module-level name -> key tables, credited only to the functions that
        # actually name the table. Crediting every emitter in the file instead
        # made `_NO_DELIVERY_VOIDS` look like fifteen different emitters.
        tables: dict[str, set[str]] = {}
        for n in tree.body:
            if isinstance(n, (ast.Assign, ast.AnnAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                names = [t.id for t in targets if isinstance(t, ast.Name)]
                keys = {s for s in _strings(n) if "." in s}
                for nm in names:
                    tables[nm] = keys
        for fn in [n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            calls = {c.func.id for c in ast.walk(fn)
                     if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
            if not (calls & CTORS):
                continue
            body = list(fn.body)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)):
                body = body[1:]                      # drop the docstring
            keys = set()
            for stmt in body:
                keys |= _strings(stmt)
            used = {n.id for stmt in body for n in ast.walk(stmt)
                    if isinstance(n, ast.Name)}
            for nm in used & set(tables):
                keys |= tables[nm]
            for k in keys & ALL:
                idx.setdefault(k, []).append(f"{mod}.{fn.name}")
    return {k: sorted(set(v)) for k, v in idx.items()}


EMITTERS = _index()


def emitters(key: str) -> list[str]:
    return EMITTERS.get(key, [])


def main() -> int:
    missing = []
    print("S = 100 · C · Q^(3/4) · P^(1/4)\n")

    print("1.1  完成度 C  — seconds, read by rubric/video/completeness.py")
    for k in sorted(registry.COMPLETENESS_INPUTS):
        e = emitters(k)
        print(f"     {k:<38} {', '.join(e) or 'MISSING'}")
        if not e:
            missing.append(k)

    for title, cats in (("1.2  质量 Q", rubric.QUALITY),
                        ("2    过程 P", rubric.TRAJECTORY)):
        print(f"\n{title}")
        for c in cats:
            print(f"  {c.id}  {c.name}")
            for lf in c.leaves:
                flag = "" if lf.implemented else "  [deferred]"
                print(f"    {lf.id}  {lf.name} ({lf.key}){flag}")
                for k in lf.sources:
                    e = emitters(k)
                    mods = sorted({FEEDS.get(x.split(".")[-1], "?") for x in e})
                    if not e and lf.implemented:
                        missing.append(k)
                    print(f"       {k:<36} {'/'.join(mods) if e else '—'}"
                          f"  {', '.join(e) or 'MISSING'}")

    # A check either scores or is deleted; this bucket is closed (registry.py).
    # Printed only if someone reopens it, so the report shows the breach.
    if registry.DIAGNOSTIC:
        print(f"\n诊断项（测量、报告、不计分）: {len(registry.DIAGNOSTIC)} —— 该桶应为空")
        for k in sorted(registry.DIAGNOSTIC):
            print(f"     {k:<38} {', '.join(emitters(k)) or 'MISSING'}")

    n_leaves = sum(len(c.leaves) for c in (*rubric.QUALITY, *rubric.TRAJECTORY))
    n_impl = sum(1 for c in (*rubric.QUALITY, *rubric.TRAJECTORY)
                 for lf in c.leaves if lf.implemented)
    rc = registry.completeness_of_rubric()
    print(f"\n叶子 {n_impl}/{n_leaves} 有实现接口；键 {rc['n_checks']} = "
          f"{rc['n_scored']} 计分 + {rc['n_completeness_inputs']} 完成度 + "
          f"{rc['n_diagnostic']} 诊断 + {rc['n_deferred']} 推迟")
    for key, why in rc["evidence_limitations"].items():
        print(f"  [证据能力缺口] {key}: {why}")
    if missing:
        print("\nMISSING (declared by the tree, emitted by nothing):")
        for k in sorted(set(missing)):
            print("  ", k)
        return 1
    print("\n每个已实现叶子的每个来源都有代码发出。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
