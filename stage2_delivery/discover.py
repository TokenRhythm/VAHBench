"""Run discovery and delivery location. Path-independent by construction.

The input to this benchmark is whatever an agent left on disk. Measured across
the six harnesses in `data/`, the delivered episodes live at six mutually
disjoint paths:

    claudecode   output/ep1_final.mp4
    codex        output/ep1_final.mp4 · output/gpt5_ep1_final.mp4
    openmontage  <model>/artifacts/<case>/renders/ep1.mp4
    videoclaw    <model>/artifacts/video/<slug>/output/<slug>_ep1.mp4
    univa        <model>/artifacts/<timestamp>_<prompt-text>/1.mp4   (no episode at all)
    vimax        <model>/working/scene_1/shots/4/cache/...

The brief provides a duration signal for otherwise unlabelled videos. Explicit
episode exports must also be admitted when they miss that duration: selecting
the delivered object and judging its duration are separate decisions.

The locator distinguishes an episode export from its assembly base, then ranks
comparable candidates by content. The agent's own declaration is treated as a
hypothesis to be checked rather than as the answer (EVAL_DESIGN.md 2.5).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

from stage2_delivery.probe import MediaInfo
from stage2_delivery.probe import ProbeCache

# General final-file hints rank candidates; explicit episode identities below
# can also admit decoded exports outside the duration discovery window.
FINAL_HINTS = re.compile(r"(final|merged|output|render|episode|ep\d|complete)", re.I)
# Names that hint at an intermediate. Same: a demotion, never an exclusion.
WORK_HINTS = re.compile(r"(clip|seg|shot|cache|work|tmp|temp|raw|scene_\d|_n\.)", re.I)
# Short E01 identifiers need a token boundary: e1a is not an episode label.
EP_NUM = re.compile(r"(?:^|[^a-z0-9])(?:ep(?:isode)?[_\-]?|"
                    r"e(?=\d{1,2}(?:$|[^a-z0-9])))(\d{1,2})(?!\d)", re.I)
SOURCE_FILE = re.compile(
    r"(?:^|[_\-])(?:clip|seg(?:ment)?|shot|scene)[_\-]?\d|"
    r"(?:^|[_\-])(?:raw|draft|preview|tmp|temp)(?:[_\-.]|$)", re.I)
SOURCE_DIR = re.compile(r"(^|/)(clips|segments|shots|scenes|raw)(/|$)", re.I)
BUILD_STAGE = re.compile(r"(?:^|/)build(?:/|$)|(?:^|/)base[_\-.]", re.I)
# Scratch trees. A file here is an intermediate however it is named: vimax's
# `working/scene_1/final_video.mp4` is the final of *that scene*, honestly
# named, and still not an episode. It was selected because `final` matched
# FINAL_HINTS and the run had declared it -- along with 158 other files.
SCRATCH = re.compile(r"(^|/)(working|cache|tmp|temp|scratch|_work|work)(/|$)", re.I)


@dataclass(frozen=True)
class Target:
    """What the brief asked for. Never guessed -- unknown stays None."""

    episodes: int | None = None
    seconds_per_episode: float | None = None
    aspect_ratio: str | None = None
    resolution: str | None = None
    source: str = "unknown"

    @property
    def known(self) -> bool:
        return self.seconds_per_episode is not None or self.episodes is not None


@dataclass
class Candidate:
    info: MediaInfo
    channels: list[str] = field(default_factory=list)   # declaration/content/name
    duration_error: float | None = None                 # |dur - target| / target
    rank: float = 0.0
    episode: int | None = None
    scratch: bool = False                              # lives under a scratch tree


@dataclass
class Run:
    """One (harness, case, model) triple -- the unit the benchmark scores."""

    root: Path
    harness: str
    case: str
    model: str | None
    target: Target
    run_json: dict | None = None
    declaration: dict | None = None          # verification.json / manifest.json
    declared_videos: list[Path] = field(default_factory=list)
    declared_names: list[str] = field(default_factory=list)
    status: str | None = None                # harness-reported, never trusted for scoring
    group: str = ""                          # levels above the harness, e.g. "agents"
    rel: str = ""                            # run root relative to the data root

    @property
    def run_id(self) -> str:
        # The run root's own path is the only identifier that is unique across
        # every layout the data has been in; harness/model/case are derived
        # labels and can collide (codex under harness/ and under openmontage/).
        return self.rel or "/".join(x for x in (self.harness, self.case, self.model) if x)


@dataclass
class Delivery:
    """The videos this run is judged on, plus everything that was considered."""

    videos: list[Candidate] = field(default_factory=list)
    selection: str = "none"      # declared | content | unassembled | none
    disagreement: list[str] = field(default_factory=list)
    n_candidates: int = 0
    reason: str = ""
    material_seconds: float = 0.0
    required_seconds: float = 0.0

    @property
    def found(self) -> bool:
        return bool(self.videos)

    @property
    def assembled(self) -> bool:
        """Did the run end with episode-shaped files, or only their material?"""
        return self.found and self.selection != "unassembled"


def _load_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def _norm_rel(root: Path, raw: str) -> Path:
    """Declared paths come back with whatever separator the producing OS used."""
    return (root / str(raw).replace("\\", "/")).resolve()


# The task block is spelled differently by every harness that writes one.
# Read all the spellings rather than imposing one -- the point of this layer is
# that the benchmark adapts to the agent, not the other way round.
_TASK_KEYS = ("target", "task", "spec", "brief")
_EPISODE_KEYS = ("episodes", "n_episodes", "episode_count")
_SECONDS_KEYS = ("seconds_per_episode", "episode_seconds", "duration_seconds",
                 "seconds", "target_duration")
_ASPECT_KEYS = ("aspect_ratio", "aspect")
_SIZE_KEYS = ("size", "resolution")


def _first(d: dict, keys) -> object | None:
    for k in keys:
        if d.get(k) not in (None, "", []):
            return d[k]
    return None


def read_target(run_root: Path, case_dir: Path) -> tuple[Target, dict | None, dict | None]:
    """Target from the run's own record; declaration from whatever it self-reported.

    Three shapes exist in `data/` and all three are read:
    `run.json.target` (_runner harnesses), `run.json.task` (codex), and
    `verification.json` (claudecode / codex, which additionally declares
    per-episode files and is therefore also a declaration channel).
    """
    rj = _load_json(run_root / "run.json") or _load_json(case_dir / "run.json")
    decl = (_load_json(run_root / "manifest.json")
            or _load_json(run_root / "verification.json")
            or _load_json(case_dir / "manifest.json")
            or _load_json(case_dir / "verification.json"))

    episodes = secs = aspect = resolution = None
    source = "unknown"

    if rj:
        block = next((rj[k] for k in _TASK_KEYS if isinstance(rj.get(k), dict)), None)
        if block is not None:
            episodes = _first(block, _EPISODE_KEYS)
            secs = _first(block, _SECONDS_KEYS)
            aspect = _first(block, _ASPECT_KEYS)
            resolution = _first(block, _SIZE_KEYS)
            source = "run.json"

    if decl:
        eps = decl.get("episodes")
        if isinstance(eps, list) and eps:
            if episodes is None:
                episodes = len(eps)
            tgt = [e.get("target") for e in eps if isinstance(e, dict) and e.get("target")]
            if secs is None and tgt:
                secs = float(tgt[0])
            source = "verification.json" if source == "unknown" else source + "+declaration"
        if aspect is None:
            aspect = _first(decl, _ASPECT_KEYS)
        if resolution is None:
            resolution = _first(decl, _SIZE_KEYS)

    return (Target(int(episodes) if episodes else None,
                   float(secs) if secs else None,
                   str(aspect) if aspect else None,
                   str(resolution) if resolution else None,
                   source),
            rj, decl)


def declared_video_paths(run_root: Path, rj: dict | None,
                         decl: dict | None) -> tuple[list[Path], list[str]]:
    """The declaration channel.

    Returns resolved paths *and* the bare names that were declared. Both are
    needed because harnesses declare at different granularity: `_runner` writes
    a run-relative path, claudecode's verification.json writes a bare filename
    (`ep1_final.mp4`) for a file that actually lives in `output/`. Matching on
    the literal string alone would report every claudecode delivery as
    "declared but missing" -- which is a locator bug, not a finding.
    """
    paths: list[Path] = []
    names: list[str] = []

    def add(raw: str) -> None:
        paths.append(_norm_rel(run_root, raw))
        names.append(Path(str(raw).replace("\\", "/")).name)

    for src, keys in ((rj, ("videos", "outputs")), (decl, ("videos", "outputs"))):
        if not src:
            continue
        for key in keys:
            for v in src.get(key) or []:
                if isinstance(v, str):
                    add(v)
                elif isinstance(v, dict) and (v.get("path") or v.get("file")):
                    add(v.get("path") or v["file"])
    if decl:
        for e in decl.get("episodes") or []:
            if isinstance(e, dict) and e.get("file"):
                add(e["file"])
    return paths, names


def _is_run_dir(p: Path) -> bool:
    """A run leaves a record. Which record differs by harness, so accept any.

    openmontage/prompt_07 has a model directory holding only stdout.log and
    trace.jsonl -- no run.json at all. Requiring run.json silently demoted that
    to a case-level run with zero videos, i.e. the harness's most interesting
    failure was invisible.
    """
    return any((p / n).is_file() for n in
               ("run.json", "trace.jsonl", "stdout.log", "verification.json",
                "manifest.json")) or (p / "artifacts").is_dir()


def _is_case_dir(p: Path) -> bool:
    return p.name.lower().startswith("prompt")


_VIDEO_SUFFIX = (".mp4", ".mov", ".mkv", ".webm")
_RECORD_NAMES = ("run.json", "trace.jsonl", "events.jsonl", "stdout.log",
                 "verification.json", "manifest.json", "costs.jsonl",
                 "TRAJECTORY.md", "prompts.json")
_NEVER_ENTER = ("node_modules", "__pycache__", "site-packages", ".git")


def _record_below(d: Path, depth: int = 4) -> bool:
    """A trajectory record anywhere under `d`, hidden directories included --
    hermes keeps its whole export under `.hermes/trajectory-exports/`."""
    if depth < 0 or d.name in _NEVER_ENTER:
        return False
    try:
        kids = list(d.iterdir())
    except OSError:
        return False
    for k in kids:
        if k.is_file() and (k.name in _RECORD_NAMES or k.suffix.lower() == ".jsonl"):
            return True
    return any(k.is_dir() and _record_below(k, depth - 1) for k in kids)


def _looks_like_run(p: Path) -> bool:
    """A run the user dropped in under any name. Three signs, any one enough:
    a run record or trajectory log directly here; a video directly here; or
    a record inside a *hidden* child (`.hermes/...`). The last one is what
    keeps a run's own subfolders (clips/, output/) from being taken for runs
    of their own: the parent is claimed first and never descended. The
    benchmark's input is a trajectory and a video; nothing about the folder
    name is required."""
    if _is_run_dir(p):
        return True
    try:
        kids = list(p.iterdir())
    except OSError:
        return False
    if any(k.is_file() and (k.suffix.lower() in _VIDEO_SUFFIX or k.suffix.lower() == ".jsonl")
           for k in kids):
        return True
    return any(k.is_dir() and k.name.startswith(".") and _record_below(k) for k in kids)


def find_runs(root: Path, harness_filter: set[str] | None = None) -> list[Run]:
    """Walk to every case directory, whatever sits above or below it.

    Nothing about depth is assumed. `data/` has held, at different times,
    `<agent>/<case>/<model>/`, `<agent>/<variant>/<case>/`, and -- after the
    2026-09-15 merge with VAHBench -- `<group>/<agent>/<model>/<case>/`, with
    one agent (openmontage) still on the old shape because a run was writing
    into it. All of these coexist in one tree, and the first version of this
    walker, which assumed the model level sat *below* the case, reported the
    merged dataset as 18 runs instead of ~90 -- it took `agents/` and
    `harness/` themselves for cases.

    A case directory is one whose name starts with `prompt`. Directories above
    it are labels: the last two become harness and model (`agents/univa/<m>`
    -> harness univa, model <m>, group agents); a lone label is the harness
    with no model level. If the case directory itself contains run directories
    (run.json, trace.jsonl, artifacts/ ...), the old shape applies and each of
    those is a run with the model taken from its name. Hidden and `_`-prefixed
    directories (`.hermes`, `_shared`, `_case_conflicts`) are never entered.
    """
    root = Path(root)
    runs: list[Run] = []

    def emit(run_root: Path, case_dir: Path, labels: list[str],
             sub_model: str | None) -> None:
        harness = labels[-2] if len(labels) >= 2 else (labels[0] if labels else "")
        model_bits = ([labels[-1]] if len(labels) >= 2 else []) + \
                     ([sub_model] if sub_model else [])
        group = "/".join(labels[:-2]) if len(labels) > 2 else ""
        target, rj, decl = read_target(run_root, case_dir)
        paths, names = declared_video_paths(run_root, rj, decl)
        try:
            rel = str(run_root.relative_to(root)).replace("\\", "/")
        except ValueError:
            rel = str(run_root)
        if rel == ".":                          # --data points at the run itself
            rel = run_root.name
        runs.append(Run(
            root=run_root, harness=harness, case=case_dir.name,
            model="/".join(model_bits) or None, group=group, rel=rel,
            target=target, run_json=rj, declaration=decl,
            declared_videos=paths, declared_names=names,
            status=(rj or {}).get("status"),
        ))

    def walk(d: Path, labels: list[str]) -> None:
        try:
            children = sorted(p for p in d.iterdir() if p.is_dir()
                              and not p.name.startswith((".", "_")))
        except OSError:
            return
        if _is_case_dir(d):
            # Old shape: the model level sits *below* the case. It only applies
            # when the case directory is not itself a run -- otherwise the run's
            # own output folder gets taken for a model and becomes the run root,
            # one level below the trajectory. That is what happened to the 15
            # openmontage runs: `prompt_XX/` holds run.json, trace.jsonl and
            # stdout.log, and `prompt_XX/artifacts/` (which holds a nested
            # `artifacts/`, so `_is_run_dir` says yes) was emitted as the run.
            # The trajectory readers look at the run root, found nothing, and
            # the whole trajectory layer abstained on runs whose log was right
            # there. A directory holding its own record is the run; what is
            # under it belongs to it.
            model_dirs = ([] if _is_run_dir(d)
                          else [p for p in children if _is_run_dir(p)])
            if model_dirs:
                for m in model_dirs:
                    emit(m, d, labels, m.name)
            else:
                emit(d, d, labels, None)
            return
        if _looks_like_run(d):
            # No `prompt*` name above or here, but a trajectory or a video
            # sits right in this directory: a run the user dropped in as-is.
            # Its name is the case; its brief is matched from the text it
            # embeds (brief.match_brief), not from the name.
            emit(d, d, labels, None)
            return
        for c in children:
            walk(c, labels + [d.name])

    if _is_case_dir(root) or _looks_like_run(root):
        walk(root, [])                          # --data is a single case or run
    else:
        for top in sorted(p for p in root.iterdir() if p.is_dir()
                          and not p.name.startswith((".", "_"))):
            walk(top, [])
    _inherit_targets(runs)
    if harness_filter:
        runs = [r for r in runs if r.harness in harness_filter
                or (r.group and r.group in harness_filter)]
    return runs


_PROMPT_KEY = re.compile(r"^(prompt[_\-]?\d+)", re.I)


def _inherit_targets(runs: list[Run]) -> None:
    """Give record-less cases the brief their siblings recorded.

    The target is a property of the prompt, not of the harness: every run of
    prompt_07 was asked for 5 x 240 s at 16:9 whoever produced it. After the
    VAHBench merge 36 of 84 cases carry no run.json or verification.json at
    all, and without a target the locator degrades to "longest video wins".
    So a case with no target takes the one its same-prompt siblings agree on
    -- and only if they agree; a prompt whose recorded targets conflict gives
    nothing, because inheriting a disputed brief would be inventing one.
    """
    by_prompt: dict[str, set[tuple]] = {}
    res_by_prompt: dict[str, set[str]] = {}
    for r in runs:
        m = _PROMPT_KEY.match(r.case)
        key = (m.group(1) if m else r.case).lower()
        t = r.target
        if t.episodes is None and t.seconds_per_episode is None:
            continue
        by_prompt.setdefault(key, set()).add(
            (t.episodes, t.seconds_per_episode, t.aspect_ratio))
        if t.resolution:
            res_by_prompt.setdefault(key, set()).add(t.resolution)
    for r in runs:
        if r.target.known:
            continue
        m = _PROMPT_KEY.match(r.case)
        key = (m.group(1) if m else r.case).lower()
        agreed = by_prompt.get(key, set())
        if len(agreed) != 1:
            continue                      # no siblings, or siblings disagree
        episodes, secs, aspect = next(iter(agreed))
        res = res_by_prompt.get(key, set())
        r.target = Target(episodes, secs, aspect,
                          next(iter(res)) if len(res) == 1 else None,
                          source="sibling_prompt")


def _duration_error(dur: float | None, target: float | None) -> float | None:
    if dur is None or not target:
        return None
    return abs(dur - target) / target


# There is no floor on how much material a run must hold before its footage
# is handed to the modules. A 0.75 threshold used to decide, on runtime alone
# and before a single pixel was read, that a run had produced nothing: across
# `data/` it sent 12 of the 13 unassembled runs straight to "no delivery",
# including 1154 s of material against an 1800 s brief. Whether the material
# is enough is what the seconds ledger computes; it is not the locator's
# question to answer in advance.


def _material_seconds(root: Path, cands: list[Candidate]) -> float:
    """Runtime of the distinct material, counting each piece once.

    The obstacle is that an agent leaves every intermediate stage on disk.
    vimax writes `working/scene_1/final_video.mp4` beside the ten shots it was
    built from; videoclaw writes `output/ep1.mp4` beside its segments; openclaw
    writes `renders/ep1.mp4` and then copies the whole tree one directory
    deeper. Adding everything up put vimax's prompt_01 at 733% of a 180 s brief
    and openclaw's prompt_09 at 96% of an 1800 s one when the honest figures
    are 215% and 64% -- the second of which changes the verdict.

    So: measure at the most assembled level the run reached. If it produced
    anything assembly-shaped, those are the material and the clips underneath
    them are not counted again; otherwise the clips are all there is. Keying
    assemblies by episode index is what makes the duplicated directory tree
    collapse -- `renders/ep1.mp4` and `.../renders/ep1.mp4` are both episode 1.

    Assemblies win whenever they exist, with no floor on how much they add up
    to. A floor was tried -- "use them only if they cover at least one episode"
    -- and it was wrong in both directions: it sent vimax's prompt_09 back to
    749 s of double-counted shots when the honest figure is its four assembled
    scenes at 234 s, and the runs it was meant to catch do not exist. Across
    all 80 runs, every assembly set either matches the brief almost exactly
    (hermes prompt_03: 361 s against 360 s) or is absent entirely; the low
    assembly-to-clip ratios come from agents that kept a lot of source footage,
    not from stray test renders.
    """
    finals = [c for c in cands if FINAL_HINTS.search(c.info.path.name)]
    if not finals:
        return sum(c.info.duration or 0.0 for c in cands)

    best: dict[object, float] = {}
    for c in finals:
        if c.episode is not None:
            key: object = c.episode
        else:
            # No index: one assembly per directory (vimax's scenes).
            try:
                key = str(c.info.path.parent.relative_to(root)).replace("\\", "/")
            except ValueError:
                key = c.info.path.parent.name
        best[key] = max(best.get(key, 0.0), c.info.duration or 0.0)
    return sum(best.values())


def _distinct(cands: list[Candidate]) -> list[Candidate]:
    """Drop byte-identical repeats -- one result polled and saved seven times.

    univa's prompt_09 holds seven files of 50.6 s at exactly 27047031 bytes:
    the same render fetched again on each poll, not seven pieces of material.
    """
    seen: set[tuple] = set()
    out: list[Candidate] = []
    for c in sorted(cands, key=lambda x: str(x.info.path)):
        key = (c.info.size_bytes, round(c.info.duration or 0.0, 2))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _episode_exports(run: Run) -> tuple[set[Path], list[str]]:
    """Resolve selective episode manifests inside archived output directories.

    A full harness inventory must not erase a smaller authored delivery list.
    Existence is only a hypothesis here; media probing still gates candidates.
    Conflicting manifests remain ambiguous instead of choosing one by filename.
    """
    from stage4_checks.process.honesty import resolve_claim
    groups: dict[frozenset[Path], list[str]] = {}
    for name in ('delivery_package.json', 'manifest.json', 'verification.json'):
        for path in sorted(run.root.rglob(name)):
            try:
                if path.stat().st_size > 2_000_000:
                    continue
                obj = json.loads(path.read_text(encoding='utf-8-sig'))
            except (OSError, ValueError):
                continue
            rows = obj.get('episodes') if isinstance(obj, dict) else None
            if not isinstance(rows, list) or not rows or len(rows) > (run.target.episodes or 6):
                continue
            resolved = []
            for row in rows:
                raw = (row.get('file') or row.get('export_path')) if isinstance(row, dict) else None
                if not isinstance(raw, str) or Path(raw).suffix.lower() not in ('.mp4', '.mov'):
                    break
                hit, status = resolve_claim(run.root, raw)
                if status != 'found':
                    break
                resolved.append(hit)
            if len(resolved) == len(rows) and len(set(resolved)) == len(rows):
                groups.setdefault(frozenset(resolved), []).append(str(path.relative_to(run.root)))
    if len(groups) == 1:
        paths, sources = next(iter(groups.items()))
        return set(paths), sources
    return set(), []


def locate_delivery(run: Run, cache: ProbeCache,
                    tolerance: float = 0.25,
                    max_files: int = 4000) -> Delivery:
    """Pick delivered episodes, distinguishing exports from assembly bases.

    `tolerance` is generous on purpose. It is a *discovery* window, not a spec
    check -- a 45 s episode against a 60 s brief is still the delivered episode,
    and spec_checks is what decides whether that runtime passes.
    """
    mp4s = [p for p in run.root.rglob("*.mp4")][:max_files]
    mp4s += [p for p in run.root.rglob("*.mov")][:max_files]
    if not mp4s:
        return Delivery(reason="run 目录下没有任何视频文件")

    declared = {p.resolve() for p in run.declared_videos}
    declared_names = set(run.declared_names)
    target_s = run.target.seconds_per_episode
    want = run.target.episodes or 0
    # A declaration is evidence only when it is selective. vimax declared 159
    # videos against a 3-episode brief -- a dump of everything it wrote, not a
    # statement of what it delivered -- and every one of them was collecting
    # the "agent says this is a deliverable" bonus.
    dump = bool(declared_names) and len(declared_names) > max(3 * want, 6)
    exports, export_sources = _episode_exports(run)

    cands: list[Candidate] = []
    for p in mp4s:
        info = cache.get(p)
        if not info.ok:
            continue
        c = Candidate(info=info)
        c.duration_error = _duration_error(info.duration, target_s)
        if p.resolve() in declared or p.name in declared_names:
            c.channels.append("declaration")
        if p.resolve() in exports:
            c.channels.append("episode_manifest")
        try:
            rel_parent = str(p.parent.relative_to(run.root)).replace("\\", "/")
        except ValueError:
            rel_parent = str(p.parent)
        c.scratch = bool(SCRATCH.search(rel_parent))
        name = p.name
        parent = p.parent.name
        if FINAL_HINTS.search(name) or FINAL_HINTS.search(parent):
            c.channels.append("name")
        m = EP_NUM.search(name) or EP_NUM.search(parent)
        if m:
            c.episode = int(m.group(1))
        if (EP_NUM.search(name) and not c.scratch and not SOURCE_FILE.search(name)
                and not SOURCE_DIR.search(rel_parent)):
            c.channels.append("episode_name")
        cands.append(c)

    if not cands:
        return Delivery(reason="视频文件都无法解码", n_candidates=len(mp4s))

    # ---- content channel: how close is this to what the brief asked for -----
    for c in cands:
        score = 0.0
        if c.duration_error is not None and c.duration_error <= tolerance:
            score += 100.0 * (1.0 - c.duration_error / tolerance)
            c.channels.append("content")
        elif target_s is None:
            # No target: fall back to "longest wins", which is the only
            # content signal left. Recorded so the report can say so.
            score += min(50.0, (c.info.duration or 0.0) / 4.0)
        if "declaration" in c.channels and not dump:
            score += 40.0
        elif "episode_manifest" in c.channels:
            score += 40.0
        if FINAL_HINTS.search(c.info.path.name) or FINAL_HINTS.search(c.info.path.parent.name):
            score += 12.0
        if WORK_HINTS.search(str(c.info.path.parent)) or WORK_HINTS.search(c.info.path.name):
            score -= 8.0
        c.rank = score

    in_window = [c for c in cands if c.duration_error is not None
                 and c.duration_error <= tolerance]
    identified = [c for c in cands if "episode_manifest" in c.channels
                  or "episode_name" in c.channels]
    pool = ([c for c in cands if c in in_window or c in identified]
            if target_s is not None else cands)

    # Scratch-tree files are not deliverables unless the agent *selectively*
    # declared them. A dump does not count as selecting anything.
    scratch_hits = [c for c in pool if c.scratch
                    and not ("episode_manifest" in c.channels or
                             ("declaration" in c.channels and not dump))]
    pool = [c for c in pool if c not in scratch_hits]

    # An episode's assembly base is not its exported version. Duration fit
    # cannot distinguish them (both may be exactly 60 s), and must not make
    # the base win when the actual export is off specification. Keep the
    # existing fallback when no later-stage file for that episode exists;
    # a selective declaration can also explicitly deliver a build file.
    def intermediate(c: Candidate) -> bool:
        rel = c.info.path.relative_to(run.root).as_posix()
        return bool(SOURCE_FILE.search(c.info.path.name)
                    or SOURCE_DIR.search(str(Path(rel).parent).replace("\\", "/"))
                    or BUILD_STAGE.search(rel))

    exported_episodes = {c.episode for c in pool if c.episode is not None
                         and ("episode_name" in c.channels or "episode_manifest" in c.channels)
                         and not intermediate(c)}
    superseded = [c for c in pool if c.episode in exported_episodes and intermediate(c)
                  and not ("episode_manifest" in c.channels
                           or ("declaration" in c.channels and not dump))]
    pool = [c for c in pool if c not in superseded]
    if not pool and scratch_hits:
        best = max(scratch_hits, key=lambda c: -c.rank)
        try:
            rel = str(best.info.path.relative_to(run.root)).replace("\\", "/")
        except ValueError:
            rel = best.info.path.name
        notes = ([f"声明了 {len(declared_names)} 个视频，目标 {want} 集：声明是全量转储，不作为交付依据"]
                 if dump else [])
        # Where the files sit stays on the record even when they are admitted
        # as material: "in a working directory" is why they are not episodes,
        # and it is the sentence a reader needs next to the assembly deduction.
        notes.append(f"{len(scratch_hits)} 个接近目标时长的视频全部位于工作目录"
                     f"（如 {rel}），是中间产物，不作为分集成片")
        return _unassembled(run, cands, want, notes) or Delivery(
            n_candidates=len(cands), disagreement=notes,
            material_seconds=_material_seconds(run.root, _distinct(cands)),
            required_seconds=want * (target_s or 0.0),
            reason=(f"{len(scratch_hits)} 个接近目标时长的视频全部位于工作目录"
                    f"（如 {rel}），是中间产物，未交付到产出位置"))

    if not pool:
        best = max(cands, key=lambda c: c.info.duration or 0.0)
        return _unassembled(run, cands, want, []) or Delivery(
            n_candidates=len(cands),
            material_seconds=_material_seconds(run.root, _distinct(cands)),
            required_seconds=want * (target_s or 0.0),
            reason=(f"{len(cands)} 个视频，最长 {best.info.duration:.1f}s，"
                    f"没有一个接近 {target_s:.0f}s 的目标时长；"
                    f"素材也不足 {want * (target_s or 0):.0f}s 的需求——本 run 没有交付"))

    pool.sort(key=lambda c: -c.rank)
    chosen = _dedupe_by_episode(pool, want)

    sel = ("declared" if chosen and all("episode_manifest" in c.channels or
           ("declaration" in c.channels and not dump) for c in chosen) else "content")
    disagreement = _cross_check(chosen, run)
    off_duration = [c for c in chosen if c not in in_window and c in identified]
    if off_duration:
        disagreement.append(f"按分集身份纳入 {len(off_duration)} 个非目标时长成片；时长偏差由完成度计分")
    if dump:
        disagreement.insert(0, f"声明了 {len(declared_names)} 个视频，目标 {want} 集："
                               "声明是全量转储，不作为交付依据")
    if export_sources:
        disagreement.append("另有可定位的分集交付清单：" + "、".join(export_sources))
    if scratch_hits:
        disagreement.append(f"另有 {len(scratch_hits)} 个近目标时长的视频位于工作目录，未计入交付")
    if superseded:
        disagreement.append(f"另有 {len(superseded)} 个同集的中间文件候选，由可解码分集成片替代")
    return Delivery(videos=chosen, selection=sel, disagreement=disagreement,
                    n_candidates=len(cands),
                    material_seconds=_material_seconds(run.root, _distinct(cands)),
                    required_seconds=want * (target_s or 0.0))


def _unassembled(run: Run, cands: list[Candidate], want: int,
                 notes: list[str]) -> Delivery | None:
    """Delivered the footage, never cut it together.

    The standard this implements: a run counts as delivered when it produced
    the material the brief asked for, and stopping one step short of the final
    concatenation is acceptable. vimax is the case it exists for -- prompt_01
    holds six assembled scenes covering roughly 190% of a 180 s brief, but no
    episode-shaped file anywhere, so the episode-length search finds nothing
    and the run used to score as having produced nothing at all.

    Returns None when there is no brief to measure against, or when the
    material falls short; the caller then reports its own reason.
    """
    target_s = run.target.seconds_per_episode
    required = want * (target_s or 0.0)
    if required <= 0:
        return None

    distinct = _distinct(cands)
    material = _material_seconds(run.root, distinct)
    if not distinct:
        return None

    # Hand the modules the most finished thing available: the assemblies if
    # there are any, otherwise the longest clips. Capped at the episode count
    # so an unassembled run does not cost more to measure than an assembled one.
    finals = [c for c in distinct if FINAL_HINTS.search(c.info.path.name)]
    chosen = sorted(finals or distinct,
                    key=lambda c: -(c.info.duration or 0.0))[:max(want, 1)]
    chosen.sort(key=lambda c: (c.episode if c.episode is not None else 99,
                               str(c.info.path)))
    return Delivery(
        videos=chosen, selection="unassembled",
        disagreement=notes + [f"未合成分集：素材 {material:.0f}s 覆盖 "
                              f"{material / required * 100:.0f}% 的需求，"
                              f"但没有分集成片，按素材达标计交付"],
        n_candidates=len(cands), material_seconds=material,
        required_seconds=required,
        reason=(f"素材 {material:.0f}s / 需求 {required:.0f}s，"
                f"缺最后的合成步骤"))


def _dedupe_by_episode(pool: list[Candidate], want: int) -> list[Candidate]:
    """Keep one file per episode index; fall back to top-N when indices are absent.

    Guards the openmontage case, where `renders/ep1.mp4` and `renders/ep1_e1.mp4`
    are both plausible and both near target -- without this, one episode is
    counted twice and another is missed.
    """
    if want <= 0:
        want = len({c.episode for c in pool if c.episode} ) or min(len(pool), 3)
    chosen: list[Candidate] = []
    seen: set[int] = set()
    for c in pool:
        if c.episode is not None:
            if c.episode in seen:
                continue
            seen.add(c.episode)
        chosen.append(c)
        if len(chosen) >= want:
            break
    chosen.sort(key=lambda c: (c.episode if c.episode is not None else 99,
                               str(c.info.path)))
    return chosen


def _cross_check(chosen: list[Candidate], run: Run) -> list[str]:
    """Channel disagreement is a finding, not an error (EVAL_DESIGN.md 2.4)."""
    out: list[str] = []
    declared_names = set(run.declared_names)
    declared_names.update(c.info.path.name for c in chosen if 'episode_manifest' in c.channels)
    on_disk = {c.info.path.name for c in chosen}

    if declared_names:
        # A declared name that resolves nowhere under the run root at all --
        # checked by name, since harnesses declare at differing granularity.
        present = {p.name for p in run.root.rglob("*") if p.is_file()}
        missing = sorted(declared_names - present)
        if missing:
            out.append(f"声明了 {len(missing)} 个不存在的视频文件："
                       f"{', '.join(missing[:3])}")
        undeclared = sorted(on_disk - declared_names)
        if undeclared:
            out.append(f"{len(undeclared)} 个成片未被声明："
                       f"{', '.join(undeclared[:3])}")
    want = run.target.episodes
    if want and len(chosen) != want:
        out.append(f"目标 {want} 集，实际定位到 {len(chosen)} 集")
    # The declaration also carries numbers; check the ones that are checkable.
    decl = run.declaration or {}
    for e in decl.get("episodes") or []:
        if not isinstance(e, dict) or not e.get("file"):
            continue
        want_dur = e.get("duration")
        hit = next((c for c in chosen if c.info.path.name == Path(str(e["file"])).name), None)
        if hit and want_dur and hit.info.duration:
            if abs(hit.info.duration - float(want_dur)) > 1.0:
                out.append(f"{e['file']} 声称 {float(want_dur):.2f}s，"
                           f"实测 {hit.info.duration:.2f}s")
    return out
