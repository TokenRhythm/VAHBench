"""Dispatch the six check modules as subprocesses.

Not importable-as-libraries, and that is not a style choice. From
av_consistency/README.md: any transformers model and the `clip` package in one
process segfaults, both import orders tried. visual_consistency needs `clip`;
physical_plausibility needs `transformers`. Importing both into one orchestrator
crashes the whole run rather than one module.

So each module gets its own process, and the interpreter is configurable per
module because they also disagree about torch builds (av_consistency pins
2.5.1+cpu and forbids transformers entirely).

Failure is contained: a module that crashes, times out, or writes nothing
produces no measurements, and the items that depended on it abstain. One broken
module must not take a case down -- that is the difference between a benchmark
that reports 12/13 items and one that reports nothing.
"""
from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass
from dataclasses import field
from dataclasses import asdict
from dataclasses import is_dataclass
from pathlib import Path

from shared.paths import REPO_ROOT

REPO = REPO_ROOT

# Prefer the caller's configured interpreter or a repository virtual environment.
# Historical interpreter defaults remain keyed for existing replay helpers.
DEFAULT_INTERPRETERS = dict.fromkeys(
    ("spec", "artifact", "visual", "physical", "av", "seam"), sys.executable)

_ENTRY = {
    "spec": ("spec_checks", "run_spec_check.py"),
    "artifact": ("artifact_checks", "run_artifact_check.py"),
    "visual": ("visual_consistency", "run_consistency.py"),
    "physical": ("physical_plausibility", "run_physical.py"),
    "av": ("av_consistency", "run_audio_check.py"),
    "seam": ("seam_checks", "run_seam_check.py"),
}


def original_root() -> Path:
    """Where the paper's detectors are expected.

    In the original working tree they were a sibling of the evaluator, not
    part of it, and they are not in this repository. Put them under
    `detectors/`, or point VAB_EVAL_DETECTOR_ROOT at the directory that holds
    the six `*_checks` / `*_consistency` / `*_plausibility` folders.
    """
    env = os.environ.get("VAB_EVAL_DETECTOR_ROOT", "").strip()
    return Path(env).expanduser() if env else REPO / "detectors"


# The original entrypoints. An entry that exists always wins.
SCRIPTS = {m: original_root().joinpath(*rel) for m, rel in _ENTRY.items()}

# The 2026-11 reference reimplementation shipped with this repository. It is
# not the paper's detector set: see the Detectors section of README.md. `physical`
# has none, and the reference `visual` and `av` passes measure only part of
# what the originals did (no face identity, no dynamic range).
REFERENCE_ROOT = REPO / "reference_detectors"
REFERENCE_SCRIPTS = {m: REFERENCE_ROOT.joinpath(*_ENTRY[m])
                     for m in ("spec", "artifact", "visual", "av", "seam")}
REFERENCE_PROBE = "import numpy"
# Set by the launcher from the configuration; VAB_EVAL_REFERENCE_DETECTORS=0
# also turns the fallback off.
USE_REFERENCE = True


def reference_enabled() -> bool:
    env = os.environ.get("VAB_EVAL_REFERENCE_DETECTORS", "").strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    if env in ("1", "true", "yes", "on"):
        return True
    return bool(USE_REFERENCE)


def resolve_script(module: str) -> tuple[Path | None, str]:
    """(entrypoint, origin). origin is "original", "reference" or "missing"."""
    script = SCRIPTS.get(module)
    if script is not None and Path(script).is_file():
        return Path(script), "original"
    ref = REFERENCE_SCRIPTS.get(module)
    if ref is not None and reference_enabled() and ref.is_file():
        return ref, "reference"
    return None, "missing"


def detector_dir(module: str) -> Path:
    """Directory to put on sys.path to import a detector's own package."""
    script, _ = resolve_script(module)
    return (script or SCRIPTS[module]).parent

# Wall-clock ceilings. The model-backed modules decode and encode every sampled
# frame; the CPU-only ones do one decode pass. Long physical checks can exceed
# an hour on the supported CPU runtime while other videos are being evaluated.
# Allow those unchanged samples to finish instead of discarding valid work.
DEFAULT_TIMEOUTS = {
    "spec": 300, "artifact": 900, "visual": 3600, "physical": 7200, "av": 1800,
    "seam": 900,
}

# Detector semantics changed; old measurements cannot stand in for a rerun.
DETECTOR_REVISIONS = {"visual": 2, "physical": 2, "av": 2, "seam": 2}


@dataclass
class ModuleRun:
    module: str
    ok: bool
    report: dict | None = None
    out_path: Path | None = None
    seconds: float = 0.0
    error: str = ""
    cmd: list[str] = field(default_factory=list)
    origin: str = ""          # "original" or "reference"; reports stay untouched


# Where an interpreter for each module may live, in order. Environment
# overrides and repository virtual environments do not require source edits.
# An import probe decides, so a
# venv that exists but lacks the module's dependencies is passed over.
ENV_OVERRIDE = {m: f"VAB_EVAL_PY_{m.upper()}" for m in DEFAULT_INTERPRETERS}
_FAMILY = {"spec": "vc", "artifact": "vc", "visual": "vc", "physical": "vc", "av": "av",
           "seam": "vc"}
PROBES = {
    "spec": "import cv2, numpy",
    "artifact": "import cv2, numpy",
    "visual": "import clip, insightface, torch",
    "physical": "import transformers, torch",
    "av": "import torch, librosa",
    "seam": "import cv2, numpy",
}
_probe_cache: dict[tuple[str, str], bool] = {}


def _probe(exe: str, code: str) -> bool:
    key = (exe, code)
    if key not in _probe_cache:
        try:
            proc = subprocess.run([exe, "-c", code], capture_output=True, timeout=240)
            _probe_cache[key] = proc.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            _probe_cache[key] = False
    return _probe_cache[key]


def candidates(module: str) -> list[tuple[str, str]]:
    """(interpreter, how) in precedence order."""
    out: list[tuple[str, str]] = []
    env = os.environ.get(ENV_OVERRIDE.get(module, ""))
    if env:
        out.append((env, "env"))
    fam = _FAMILY.get(module, "vc")
    for base in (REPO / ".venvs" / fam, REPO / f"{fam}env", REPO.parent / f"{fam}env"):
        out.append((str(base / "Scripts" / "python.exe"), "repo_venv"))
        out.append((str(base / "bin" / "python"), "repo_venv"))
    if DEFAULT_INTERPRETERS.get(module):
        out.append((DEFAULT_INTERPRETERS[module], "default"))
    out.append((sys.executable, "current"))
    return out


def resolve_interpreter(module: str, overrides: dict | None = None,
                        probes: dict | None = None) -> tuple[str | None, str]:
    """The first candidate that exists and can import the module's
    dependencies, and where it came from; (None, "none") when nothing can."""
    if overrides and overrides.get(module):
        return overrides[module], "override"
    if probes:
        code = probes.get(module, "import sys")
    elif resolve_script(module)[1] == "reference":
        code = REFERENCE_PROBE          # numpy only: no OpenCV, no model weights
    else:
        code = PROBES.get(module, "import sys")
    for exe, how in candidates(module):
        if not exe or not (Path(exe).is_file() or exe == sys.executable):
            continue
        if _probe(exe, code):
            return exe, how
    return None, "none"


def _interpreter(module: str, overrides: dict | None) -> str | None:
    return resolve_interpreter(module, overrides)[0]


def build_command(module: str, video: Path, out: Path, *,
                  target=None, device: str = "cpu",
                  shots_json: Path | None = None,
                  reference_images: list[Path] | None = None,
                  scene_references: list[Path] | None = None,
                  source_clips: list[Path] | None = None,
               seam_times: list[float] | None = None,
               script: Path | None = None) -> list[str]:
    """Each module has its own flags; this is the only place that knows them."""
    script = script or resolve_script(module)[0] or SCRIPTS[module]
    cmd = [str(script), "--video", str(video), "--out", str(out)]

    if module == "spec":
        if target is not None:
            if target.seconds_per_episode:
                cmd += ["--target-duration", str(target.seconds_per_episode)]
            if target.resolution:
                cmd += ["--target-resolution", str(target.resolution).replace("×", "x")]
            if target.aspect_ratio:
                cmd += ["--target-aspect", str(target.aspect_ratio).replace("：", ":")]
        if source_clips:
            cmd += ["--source-clips", *[str(p) for p in source_clips]]
    elif module in ("visual", "physical"):
        cmd += ["--device", device]
        if shots_json:
            cmd += ["--shots-json", str(shots_json)]
        if module == "visual":
            # Identity and temporal-repeat measurements remain enabled. DINO
            # and CLIP scene proxies are not used by the fixed scoring rubric.
            if not reference_images and not scene_references:
                cmd += ["--no-visual"]
            if reference_images:
                cmd += ["--reference-images", *[str(p) for p in reference_images]]
            if scene_references:
                cmd += ["--scene-references", *[str(p) for p in scene_references]]
        else:
            cmd += ["--no-scale"]
    elif module == "av":
        # Full audio cleanliness is unchanged; unbound face/voice proxies
        # cannot support a score. Existing full reports are valid supersets.
        cmd += ["--device", device, "--no-visual"]
    elif module == "seam":
        # seams and source clips both come from the run itself (evidence.seam)
        if seam_times:
            cmd += ["--seam-times", ",".join(f"{t:.3f}" for t in seam_times)]
        if source_clips:
            cmd += ["--source-clips", *[str(p) for p in source_clips]]
    cmd.append("--quiet") if module != "av" else None
    return [c for c in cmd if c]


def run_module(module: str, video: Path, out_dir: Path, *,
               interpreters: dict | None = None,
               timeouts: dict | None = None,
               tag: str = "",
               **kw) -> ModuleRun:
    """`tag` distinguishes cached reports of the same module run with different
    arguments -- a visual pass seeded with reference images is cached as
    `visual_refs.json` beside the plain `visual.json`, not over it."""
    import time

    out_dir = _io_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    revision = f"_r{DETECTOR_REVISIONS[module]}" if module in DETECTOR_REVISIONS else ""
    script, origin = resolve_script(module)
    if script is None:
        return ModuleRun(module, False, error=f"找不到 {SCRIPTS[module]}")
    # A reference report never shares a cache file with an original one.
    suffix = "_ref" if origin == "reference" else ""
    out = _io_path(out_dir / f"{module}{tag}{revision}{suffix}.json")
    exe = _interpreter(module, interpreters)
    if exe is None:
        tried = "、".join(dict.fromkeys(e for e, _ in candidates(module)))
        return ModuleRun(module, False,
                         error=f"没有能导入 {module} 依赖的解释器（试过 {tried}）")
    meta_path = _io_path(out.with_suffix(".cache.json"))
    signature = _cache_signature(module, video, exe, kw)
    if out.is_file() and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            raw = out.read_bytes()
            if meta.get("signature") == signature and meta.get("report_sha256") == hashlib.sha256(raw).hexdigest():
                report = json.loads(raw)
                if isinstance(report, dict):
                    return ModuleRun(module, True, report, out, 0.0, origin=origin)
        except (OSError, json.JSONDecodeError):
            pass

    # A failed refresh must never read a previous report as today's result.
    import uuid
    pending = _io_path(out.with_name(f".pending-{uuid.uuid4().hex[:12]}.json"))
    cmd = [exe, *build_command(module, video, pending, script=script, **kw)]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    if module == "physical":
        # All physics backends use public weights. Some transformers versions
        # lose token=False in nested config loads, so also disable implicit auth.
        env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", env=env,
                              timeout=(timeouts or DEFAULT_TIMEOUTS).get(module, 900))
    except subprocess.TimeoutExpired:
        pending.unlink(missing_ok=True)
        return ModuleRun(module, False, seconds=time.time() - t0,
                         error="超时", cmd=cmd)
    except OSError as exc:
        pending.unlink(missing_ok=True)
        return ModuleRun(module, False, seconds=time.time() - t0,
                         error=f"{type(exc).__name__}: {exc}", cmd=cmd)
    dt = time.time() - t0
    out.with_suffix(".log").write_text(
        (proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")

    # A non-zero exit is not automatically a failure: spec_checks exits non-zero
    # when a video *fails a check*, which is a result, not a crash. The report
    # file is the arbiter.
    if pending.is_file():
        try:
            raw = pending.read_bytes()
            report = json.loads(raw)
            if not isinstance(report, dict):
                raise ValueError("报告必须是 JSON 对象")
            pending.replace(out)
            meta_path.write_text(json.dumps({"signature": signature,
                "report_sha256": hashlib.sha256(raw).hexdigest()}, indent=2), encoding="utf-8")
            return ModuleRun(module, True, report, out, dt, cmd=cmd, origin=origin)
        except (OSError, ValueError) as e:
            pending.unlink(missing_ok=True)
            return ModuleRun(module, False, seconds=dt, cmd=cmd,
                             error=f"报告无法解析：{e}")
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    return ModuleRun(module, False, seconds=dt, cmd=cmd,
                     error=f"rc={proc.returncode} " + (tail[-1][:200] if tail else "无输出"))


def _io_path(path):
    """Windows report/temp paths may exceed MAX_PATH in deeply nested runs."""
    path = Path(path)
    if os.name == "nt":
        absolute = os.path.abspath(path)
        if not absolute.startswith("\\\\?\\") and len(absolute) >= 230:
            absolute = ("\\\\?\\UNC\\" + absolute[2:] if absolute.startswith("\\\\")
                        else "\\\\?\\" + absolute)
            return Path(absolute)
    return path


def _cache_signature(module, video, exe, options):
    """Invalidate on media/reference changes, arguments, code or interpreter."""
    def identity(p):
        p = Path(p).resolve()
        try:
            stat = _io_path(p).stat()
            return {"path": str(p), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        except OSError:
            return {"path": str(p), "missing": True}

    def normal(value):
        if isinstance(value, Path):
            return identity(value)
        if is_dataclass(value):
            return normal(asdict(value))
        if isinstance(value, dict):
            return {k: normal(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [normal(v) for v in value]
        return value

    code = hashlib.sha256()
    script, origin = resolve_script(module)
    source_dirs = [(script or SCRIPTS[module]).parent]
    if origin == "reference":
        source_dirs.append(REFERENCE_ROOT / "vahref")      # shared decode and features
    for source_dir in source_dirs:
        for path in sorted(source_dir.rglob("*.py")):
            if any(part in {".venv", "venv", "__pycache__", "tests"} for part in path.relative_to(source_dir).parts):
                continue
            code.update(str(path.relative_to(source_dir)).encode())
            code.update(path.read_bytes())
    payload = {"version": 1, "module": module, "revision": DETECTOR_REVISIONS.get(module, 0),
               "origin": origin,
               "video": identity(video), "interpreter": identity(exe),
               "options": normal(options), "detector_code": code.hexdigest()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def unwrap_single(report: dict | None) -> dict | None:
    """Batch-mode reports wrap the per-video dict; single-video mode does not."""
    if not isinstance(report, dict):
        return report
    if "reports" in report and isinstance(report["reports"], list) and report["reports"]:
        return report["reports"][0]
    return report
