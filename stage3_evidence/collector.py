"""Evidence on demand. No stages.

An expectation asks the bus for what it needs -- a transcript, the identity
clusters of an episode, a visual pass seeded with reference images, the
images a trajectory produced, the subtitle file for an episode -- and the bus
computes it the first time and remembers it. Whether a module runs, and with
which arguments, is decided by what the expectations need, not by a pipeline
diagram. That is the concrete meaning of "not a rigid sequential flow".

The couplings that live here, none of which any single module could express:

* `character_sheets()` joins the brief's character names to images the
  trajectory produced (`assets/characters/maya.png`) -- preferring the ones
  the trajectory says it conditioned on -- and `seeded_visual()` hands them
  to visual_consistency as `--reference-images`. A check the task's
  text-only briefs made N/A becomes measurable on evidence the agent itself
  generated.
* `conditioning_images()` pairs each delivered clip with the keyframe the
  agent generated it from (`ep1_c03.png` <-> `ep1_c03.mp4`), and
  `segments()` turns the artifact module's splices into the shot boundaries
  the visual module uses -- so "does shot i match its own conditioning image"
  is asked about the right shot. Three modules, one question.
* `source_clips()` gives spec_checks the clips an episode was assembled from,
  so `retiming` -- the strongest padding detector -- can run at all.
* `transcript()` reports `silent=True` from the av module's rms before ever
  calling Whisper, so expectations about dialogue VOID on a silent track
  instead of passing "0 words < 125".
* `subtitles_for()` maps the agent's .srt/.vtt/.ass files onto episodes so
  the text it promised can be held against the audio it shipped.

Every configuration the bus performs is written to the flow ledger, so a
report says which upstream facts reconfigured which downstream module.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import stage4_checks.dispatch as runners
from stage2_delivery.discover import EP_NUM
from stage4_checks.reports import MIN_SEGMENT_FRAMES
from stage4_checks.reports import dig
from stage4_checks.cross_checks import Ledger


from shared.paths import PACKAGE_ROOT

WORKER = PACKAGE_ROOT / "stage3_evidence" / "workers" / "asr_worker.py"
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")
VIDEO_EXT = (".mp4", ".mov", ".mkv", ".webm")
SUB_EXT = (".srt", ".vtt", ".ass")
TEXT_EXT = (".md", ".txt", ".json", ".srt", ".ass", ".js", ".yaml", ".yml")

CLIP_NUM = re.compile(r"(?:^|[_\-])(?:clip|shot|seg|segment|scene|sc|c|s|part)"
                      r"[_\-]?(\d{1,3})(?!\d)", re.I)
# The `e1s10` / `e1_s1_hook` scheme (openmontage runs: clip_e1s10.mp4,
# seg_e1s10.mp4, e1_s1_hook.mp4): episode and scene in one token, no "ep".
# Used when the shared episode forms find nothing; the compact e1s10 form
# has no token boundary after its episode number.
EP_NUM_ES = re.compile(r"(?:^|[^a-z0-9])e(\d{1,2})[_\-]?s\d{1,3}(?!\d)", re.I)
CLIP_NUM_ES = re.compile(r"(?:^|[^a-z0-9])e\d{1,2}[_\-]?s(\d{1,3})(?!\d)", re.I)
MASTER = re.compile(r"(^|[_\-])(raw|master|nosub|uncut)([_\-]|$)", re.I)
SCRATCH = re.compile(r"(^|/)(working|cache|tmp|temp|scratch|_work|work)(/|$)", re.I)


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def episode_of(p: Path) -> int | None:
    for s in (p.stem, p.parent.name):
        m = EP_NUM.search(s)
        if m:
            return int(m.group(1))
    m = EP_NUM_ES.search(p.stem)
    return int(m.group(1)) if m else None


def clip_index(p: Path) -> int | None:
    m = CLIP_NUM.search(p.stem) or CLIP_NUM_ES.search(p.stem)
    return int(m.group(1)) if m else None


# --------------------------------------------------------- subtitles ----
_TS = re.compile(r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")


def _ts(m) -> float:
    h, mi, s, frac = m.groups()
    ms = int(frac) * (10 ** (3 - len(frac)))
    return int(h or 0) * 3600 + int(mi) * 60 + int(s) + ms / 1000.0


def parse_subtitles(text: str, suffix: str) -> list[dict]:
    """.srt / .vtt / .ass -> [{start, end, text}] in seconds."""
    cues: list[dict] = []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if suffix == ".ass":
        for line in text.splitlines():
            if not line.startswith("Dialogue:"):
                continue
            parts = line.split(",", 9)
            if len(parts) < 10:
                continue
            a, b = _TS.search(parts[1]), _TS.search(parts[2])
            body = re.sub(r"\{[^}]*\}", "", parts[9]).replace("\\N", " ").replace("\\n", " ")
            if a and b:
                cues.append({"start": _ts(a), "end": _ts(b), "text": body.strip()})
        return cues
    for blk in re.split(r"\n\s*\n", text):
        lines = [ln for ln in blk.strip().splitlines() if ln.strip()]
        ti = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if ti is None:
            continue
        a, b = lines[ti].split("-->", 1)
        ma, mb = _TS.search(a), _TS.search(b)
        if not ma or not mb:
            continue
        body = " ".join(re.sub(r"<[^>]+>", "", ln).strip() for ln in lines[ti + 1:])
        cues.append({"start": _ts(ma), "end": _ts(mb), "text": body.strip()})
    return cues


_SIL_START = re.compile(r"silence_start:\s*([0-9.]+)")
_SIL_END = re.compile(r"silence_end:\s*([0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)")


def parse_silencedetect(text: str) -> list[dict]:
    """ffmpeg's silencedetect log -> [{start, end, duration}]. A run that is
    still open when the stream ends has no silence_end line; it is dropped
    rather than guessed."""
    runs, start = [], None
    for line in text.splitlines():
        m = _SIL_START.search(line)
        if m:
            start = float(m.group(1))
            continue
        m = _SIL_END.search(line)
        if m and start is not None:
            end, dur = float(m.group(1)), float(m.group(2))
            runs.append({"start": round(start, 3), "end": round(end, 3),
                         "duration": round(dur, 3)})
            start = None
    return runs


def _ffmpeg() -> str | None:
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    probe = shutil.which("ffprobe")
    if probe:
        cand = Path(probe).with_name("ffmpeg" + Path(probe).suffix)
        if cand.is_file():
            return str(cand)
    return None


class EvidenceBus:
    def __init__(self, run, delivery, out_dir: Path, *, device: str = "cpu",
                 asr_model: str = "base", facts=None, ledger: Ledger | None = None):
        self.run, self.delivery, self.device = run, delivery, device
        self.asr_model = asr_model
        self.facts = facts
        self.ledger = ledger if ledger is not None else Ledger()
        self.dir = Path(out_dir) / "modules" / _safe(run.run_id)
        self._memo: dict = {}
        self.log: list[str] = []          # what was computed, in demand order
        self.seeds: dict[str, dict] = {}  # per video: what seeded the visual pass
        self.sheet_sources: dict[str, str] = {}
        self.errors: dict[tuple, str] = {}   # (module, video, tag) -> why it failed
        self.tags: dict[tuple, str] = {}     # (module, video) -> cache tag actually used

    # ------------------------------------------------------------ helpers --
    def _vdir(self, video: Path) -> Path:
        video = Path(video)
        same_stem = [c for c in self.delivery.videos if Path(c.info.path).stem == video.stem]
        # Independent scenes may all be called final_video.mp4 (real ViMax
        # runs). Their measurements must coexist, not overwrite one another.
        stem = video.stem
        if len(same_stem) > 1:
            stem += "_" + self._sig(video.resolve())
        d = runners._io_path(self.dir / stem)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _remember(self, key, fn):
        if key not in self._memo:
            self._memo[key] = fn()
            self.log.append(str(key))
        return self._memo[key]

    def _is_scratch(self, p: Path) -> bool:
        try:
            rel = str(p.parent.relative_to(self.run.root)).replace("\\", "/")
        except ValueError:
            rel = str(p.parent)
        return bool(SCRATCH.search(rel))

    def _episode(self, video: Path) -> int | None:
        for c in self.delivery.videos:
            if Path(c.info.path) == Path(video) and c.episode:
                return c.episode
        return episode_of(Path(video))

    # ----------------------------------------------------------- modules --
    def module(self, name: str, video: Path, **kw) -> dict | None:
        """A module report, computed once per (module, video, tag)."""
        tag = kw.pop("tag", "")
        key = (name, str(video), tag)

        def go():
            r = runners.run_module(name, Path(video), self._vdir(video),
                                   target=self.run.target, device=self.device,
                                   tag=tag, **kw)
            if not r.ok:
                self.errors[key] = r.error or "failed"
                return None
            return runners.unwrap_single(r.report)
        return self._remember(key, go)

    def status(self, name: str, video: Path, tag: str | None = None) -> str:
        if tag is None:
            tag = self.tags.get((name, str(video)), "")
        key = (name, str(video), tag)
        if key not in self._memo:
            return "not_run"
        return "ok" if self._memo[key] is not None else self.errors.get(key, "failed")

    @staticmethod
    def _sig(*parts) -> str:
        """Cache tag for a seeded pass: the report file is named after what
        seeded it, so a change in the seeds (a keyframe found, a shot boundary
        moved) is a new file rather than a stale hit."""
        import hashlib
        return hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:8]

    def av(self, video: Path) -> dict | None:
        return self.module("av", video)

    def artifact(self, video: Path) -> dict | None:
        return self.module("artifact", video)

    def _shots_tag(self, shots: Path | None) -> str:
        return f"_shots_{self._sig(shots.read_text(encoding='utf-8'))}" if shots else ""

    def _on_shots(self, name: str, video: Path, dst: str) -> dict | None:
        """A module run on the pixel-recovered shot boundaries when there are
        any. Every cross-shot number in the rubric -- identity, scale, scene
        -- is only comparable between modules when they agree what a cut is,
        and until this every module cut the video its own way."""
        shots = self.shot_boundaries(video)
        tag = self._shots_tag(shots)
        self.tags[(name, str(video))] = tag
        if shots and (name, str(video), tag) not in self._memo:
            self.ledger.add("splices_seed_shots", ["artifact.splices"], dst, "configure",
                            f"{Path(video).name}: 像素拼接点作为 {name} 模块的镜头边界")
        return self.module(name, video, shots_json=shots, tag=tag)

    def visual(self, video: Path) -> dict | None:
        return self._on_shots("visual", video, "visual.identity_consistency")

    def physical(self, video: Path) -> dict | None:
        return self._on_shots("physical", video, "physical.penetration")

    def spec(self, video: Path) -> dict | None:
        """spec_checks, with the episode's source clips when the trajectory
        left them -- the only way `retiming` gets past "undetermined"."""
        srcs = self.source_clips(video)
        tag = f"_v2_src_{self._sig(*srcs)}" if srcs else "_v2"
        self.tags[("spec", str(video))] = tag
        if srcs:
            return self.module("spec", video, source_clips=list(srcs), tag=tag)
        return self.module("spec", video, tag=tag)

    def seam(self, video: Path) -> dict | None:
        """seam_checks on the reconciled segments -- the same cuts every other
        module gets -- and, when the trajectory left one source clip per
        segment, on those clips too, which is what turns "the cut lands on a
        still" into "and the agent had a second of slack it did not use".
        No seams (one segment) means nothing to test; the module says so."""
        segs = self.reconciled_segments(video)[0]
        if not segs:
            # No splice structure (the artifact report is missing or carries
            # none). Running the module without seam times would make it
            # answer "single segment, nothing to test" -- a statement about
            # the film that nothing measured. Unknown stays unknown.
            key = ("seam", str(video), "")
            self.tags[("seam", str(video))] = ""
            if key not in self._memo:
                self._memo[key] = None
                self.errors[key] = "切点结构未知：artifact 报告不可用，切点连续性未测"
                self.log.append(str(key))
            return None
        times = [round(float(s["end"]), 3) for s in (segs or [])[:-1]] if segs and len(segs) >= 2 else []
        clips = self.episode_clips(video) if times else []
        srcs = list(clips) if clips and len(clips) == len(segs) else []
        tag = f"_{self._sig(*times, *srcs)}" if times else ""
        self.tags[("seam", str(video))] = tag
        if times and ("seam", str(video), tag) not in self._memo:
            self.ledger.add("splices_seed_seam", ["artifact.splices"], "seam.cut_continuity",
                            "configure", f"{Path(video).name}: {len(times)} 个拼接点交给 seam_checks 做手柄检验")
            if srcs:
                self.ledger.add("clips_seed_seam", ["produced"], "seam.cut_continuity", "configure",
                                f"{Path(video).name}: {len(srcs)} 个源片段交给 seam_checks 做裁剪余量归因")
            elif clips:
                self.ledger.add("clips_seed_seam", ["produced"], "seam.cut_continuity", "annotate",
                                f"{Path(video).name}: 源片段 {len(clips)} 个与段数 {len(segs)} 不符，"
                                "裁剪余量不归因")
        return self.module("seam", video, seam_times=times or None, source_clips=srcs or None, tag=tag)

    # -------------------------------------------------------- transcript --
    def silent(self, video: Path) -> bool | None:
        av = self.av(video)
        if not av:
            return None
        if dig(av, "audio_stream.present") is False:
            return True
        rms, sig = dig(av, "cleanliness.rms"), dig(av, "cleanliness.signal_db")
        if rms is None and sig is None:
            return None
        return bool((rms is not None and float(rms) < 1e-4)
                    or (sig is not None and float(sig) < -90))

    def transcript(self, video: Path) -> dict | None:
        """{"text", "n_words", "segments", "silent", "model"} or None."""
        key = ("asr", str(video))

        def go():
            if self.silent(video):
                absent = dig(self.av(video), "audio_stream.present") is False
                return {"text": "", "n_words": 0, "segments": [], "silent": True,
                        "audio_present": not absent,
                        "model": "skipped:no_audio" if absent else "skipped:silent"}
            # New timing semantics must not silently reuse segment-only ASR.
            # Include model and media identity; preserve historical asr.json.
            try:
                stat = video.stat()
            except OSError:
                return None
            tag = self._sig(video, stat.st_size, stat.st_mtime_ns,
                            self.asr_model, "word-timestamps-v1")
            out = self._vdir(video) / f"asr_words_{tag}.json"
            if not out.is_file():
                # The worker imports whisper and nothing else of the av
                # module, so that is what the interpreter is probed for.
                exe = runners.resolve_interpreter("av", probes={"av": "import whisper"})[0]
                if not exe or not Path(exe).is_file() or not WORKER.is_file():
                    return None
                cmd = [exe, str(WORKER), "--video", str(video), "--out", str(out),
                       "--model", self.asr_model]
                try:
                    subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=1800,
                                   env={**os.environ, "PYTHONIOENCODING": "utf-8"})
                except subprocess.TimeoutExpired:
                    return None
                if not out.is_file():
                    return None
            try:
                d = json.loads(out.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
            d["silent"] = False
            return d
        return self._remember(key, go)

    # ----------------------------------------------------------- silence --
    def silence_runs(self, video: Path, noise_db: float = -45.0,
                     min_s: float = 3.0) -> list[dict] | None:
        """Runs of near-nothing on the track, from ffmpeg's silencedetect.
        Deterministic, no venv; cache includes media identity and parameters.
        None when the input cannot be measured."""
        video = Path(video)
        try:
            stat = runners._io_path(video).stat()
        except OSError:
            return None
        tag = self._sig(video.resolve(), stat.st_size, stat.st_mtime_ns,
                        noise_db, min_s, "silence-v2")
        key = ("silence", tag)

        def go():
            out = self._vdir(video) / f"silence_{tag}.json"
            if out.is_file():
                try:
                    return json.loads(out.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    pass
            exe = _ffmpeg()
            if not exe:
                return None
            cmd = [exe, "-nostats", "-hide_banner", "-i", str(video), "-vn",
                   "-af", f"silencedetect=noise={noise_db}dB:d={min_s}", "-f", "null", "-"]
            try:
                proc = subprocess.run(cmd, capture_output=True, encoding="utf-8",
                                      errors="replace", timeout=600)
            except (subprocess.TimeoutExpired, OSError):
                return None
            if proc.returncode != 0:
                return None
            runs = parse_silencedetect(proc.stderr or "")
            out.write_text(json.dumps({"noise_db": noise_db, "min_s": min_s, "runs": runs},
                                      indent=1), encoding="utf-8")
            return {"noise_db": noise_db, "min_s": min_s, "runs": runs}
        got = self._remember(key, go)
        return None if got is None else got.get("runs", [])

    # ------------------------------------------------------- trajectory --
    def files(self, exts: tuple[str, ...], max_n: int = 5000) -> list[Path]:
        key = ("files", exts)

        def go():
            out = []
            for p in self.run.root.rglob("*"):
                if len(out) >= max_n:
                    break
                if p.is_file() and p.suffix.lower() in exts \
                        and not any(part.startswith((".", "_"))
                                    for part in p.relative_to(self.run.root).parts):
                    out.append(p)
            return out
        return self._remember(key, go)

    def produced_images(self) -> list[Path]:
        """Images the run produced: trajectory `produced` edges first, then the
        filesystem -- the trajectory may not record every file it wrote."""
        out: dict[str, Path] = {}
        for a in (getattr(self.facts, "produced", None) or []):
            p = Path(a.path)
            if p.suffix.lower() in IMAGE_EXT and p.is_file():
                out[str(p.resolve())] = p
        for p in self.files(IMAGE_EXT):
            out.setdefault(str(p.resolve()), p)
        return list(out.values())

    def delivered_videos(self) -> list[Path]:
        return [c.info.path for c in self.delivery.videos] if self.delivery.found else []

    def produced_clips(self) -> list[Path]:
        """Video files other than the delivered episodes."""
        key = ("clips",)

        def go():
            delivered = {Path(v).resolve() for v in self.delivered_videos()}
            out: dict[str, Path] = {}
            for a in (getattr(self.facts, "produced", None) or []):
                p = Path(a.path)
                if p.suffix.lower() in VIDEO_EXT and p.is_file() \
                        and p.resolve() not in delivered:
                    out[str(p.resolve())] = p
            for p in self.files(VIDEO_EXT):
                if p.resolve() not in delivered:
                    out.setdefault(str(p.resolve()), p)
            return list(out.values())
        return self._remember(key, go)

    def episode_clips(self, video: Path) -> list[Path]:
        """Clip-like files of this episode (`ep1_c01.mp4` ...) in clip order,
        one per index. A copy outside a scratch tree beats one inside it
        (`assets/clips/ep1_c01.mp4` over `assets/work/ep1_c01_n.mp4`)."""
        ep = self._episode(video)
        if ep is None:
            return []
        best: dict[int, Path] = {}
        for p in self.produced_clips():
            if episode_of(p) != ep:
                continue
            idx = clip_index(p)
            if idx is None:
                continue
            cur = best.get(idx)
            if cur is None or (self._is_scratch(cur), str(cur)) > (self._is_scratch(p), str(p)):
                best[idx] = p
        return [best[i] for i in sorted(best)]

    def source_clips(self, video: Path) -> list[Path]:
        """What spec.retiming compares the delivery against: the episode's
        clips, or failing that a `<ep>_raw` master anywhere in the run."""
        key = ("source_clips", str(video))

        def go():
            assembly = self.assembly_evidence(video)
            if assembly and assembly.get("verified"):
                self.ledger.add("clips_seed_retiming", ["produced"], "spec.retiming",
                                "configure", f"{Path(video).name}: 使用已执行装配记录中的有序源片段")
                return [Path(p) for p in assembly["sources"]]
            manifest = self.concat_sources(video)
            if manifest:
                self.ledger.add("clips_seed_retiming", ["concat_manifest"], "spec.retiming",
                                "configure", f"{Path(video).name}: 使用有序拼接清单，保留重复引用")
                return manifest
            clips = self.episode_clips(video)
            if clips:
                self.ledger.add("clips_seed_retiming", ["produced"], "spec.retiming",
                                "configure",
                                f"{Path(video).name}: {len(clips)} 个源片段交给 spec_checks 做 retiming 比对")
                return clips
            ep = self._episode(video)
            if ep is None:
                return []
            for p in sorted(self.produced_clips(), key=lambda q: (self._is_scratch(q), str(q))):
                if episode_of(p) == ep and MASTER.search(p.stem):
                    self.ledger.add("clips_seed_retiming", ["produced"], "spec.retiming",
                                    "configure",
                                    f"{Path(video).name}: 母版 {p.name} 交给 spec_checks 做 retiming 比对")
                    return [p]
            return []
        return self._remember(key, go)

    def assembly_evidence(self, video: Path) -> dict | None:
        from stage1_inputs.extractors import _read_tool_events
        from stage4_checks.process.honesty import executed_assemblies
        records = self._remember(("executed_assemblies",),
                                 lambda: executed_assemblies(self.run.root, _read_tool_events(self.run.root)[0]))
        return records.get(str(video.resolve()).casefold())

    def concat_sources(self, video: Path) -> list[Path]:
        """Resolve an unambiguous episode concat list, never execute it.

        Relocated runs retain old absolute paths and $WORKSPACE_DIR. Resolve
        only unique suffix matches inside this run. A missing entry, trim
        directive, or conflicting candidate list makes the evidence unknown.
        These are candidate sources; the spec/pixel checks still validate
        correspondence, rather than trusting the manifest as a hard cap.
        """
        ep = self._episode(video)
        if ep is None:
            return []
        root = self.run.root.resolve()
        pool = [p.resolve() for p in self.produced_clips() if p.resolve().is_relative_to(root)]
        sequences = set()
        candidates = {}
        for manifest in self.files((".txt", ".ffconcat")):
            if episode_of(manifest) != ep:
                continue
            try:
                lines = manifest.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError):
                continue
            paths = []
            valid = True
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#") or line == "ffconcat version 1.0":
                    continue
                match = re.fullmatch(r"file\s+(['\"])(.*?)\1", line)
                if not match:
                    valid = False
                    break
                raw = match[2].replace("\\", "/")
                parts = raw.split("/")
                hits = []
                for n in range(len(parts), 0, -1):
                    suffix = "/".join(parts[-n:]).lower()
                    hits = [p for p in pool if p.as_posix().lower().endswith("/" + suffix)]
                    if hits:
                        break
                if len(hits) != 1:
                    valid = False
                    break
                paths.append(hits[0])
            if valid and paths:
                key = tuple(str(p) for p in paths)
                sequences.add(key)
                candidates[key] = paths
        return candidates[next(iter(sequences))] if len(sequences) == 1 else []

    # ---------------------------------------------- pixels -> structure --
    def segments(self, video: Path) -> list[dict] | None:
        """Major segments from the artifact module's splices, in seconds.
        Slivers (< MIN_SEGMENT_FRAMES) are join glitches and merge into the
        segment before them."""
        rep = self.artifact(video)
        if not rep:
            return None
        fps = float(dig(rep, "meta.fps", "video_meta.fps") or 24.0)
        frames = dig(rep, "splices.segment_frames")
        if not isinstance(frames, list) or not frames:
            return None
        segs: list[dict] = []
        t = 0.0
        for n in frames:
            n = int(n)
            dur = n / fps
            if n < MIN_SEGMENT_FRAMES and segs:
                segs[-1]["end"] += dur
                segs[-1]["frames"] += n
            else:
                segs.append({"start": t, "end": t + dur, "frames": n})
            t += dur
        return segs

    def expected_seams(self, video: Path, n_clips: int) -> list[float] | None:
        """Where the trajectory says the cuts are: cumulative declared clip
        lengths (one per generation request when the log has them, else the
        typical length), for this episode. None when the trajectory never
        recorded clip lengths."""
        req = getattr(self.facts, "requested", None)
        durs = list(getattr(req, "clip_durations", None) or [])
        if not durs or n_clips < 2:
            return None
        clips = getattr(getattr(self.facts, "claims", None), "declared_clips_per_episode", None) or []
        ep = self._episode(video)
        ep_durs: list[float] | None = None
        if clips and ep and ep - 1 < len(clips) and len(durs) == sum(clips) \
                and clips[ep - 1] == n_clips:
            start = sum(clips[: ep - 1])
            ep_durs = [float(x) for x in durs[start: start + n_clips]]
        if ep_durs is None:
            typical = sum(float(x) for x in durs) / len(durs)
            ep_durs = [typical] * n_clips
        seams, t = [], 0.0
        for d in ep_durs[:-1]:
            t += d
            seams.append(round(t, 3))
        return seams

    def segments_for(self, video: Path, n_want: int) -> tuple[list[dict] | None, list[str]]:
        """Segments reconciled to an expected clip count, two ways.

        1. By the trajectory's declared structure. With clip lengths on record
           the seams are known in advance; a pixel cut that sits at none of
           them (claudecode ep3: a cut at 47.6 s inside the fifth 10 s clip)
           is not a clip boundary for the shot <-> keyframe pairing, and the
           pieces on either side are one clip again. Each expected seam takes
           the nearest measured cut within a second.
        2. Failing that, by size. A bad seam can shatter one clip into pieces
           that survive the sliver cut -- ep1's sixth clip came back as
           17+4+4+25+4+186 frames -- so the shortest segments under two
           seconds are merged into a neighbour until the counts agree.

        Every merge is returned as a note. Nothing two seconds or longer is
        merged by size, and a mismatch that survives is reported, not forced.
        """
        segs = self.segments(video)
        if not segs:
            return None, []
        segs = [dict(s) for s in segs]
        notes: list[str] = []
        seams = self.expected_seams(video, n_want) if len(segs) > n_want else None
        if seams:
            starts = [s["start"] for s in segs[1:]]
            keep: set[float] = set()
            for e in seams:
                near = [b for b in starts if abs(b - e) <= 1.0]
                if near:
                    keep.add(min(near, key=lambda b: abs(b - e)))
            if len(keep) == n_want - 1:
                merged: list[dict] = [dict(segs[0])]
                for s in segs[1:]:
                    if s["start"] in keep:
                        merged.append(dict(s))
                    else:
                        merged[-1]["end"] = s["end"]
                        merged[-1]["frames"] += s["frames"]
                        notes.append(f"{Path(video).name}: {s['start']:.1f}s 处的像素切点不在"
                                     f"声明片长的任何接缝上，按声明结构并回前一段")
                return merged, notes
        short = 2.0
        while len(segs) > n_want:
            i = min(range(len(segs)), key=lambda k: segs[k]["frames"])
            dur = segs[i]["end"] - segs[i]["start"]
            if dur >= short:
                break
            j = i - 1 if i > 0 else i + 1          # into the previous segment, else the next
            a, b = sorted((i, j))
            merged = {"start": segs[a]["start"], "end": segs[b]["end"],
                      "frames": segs[a]["frames"] + segs[b]["frames"]}
            notes.append(f"{Path(video).name}: {segs[i]['frames']} 帧的碎段并入相邻段")
            segs[a:b + 1] = [merged]
        return segs, notes

    def reconciled_segments(self, video: Path) -> tuple[list[dict] | None, list[str], bool]:
        """Segments, reconciled to the conditioning keyframes when the
        trajectory left one per clip; plain splice segments otherwise. The
        flag says whether keyframe i can be paired with segment i."""
        key = ("reconciled_segments", str(video))

        def go():
            scene = self.conditioning_images(video)
            if scene:
                segs, notes = self.segments_for(video, len(scene))
                if segs and len(segs) == len(scene):
                    return segs, notes, True
                return self.segments(video), notes, False
            return self.segments(video), [], False
        return self._remember(key, go)

    def shot_boundaries(self, video: Path) -> Path | None:
        """The one shot definition every module gets, as a shots file. None
        when the pixels recovered fewer than two segments -- then each module
        falls back to its own scene detection, and says so in its report."""
        key = ("shot_boundaries", str(video))

        def go():
            segs = self.reconciled_segments(video)[0]
            return self.shots_json(video, segs) if segs and len(segs) >= 2 else None
        return self._remember(key, go)

    def shots_json(self, video: Path, segs: list[dict] | None = None) -> Path | None:
        segs = segs if segs is not None else self.segments(video)
        if not segs or len(segs) < 2:
            return None
        p = self._vdir(video) / "shots_from_splices.json"
        text = json.dumps({"shots": [
            {"shot_id": i, "start": round(s["start"], 3), "end": round(s["end"], 3)}
            for i, s in enumerate(segs)]}, indent=1)
        # Downstream signatures include input-file identity. Rewriting an
        # unchanged generated sidecar would invalidate every fresh bus/run.
        if not p.is_file() or p.read_text(encoding='utf8') != text:
            p.write_text(text, encoding='utf8')
        return p

    # ----------------------------------------------- trajectory -> refs --
    def _resolve_asset(self, raw):
        from stage4_checks.process.honesty import resolve_claim
        return self._remember(("resolve_asset", str(raw)),
                              lambda: resolve_claim(self.run.root, raw)[0])

    def _declared_references(self) -> list[Path]:
        """Images the trajectory says it conditioned on, resolved against
        unique full/relocated paths. A shared basename is not a binding."""
        refs = set()
        for c in (getattr(self.facts, "conditioning", None) or []):
            for r in (getattr(c, "references", None) or []):
                path = self._resolve_asset(r)
                if path and path.suffix.lower() in IMAGE_EXT:
                    refs.add(path)
        return sorted(refs)

    def reference_clips(self, video: Path) -> list[Path]:
        """Inventory for asset coverage, including directory-scoped shot outputs.

        A scene-local shot manifest identifies clips even when every output is
        called video.mp4. This inventory is not a pixel/timing correspondence.
        """
        assembly = self.assembly_evidence(video)
        if assembly and assembly.get("verified"):
            return [Path(p) for p in assembly["sources"]]
        clips = self.episode_clips(video)
        if clips:
            return clips
        shots = Path(video).parent / "shots"
        out = []
        for record in shots.glob("*/shot_description.json"):
            try:
                description = json.loads(record.read_text(encoding="utf8"))
                idx = int(record.parent.name)
                clip = record.parent / "video.mp4"
                if description.get("idx") == idx and clip.is_file():
                    out.append((idx, clip))
            except (OSError, ValueError, TypeError):
                continue
        return [p for _, p in sorted(out)]

    def character_sheets(self, characters) -> dict[str, Path]:
        """brief character -> an image the trajectory named after them.

        Images the trajectory declares as conditioning references are searched
        first, then everything it produced. Match on a name token as a whole
        word in the filename or its parent directory
        (`assets/characters/maya.png`, `maya_chen_sheet.png`); one image per
        character, the earliest by name, extra candidates logged not guessed.
        """
        key = ("sheets", tuple(getattr(c, "name", str(c)) for c in characters))

        def go():
            pools = ((self._declared_references(), "conditioning"),
                     (self.produced_images(), "filesystem"))
            sheets: dict[str, Path] = {}
            for c in characters:
                toks = getattr(c, "tokens", None) or [c.first]
                for pool, src in pools:
                    hits: list[Path] = []
                    for tok in toks:
                        pat = re.compile(rf"(^|[^a-z]){re.escape(tok)}([^a-z]|$)", re.I)
                        hits = sorted(p for p in pool
                                      if pat.search(p.stem) or pat.search(p.parent.name))
                        if hits:
                            break
                    if hits:
                        sheets[c.name] = hits[0]
                        self.sheet_sources[c.name] = src
                        if len(hits) > 1:
                            self.log.append(f"{c.name}: {len(hits)} candidate sheets, "
                                            f"using {hits[0].name}")
                        break
            return sheets
        return self._remember(key, go)

    def keyframes_for_clips(self, video: Path) -> dict[int, Path]:
        """clip index -> the image with the clip's stem (`ep1_c03.png` for
        `ep1_c03.mp4`): the frame the agent conditioned image-to-video on.
        Partial by design; `conditioning_images` demands completeness."""
        clips = self.reference_clips(video)
        if not clips:
            return {}
        by_stem: dict[str, list[Path]] = {}
        for p in self.produced_images():
            by_stem.setdefault(p.stem.lower(), []).append(p)
        out: dict[int, Path] = {}
        indices = [clip_index(c) for c in clips]
        if None in indices or len(set(indices)) != len(indices):
            indices = list(range(1, len(clips) + 1))
        for idx, c in zip(indices, clips):
            stem = c.stem.lower()
            bound = set()
            for edge in getattr(self.facts, "conditioning", None) or []:
                if not edge.target:
                    continue
                target = self._resolve_asset(edge.target)
                if target != c.resolve():
                    continue
                for raw in edge.references or []:
                    ref = self._resolve_asset(raw)
                    if ref and ref.suffix.lower() in IMAGE_EXT:
                        bound.add(ref)
            if bound:
                out[idx] = sorted(bound)[0]
                continue
            frame = c.parent / "first_frame.png"
            selector = c.parent / "first_frame_selector_output.json"
            if c.name == "video.mp4" and frame.is_file() and selector.is_file():
                try:
                    selected = json.loads(selector.read_text(encoding="utf8"))
                    if selected.get("text_prompt") and isinstance(selected.get("reference_image_path_and_text_pairs"), list):
                        out[idx] = frame
                        continue
                except (OSError, ValueError, TypeError):
                    pass
            # A frame reused from the previous shot has no new selector call.
            # The shot manifest still names its first-frame design; count the
            # actual local asset, not a second image-generation request.
            if c.name == "video.mp4" and frame.is_file():
                try:
                    shot = json.loads((c.parent / "shot_description.json").read_text(encoding="utf8"))
                    if shot.get("idx") == int(c.parent.name) and shot.get("ff_desc"):
                        out[idx] = frame
                        continue
                except (OSError, ValueError, TypeError):
                    pass
            cands = by_stem.get(stem) or by_stem.get(re.sub(r"_(n|norm|raw)$", "", stem)) or []
            local = [p for p in cands if p.parent == c.parent]
            cands = local or cands
            if len(cands) == 1:
                out[idx] = cands[0]
        return out

    def conditioning_images(self, video: Path) -> list[Path]:
        """One image per clip, in clip order, or nothing. The visual module
        maps reference i to shot i by order, so a single missing keyframe
        would put every later shot against the wrong image."""
        clips = self.episode_clips(video)
        kf = self.keyframes_for_clips(video)
        if not clips or len(kf) != len(clips):
            return []
        return [kf[clip_index(c)] for c in clips]

    # The plan's room per delivered shot used to be written out here as
    # `declared_locations.json` and handed to the visual module as
    # `--declared-locations`, so it could report `declared_grouping` for
    # `visual.scene_grouping`. That check was deleted 2026-09-22 (its
    # contradiction count scored at AUC ~0.5), and a seed nothing reads is
    # runtime spent for no measurement; the module still accepts the flag.

    # ---------------------------------------------------------- plan text --
    def plan_spans(self, video: Path) -> list[dict]:
        """The agent's shot plan for *this* episode: [{start, end, text}].

        This is the primitive every text <-> module link is built on. A
        module reports what happened at a time; the plan says what was meant
        to be there. Reading the two together needs no module flag -- the
        join is done here, on the module's own timestamps. Filtered to the
        video's episode, because every episode restarts the clock: another
        episode's plan over this one's cuts is worse than none.
        """
        key = ("plan_spans", str(video))

        def go():
            plan = getattr(getattr(self.facts, "continuity", None), "shot_plan", None) or []
            ep = self._episode(video)
            if ep is not None and any(getattr(p, "episode", None) for p in plan):
                plan = [p for p in plan if p.episode == ep]
            return [{"start": float(p.start), "end": float(p.end), "text": str(p.text or "")}
                    for p in plan]
        return self._remember(key, go)

    def text_at(self, video: Path, start: float, end: float) -> list[str]:
        """Plan text for the beats overlapping [start, end], in plan order."""
        return [s["text"] for s in self.plan_spans(video)
                if min(s["end"], float(end)) - max(s["start"], float(start)) > 0]

    def seeded_visual(self, video: Path, characters, brief=None,
                      facts=None) -> dict | None:
        """One visual pass per episode seeded with everything upstream can
        supply -- character sheets as --reference-images, per-clip keyframes
        as --scene-references, splice-derived shots as --shots-json. Computed
        once; both facets of O3 read it. None when there is nothing to seed."""
        key = ("visual_seeded", str(video))

        def go():
            name = Path(video).name
            sheets = self.character_sheets(characters)
            refs = list(sheets.values())
            scene = self.conditioning_images(video)
            segs, notes, aligned = self.reconciled_segments(video)
            self.log.extend(notes)
            shots = self.shot_boundaries(video)
            if scene and not aligned:
                self.log.append(f"{name}: {len(scene)} 张条件关键帧，但像素恢复 "
                                f"{len(segs) if segs else 0} 段，顺序无法对齐，场景参考未启用")
                scene = []
            elif scene and shots:
                self.ledger.add("splices_seed_shots", ["artifact.splices"],
                                "visual.scene_reference", "configure",
                                f"{name}: 像素拼接点给出 {len(segs)} 段，作为视觉模块的镜头边界"
                                + (f"（并入 {len(notes)} 个碎段）" if notes else ""))
            self.seeds[str(video)] = {
                "sheets": {k: str(v) for k, v in sheets.items()},
                "sheet_sources": {k: self.sheet_sources.get(k) for k in sheets},
                "scene_references": [str(p) for p in scene],
                "shots_json": str(shots) if shots else None}
            if refs:
                self.ledger.add("sheets_seed_o3", ["produced", "conditioning"],
                                "brief.characters_anchored", "configure",
                                f"{name}: {len(refs)} 张角色图作为 --reference-images"
                                f"（来源：{', '.join(sorted(set(self.sheet_sources.values())))}）")
            if scene:
                self.ledger.add("keyframes_seed_scene", ["produced"], "visual.scene_reference",
                                "configure", f"{name}: {len(scene)} 张逐片段关键帧作为 --scene-references")
            if not refs and not scene:
                return None
            shot_sig = shots.read_text(encoding="utf-8") if shots else ""
            tag = f"_seeded_{self._sig(*refs, *scene, shot_sig)}"
            self.tags[("visual", str(video))] = tag
            return self.module("visual", video, reference_images=refs or None,
                               scene_references=scene or None, shots_json=shots,
                               tag=tag)
        return self._remember(key, go)

    # --------------------------------------------------------- subtitles --
    def subtitle_files(self) -> list[Path]:
        return self.files(SUB_EXT)

    def subtitles_for(self, video: Path) -> tuple[Path, list[dict]] | None:
        """The subtitle file for this episode, parsed. Matched on episode
        number in the file name or its folder; when no file carries one, a set
        of exactly as many files as episodes is paired by order."""
        key = ("subs", str(video))

        def go():
            subs = self.subtitle_files()
            if not subs:
                return None
            ep = self._episode(video)
            cands = [p for p in subs if episode_of(p) == ep] if ep is not None else []
            if not cands:
                vids = sorted(self.delivered_videos(),
                              key=lambda v: (self._episode(v) or 0, str(v)))
                if len(subs) == len(vids) and Path(video) in vids:
                    cands = [sorted(subs)[vids.index(Path(video))]]
            if not cands:
                return None
            stem = Path(video).stem.lower()
            # Among the episode's files (codex ships ep1.srt *and* ep1.ass),
            # the one with the most cues wins; ties go to the copy outside a
            # scratch tree, then the closest name, then .srt before .ass.
            parsed = []
            for p in cands:
                try:
                    cues = parse_subtitles(p.read_text(encoding="utf-8", errors="replace"),
                                           p.suffix.lower())
                except OSError:
                    continue
                parsed.append((p, cues))
            if not parsed:
                return None
            parsed.sort(key=lambda pc: (-len(pc[1]), self._is_scratch(pc[0]),
                                        -len(os.path.commonprefix([pc[0].stem.lower(), stem])),
                                        SUB_EXT.index(pc[0].suffix.lower())
                                        if pc[0].suffix.lower() in SUB_EXT else 9,
                                        str(pc[0])))
            return parsed[0]
        return self._remember(key, go)

    def text_artifacts(self, max_bytes: int = 200_000) -> list[tuple[Path, str]]:
        key = ("texts",)

        def go():
            out = []
            for p in self.files(TEXT_EXT, max_n=400):
                try:
                    if p.stat().st_size > max_bytes:
                        continue
                    out.append((p, p.read_text(encoding="utf-8", errors="replace")))
                except OSError:
                    continue
            return out
        return self._remember(key, go)
