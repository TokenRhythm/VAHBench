"""Capability check. Run before any work, print what will and will not happen.

A benchmark that silently skips half its checks and prints a score is worse
than one that refuses to start. Every capability missing here turns into
abstentions downstream, which lowers coverage rather than the score -- but the
operator should know *before* a two-hour sweep, not after.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import stage1_inputs.client as llm
import stage4_checks.dispatch as runners


@dataclass
class Capability:
    name: str
    ok: bool
    detail: str = ""
    impact: str = ""


@dataclass
class Preflight:
    caps: list[Capability] = field(default_factory=list)
    runnable_modules: list[str] = field(default_factory=list)
    # module -> "original" | "reference" | "missing"
    module_origins: dict[str, str] = field(default_factory=dict)
    lm_ready: bool = False
    lm_provider: str = ""

    @property
    def fatal(self) -> list[Capability]:
        return [c for c in self.caps if not c.ok and c.impact.startswith("致命")]

    def render(self) -> str:
        lines = []
        for c in self.caps:
            mark = "✓" if c.ok else "✗"
            lines.append(f"  {mark} {c.name:<22}{c.detail}")
            if not c.ok and c.impact:
                lines.append(f"      → {c.impact}")
        return "\n".join(lines)


def check(data_dir: Path, *, want_modules: tuple[str, ...],
          want_lm: bool, provider: str | None = None,
          model: str = llm.DEFAULT_MODEL) -> Preflight:
    pf = Preflight()

    d = Path(data_dir)
    pf.caps.append(Capability(
        "轨迹目录", d.is_dir(), str(d),
        "致命：目录不存在，无法继续" if not d.is_dir() else ""))

    ff = shutil.which("ffprobe")
    pf.caps.append(Capability(
        "ffprobe", bool(ff), ff or "不在 PATH 上",
        "致命：无法探测视频时长，成片定位完全依赖它" if not ff else ""))

    for m in want_modules:
        script, origin = runners.resolve_script(m)
        pf.module_origins[m] = origin
        if script is None:
            pf.caps.append(Capability(f"模块 {m}", False,
                                      f"找不到 {runners.SCRIPTS[m]}，也没有参考实现",
                                      "该模块的测量不可用；部分模式仅执行已有检查"))
            continue
        exe, how = runners.resolve_interpreter(m)
        ok = exe is not None
        label = "原始检测器" if origin == "original" else "参考重实现，非论文检测器"
        need = ("numpy" if origin == "reference" else
                runners.PROBES.get(m, "").replace("import ", ""))
        detail = (f"{label} · {exe}（{how}）" if ok else
                  f"{label}：没有能导入 {need} 的解释器；可设 {runners.ENV_OVERRIDE[m]}，或建 "
                  f"{runners.REPO / '.venvs' / runners._FAMILY[m]}")
        if ok and origin == "reference" and not shutil.which("ffmpeg"):
            ok, detail = False, f"{label}：ffmpeg 不在 PATH 上，无法解码画面和音频"
        if ok:
            pf.runnable_modules.append(m)
        pf.caps.append(Capability(
            f"模块 {m}", ok, detail,
            f"{m} 的全部检查将弃权，覆盖率下降" if not ok else ""))

    if want_lm:
        # Not marked fatal here: whether the model is *needed* is a property
        # of the data, and main() measures it by running the deterministic
        # pass over every run before any work. With gaps and no key the sweep
        # refuses to start (run_eval.lm_gate); with no gaps it proceeds.
        resolved = llm.resolve_provider(provider)
        pf.lm_ready = resolved is not None
        pf.lm_provider = resolved[0] if resolved else ""
        key_envs = ([llm.PROVIDERS[provider][1]] if provider in llm.PROVIDERS
                    else [env for _, env in llm.PROVIDERS.values()])
        pf.caps.append(Capability(
            "LM 抽取", pf.lm_ready,
            f"{pf.lm_provider} · {model}" if pf.lm_ready else
            "未找到 API key（试过 " + "、".join(key_envs) + "）",
            "LM 是清单抽取的一部分：若任一 run 在规则抽取后仍有缺口，将终止"
            if not pf.lm_ready else ""))
    else:
        pf.caps.append(Capability("LM 抽取", True, "已跳过（--discover-only 不评分）"))

    return pf
