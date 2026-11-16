"""Everything the evaluator needs beyond "which data, where, how thorough".

The CLI used to carry nineteen flags. Fourteen of them were knobs nobody varies
between runs -- device, model ids, key locations, worker counts -- and the three
filters (`--harness`, `--case`, `--limit`) all did what a substring on `--run`
already does. They now live in ``config/default.yaml``, which has the further
advantage of being a file you can archive beside a report: a run becomes
reproducible from its config rather than from a command line nobody wrote down.

Precedence: environment variable > config file > built-in default.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import yaml

from stage1_inputs.client import DEFAULT_MODEL
from shared.paths import CONFIG_ROOT

ALL_MODULES = ("spec", "artifact", "visual", "physical", "av", "seam")
DEFAULT_CONFIG = CONFIG_ROOT / "default.yaml"

# config key -> environment override
_ENV = {
    "device": "VAB_EVAL_DEVICE",
    "jobs": "VAB_EVAL_JOBS",
    "asr_model": "VAB_EVAL_ASR_MODEL",
    "lm.provider": "VAB_EVAL_LM_PROVIDER",
    "lm.model": "VAB_EVAL_LM_MODEL",
    "lm.max_chars": "VAB_EVAL_LM_MAX_CHARS",
    "lm.max_tokens": "VAB_EVAL_LM_MAX_TOKENS",
}


@dataclass
class LMSettings:
    provider: str = "openrouter"
    model: str = DEFAULT_MODEL
    max_chars: int = 120_000
    # Output budget. The extraction model reasons, and reasoning tokens are
    # billed against this same ceiling: at 16384 a 125k-character prompt
    # spent the whole budget thinking and returned an empty string with
    # finish_reason=length, which reads exactly like 'found nothing'.
    max_tokens: int = 32_768
    # None selects the native environment variable for lm.provider.
    api_key_env: str | None = None
    api_key_file: str | None = None


@dataclass
class Settings:
    modules: tuple[str, ...] = ALL_MODULES
    device: str = "cpu"
    cut_points: str | None = None
    jobs: int = 1
    asr_model: str = "base"
    # Use the bundled reference reimplementation where an original detector
    # is absent. Never makes a run "full": see the Detectors section of README.md.
    reference_detectors: bool = True
    lm: LMSettings = field(default_factory=LMSettings)
    source: str = "<defaults>"

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        path = path or DEFAULT_CONFIG
        if not path.is_file():
            raise ValueError(f"Configuration file does not exist: {path}")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError("Configuration must be valid YAML") from exc
        raw = {} if raw is None else raw
        if not isinstance(raw, dict):
            raise ValueError("Configuration must be a mapping")
        unknown = set(raw) - {"modules", "device", "cut_points", "jobs", "asr_model", "lm",
                              "reference_detectors"}
        if unknown:
            raise ValueError("Unknown configuration field(s): " + ", ".join(map(str, unknown)))

        lm_raw = raw.get("lm")
        lm_raw = {} if lm_raw is None else lm_raw
        if not isinstance(lm_raw, dict) or set(lm_raw) - set(LMSettings.__dataclass_fields__):
            raise ValueError("lm must contain only documented LM settings")
        lm = LMSettings(
            provider=_pick("lm.provider", lm_raw.get("provider"), LMSettings.provider),
            model=_pick("lm.model", lm_raw.get("model"), LMSettings.model),
            max_chars=int(_pick("lm.max_chars", lm_raw.get("max_chars"),
                                LMSettings.max_chars)),
            max_tokens=int(_pick("lm.max_tokens", lm_raw.get("max_tokens"),
                                 LMSettings.max_tokens)),
            api_key_env=lm_raw.get("api_key_env") or LMSettings.api_key_env,
            api_key_file=lm_raw.get("api_key_file"),
        )

        modules = raw.get("modules", list(ALL_MODULES))
        if modules == ["none"]:  # compatibility with earlier public examples
            modules = []
        if not isinstance(modules, list) or any(m not in ALL_MODULES for m in modules):
            raise ValueError("modules must be a list of supported module names (or [] to disable)")
        jobs = int(_pick("jobs", raw.get("jobs"), cls.jobs))
        if jobs < 1 or lm.max_chars < 1 or lm.max_tokens < 1:
            raise ValueError("jobs, lm.max_chars and lm.max_tokens must be positive")
        reference = raw.get("reference_detectors", True)
        if not isinstance(reference, bool):
            raise ValueError("reference_detectors must be true or false")
        return cls(
            modules=tuple(m for m in modules if m in ALL_MODULES),
            device=str(_pick("device", raw.get("device"), cls.device)),
            cut_points=raw.get("cut_points"),
            jobs=jobs,
            asr_model=str(_pick("asr_model", raw.get("asr_model"), cls.asr_model)),
            reference_detectors=reference,
            lm=lm,
            source=str(path) if path.exists() else "<defaults>",
        )


def _pick(key: str, from_file, default):
    env = os.environ.get(_ENV.get(key, ""))
    if env not in (None, ""):
        return env
    return default if from_file in (None, "") else from_file
