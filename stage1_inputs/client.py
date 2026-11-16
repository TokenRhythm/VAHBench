"""Minimal chat client for the trajectory extractor. Stdlib only.

Scope is deliberately tiny: one blocking call that takes a prompt and returns
text, configured for greedy decoding, with the key read from the environment and
never written anywhere. It exists so `extractors.lm_extract` can be handed a
`call_fn` without that module owning credentials or a socket.

**What leaves the machine.** Calling this sends trajectory file contents to the
configured provider. Trajectories are execution logs: they can contain absolute
paths, model ids, costs, and whatever prose the agent wrote about its own run.
The pipeline redacts recognised credentials before this client receives a
prompt. `lm.max_chars` controls chunk size, not the total sent; all chunks are
read. `facts.source_files` records which files supplied the extraction input.

**Reproducibility is weaker here than for local weights.** A checkpoint on disk
is fixed; an API-served model is whatever the provider is running today, and
`temperature=0` does not change that. So every response records the model string
the API reports back, alongside the one requested -- when they diverge, the
report says so instead of quietly claiming determinism it does not have.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from dataclasses import field

# Provider -> (endpoint, env var). OpenRouter first: it fronts DeepSeek and most
# others behind one key, which is the common case here.
PROVIDERS = {
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions",
                   "OPENROUTER_API_KEY"),
    "deepseek": ("https://api.deepseek.com/chat/completions",
                 "DEEPSEEK_API_KEY"),
    "openai": ("https://api.openai.com/v1/chat/completions",
               "OPENAI_API_KEY"),
}

# Explicit model version; avoid a moving "latest" alias. Recorded in each audit.
DEFAULT_MODEL = "z-ai/glm-5.3"

_TRANSPORT_RETRIES = 2          # transport only -- never on unparseable content


def resolve_provider(preferred: str | None = None) -> tuple[str, str, str] | None:
    """Resolve a provider; an explicit selection never falls back to another.

    The key is returned for immediate use and never logged, echoed, stored in a
    report, or included in an error message.
    """
    order = [preferred] if preferred else list(PROVIDERS)
    for name in order:
        if name not in PROVIDERS:
            continue
        url, env = PROVIDERS[name]
        key = os.environ.get(env, "").strip()
        if key:
            return name, url, key
    return None


def available() -> list[str]:
    return [n for n, (_, env) in PROVIDERS.items() if os.environ.get(env, "").strip()]


@dataclass
class CallLog:
    """What actually happened, for the report. No prompt text, no key."""

    model_requested: str = ""
    model_served: str | None = None
    provider: str = ""
    n_calls: int = 0
    total_prompt_chars: int = 0
    errors: list[str] = field(default_factory=list)
    reasoning_effort: str | None = None

    @property
    def model_drifted(self) -> bool:
        return bool(self.model_served and self.model_served != self.model_requested)


def make_call_fn(model: str = DEFAULT_MODEL, *, provider: str | None = None,
                 timeout: float = 300.0, log: CallLog | None = None,
                 max_tokens: int = 32768):
    # max_tokens is generous because the default model is a reasoning model:
    # `reasoning` tokens are billed against the same budget as `content`, so a
    # 4096 ceiling on a 30k-character extraction left nothing for the answer and
    # returned an empty string with finish_reason=length. That looked exactly
    # like "the model found nothing", which is the wrong lesson to draw.
    """Return `call_fn(prompt) -> str`, or None when no key is configured.

    Decoding is greedy: temperature 0, top_p 1, fixed seed where the provider
    honours it. That is the strongest determinism an API allows; see the module
    docstring for why it is still weaker than a local checkpoint.
    """
    resolved = resolve_provider(provider)
    if resolved is None:
        return None
    name, url, key = resolved
    # GLM-5.3 defaults to max reasoning on OpenRouter. In the real checklist
    # audit it consumed 32766/32768 tokens before producing an answer.
    reasoning_effort = "low" if name == "openrouter" and model == "z-ai/glm-5.3" else None
    if log is not None:
        log.provider, log.model_requested = name, model
        log.reasoning_effort = reasoning_effort

    def call_fn(prompt: str) -> str:
        request_body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "top_p": 1,
            "seed": 0,
            "max_tokens": max_tokens,
        }
        if reasoning_effort is not None:
            request_body["reasoning"] = {"effort": reasoning_effort}
            request_body["response_format"] = {"type": "json_object"}
        body = json.dumps(request_body).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST", headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost/video-agent-benchmark",
            "X-Title": "video-agent-benchmark trajectory extractor",
        })
        last = ""
        for attempt in range(_TRANSPORT_RETRIES + 1):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    payload = json.loads(r.read().decode("utf-8"))
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                # Transport only. A well-formed reply whose *content* does not
                # parse is the extractor's problem and is never retried there.
                # Exception text can echo credentials or request contents.
                # Keep only the class and HTTP status in persisted reports.
                last = type(e).__name__
                if isinstance(e, urllib.error.HTTPError):
                    last += f": HTTP {e.code}"
                if attempt == _TRANSPORT_RETRIES:
                    if log is not None:
                        log.errors.append(last)
                    return ""
                time.sleep(1.5 * (attempt + 1))
        else:                                             # pragma: no cover
            return ""

        if log is not None:
            log.n_calls += 1
            log.total_prompt_chars += len(prompt)
            served = payload.get("model")
            if served:
                log.model_served = served
        try:
            choice = payload["choices"][0]
            content = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            if log is not None:
                log.errors.append("响应缺少 choices[0].message")
            return ""

        fr = choice.get("finish_reason") or choice.get("native_finish_reason")
        if log is not None and (not content or fr == "length"):
            # Say why it was empty or cut short. A silent "" is
            # indistinguishable from the model honestly finding nothing, and
            # that ambiguity cost a full debugging cycle here; a reply cut off
            # by the output budget is non-empty and invalid JSON, which the
            # extractor reports as "could not parse" unless told otherwise.
            usage = payload.get("usage") or {}
            rt = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
            hint = f"finish_reason={fr}"
            if fr == "length":
                hint += f"；输出预算耗尽（reasoning_tokens={rt}，max_tokens={max_tokens}），调大 lm.max_tokens（config/default.yaml 或 VAB_EVAL_LM_MAX_TOKENS）"
            elif choice["message"].get("refusal"):
                hint += "；模型拒绝作答"
            log.errors.append(("content 为空：" if not content else "content 被截断：") + hint)
        return content

    return call_fn
