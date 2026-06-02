"""Thin LLM client factory."""

import asyncio
import json
import os
from openai import OpenAI, AsyncOpenAI, APIError, APIConnectionError, APITimeoutError, RateLimitError

PROVIDERS = {
    # V4-flash is the cheap non-reasoning chat model (~$0.14/Mout) and the
    # closest successor to V3.2 for the "harness lift on a non-reasoner" thesis
    # condition. Pinned to the explicit version slug (the `deepseek-chat` alias
    # is being sunset 2026-07-24). `thinking: disabled` is the current default
    # for this slug but we set it explicitly so a future default shift can't
    # silently turn on reasoning mid-thesis.
    "deepseek": {
        "env_key": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-v4-flash",
        "thinking_disabled": True,
    },
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "base_url": None,
        "default_model": "gpt-4o",
    },
    # DeepSeek's first-party `deepseek-chat` alias moved to V4 (2026-05); V3.2
    # is reachable through OpenRouter. Model is pinned to an explicit version
    # slug — never an alias — so the served model can't shift underneath a run.
    "openrouter": {
        "env_key": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "deepseek/deepseek-v3.2",
        # Pin the inference backend: OpenRouter otherwise routes a slug across
        # multiple hosts (differing quant/engine), which would vary the served
        # stack request-to-request. AtlasCloud/fp8 picked 2026-05-18 after a
        # cross-provider stability probe on V3.2 (probe v7): 16/16 clean parses
        # at ~$0.38/Mout vs Friendli native ~$1.50/Mout — same model, ~4×
        # cheaper, equivalent observed stability. Originally went with Novita
        # but it returned NOT_ENOUGH_BALANCE 403s mid-run despite a funded OR
        # account — appears to be Novita-side billing gate, not OR. AtlasCloud
        # works on the same balance. Friendli kept as fallback for accuracy
        # parity reruns if quantization ever shows headline drift.
        # External corroboration (LocalLLaMA discussion, 2025-08): community
        # reports specifically flag Novita's low-cost deployments as suspected
        # of undisclosed/aggressive quantization and weaker configuration
        # quality — reinforces the decision to pin away from novita/fp8 here.
        "or_provider": "atlas-cloud/fp8",
    },
    # Friendli native bf16 — kept as the headline-defensible non-quantized backend
    # for the final B7 thesis number. ~4× cost of AtlasCloud FP8 on output side,
    # ~1.5× end-to-end (per the prompt-heavy 87/13 workload). Use for the
    # canonical pass@2 reporting, not for exploration sweeps.
    "openrouter-friendli": {
        "env_key": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "deepseek/deepseek-v3.2",
        "or_provider": "friendli",
    },
    # Cross-model panel for C9 — added 2026-06-01. Both providers pinned to
    # backends that support tool_choice="required" (probed via
    # scripts/cross_model_provider_probe.py). Atlas-cloud serves Qwen3 as well
    # as V3.2, so we keep the same provider relationship; gpt-oss is served by
    # deepinfra (one of two `required`-supporting backends along with novita).
    # Pinned to alibaba — Qwen's native deployment on OpenRouter. Trade-off
    # vs atlas-cloud/fp8: loses exact V3.2 quantization parity (alibaba serves
    # BF16 native instead of FP8), gains independent rate-limit pool (atlas
    # and novita both rate-limited at concurrency=5 on Qwen3; alibaba is the
    # model creator's own infra). Methodological cleanliness preserved:
    # single backend per model. Writeup framing: "Qwen3 served at native BF16
    # via Alibaba's OpenRouter endpoint — the model creator's reference
    # deployment, precision class differs from V3.2's FP8 but matches the
    # published model card." Tool-calling verified at 100% adherence during
    # earlier probe.
    "openrouter-qwen3": {
        "env_key": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "qwen/qwen3-235b-a22b-2507",
        "or_provider": "alibaba",
    },
    # Llama-3.3-70B-Instruct on nebius/fp8 — matches V3.2's FP8 quantization
    # exactly. Verified 2026-06-01: 100% tool-call adherence at `tool_choice="auto"`
    # on 8 ARC eval tasks; $20.26/400 V3.2-mix projection. Dense 70B (vs
    # V3.2's 37B-active MoE) — different architecture family for the cross-model
    # transfer claim. Meta/US lab — maximum independence from DeepSeek + Alibaba.
    # Replaced gpt-oss-120b as the panel's third model after gpt-oss was found to
    # be a CoT-RL'd reasoning model (structural confound with the "non-reasoning"
    # framing). `tool_choice` left as default "auto" for parity with V3.2.
    "openrouter-llama-3.3": {
        "env_key": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "meta-llama/llama-3.3-70b-instruct",
        "or_provider": "nebius/fp8",
    },
    # Llama-4-Maverick on google-vertex/us-east5 — only OpenRouter backend
    # that supports tool calling for Maverick (5 of 5 other backends return
    # 404 "no endpoints found that support tool use"). MMLU-Pro 80.5, MoE
    # 400B/17B-active. Meta/US lab. Replaces Llama-3.3-70B as the third panel
    # member after Llama-3.3 showed insufficient capability for the harness
    # (0/15 definers got non-zero train on a 5-task probe). 88% tool-call
    # adherence on 8 ARC tasks. Cost projection: $55.62/400 V3.2-mix. No FP8
    # quantization parity (Vertex serves native precision) — mild
    # methodological gap, documented in writeup.
    "openrouter-llama-4-maverick": {
        "env_key": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "meta-llama/llama-4-maverick",
        "or_provider": "google-vertex/us-east5",
    },
    # Kimi K2 0905 on atlas-cloud/fp8 — matches V3.2's exact provider+
    # quantization profile. 1T total / 32B active MoE, Moonshot/China.
    # 100% tool-call adherence on 8 ARC tasks (verified 2026-06-01).
    # Replaces Llama-3.3 / Llama-4-Maverick as the third panel slot after
    # both Meta options showed insufficient capability for our harness
    # (0/5 test-correct on the 5-task probe). The agentic-RL confound is
    # accepted and documented per panel framing: "frontier-adjacent open
    # MoE instruct models, all served as direct-answer/non-thinking" —
    # NOT a "no RL ever" claim. Cost: $103.19/400 V3.2-mix projection,
    # comparable to V3.2's $100.27.
    "openrouter-kimi-k2": {
        "env_key": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "moonshotai/kimi-k2-0905",
        "or_provider": "atlas-cloud/fp8",
        # Kimi K2 + response_format=json_object + complex reasoning prompts
        # produces malformed output (e.g. emits a scalar "1.1e111" instead of
        # a structured answer). Disabling json_object mode for baseline/CoT
        # lets Kimi reason normally and emit a parseable JSON code block.
        # Verified 2026-06-01.
        "json_response_format": False,
    },
}


def _build_kwargs(provider: str) -> dict:
    config = PROVIDERS[provider]
    kwargs = {"api_key": os.environ[config["env_key"]]}
    if config["base_url"]:
        kwargs["base_url"] = config["base_url"]
    return kwargs


def create_client(provider: str = "deepseek") -> OpenAI:
    """Return a pre-configured OpenAI-compatible client."""
    return OpenAI(**_build_kwargs(provider))


def create_async_client(provider: str = "deepseek") -> AsyncOpenAI:
    """Return a pre-configured async OpenAI-compatible client."""
    return AsyncOpenAI(**_build_kwargs(provider))


def get_default_model(provider: str = "deepseek") -> str:
    """Return the default model name for a provider."""
    return PROVIDERS[provider]["default_model"]


def _is_transient_provider_error(exc: BaseException) -> bool:
    """Classify whether an exception is provider-side infrastructure flakiness
    (safe to retry) vs. a real model/code error (must surface).

    Retry policy is documented in PLAN.md (Track B4 incident log): we retry on
    rate-limits, transport errors, malformed HTTP bodies (JSONDecodeError from
    inside the SDK), and degenerate completions like `response.choices = null`
    (which surface as `'NoneType' object is not subscriptable` at the call
    site that does `response.choices[0]`). We never retry on a result the
    model actually produced.
    """
    if isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIError):
        status = getattr(exc, "status_code", None)
        if status is None or status >= 500 or status == 429:
            return True
        return False
    if isinstance(exc, json.JSONDecodeError):
        return True
    if isinstance(exc, TypeError) and "NoneType" in str(exc) and "subscriptable" in str(exc):
        return True
    return False


async def with_retry(call, *, attempts: int = 3, base_delay: float = 1.0, label: str = "api"):
    """Retry a zero-arg async callable on transient provider errors.

    Exponential backoff (base_delay, 2x, 4x, ...). Non-transient errors
    propagate immediately. The final attempt's failure is re-raised.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await call()
        except BaseException as exc:
            if not _is_transient_provider_error(exc) or attempt == attempts:
                raise
            last_exc = exc
            delay = base_delay * (2 ** (attempt - 1))
            print(f"[{label}] transient error (attempt {attempt}/{attempts}): {type(exc).__name__}: {str(exc)[:120]} — retrying in {delay:.1f}s", flush=True)
            await asyncio.sleep(delay)
    assert last_exc is not None  # unreachable
    raise last_exc


def with_retry_sync(call, *, attempts: int = 5, base_delay: float = 1.5, label: str = "api"):
    """Sync sibling of with_retry — used by baseline/CoT (ThreadPoolExecutor
    workers, no asyncio loop). Slightly more aggressive defaults (5 attempts,
    1.5s base) since single-call workloads benefit from longer retry windows
    when many workers hit a provider's burst limit simultaneously.
    """
    import time
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except BaseException as exc:
            if not _is_transient_provider_error(exc) or attempt == attempts:
                raise
            last_exc = exc
            delay = base_delay * (2 ** (attempt - 1))
            print(f"[{label}] transient error (attempt {attempt}/{attempts}): {type(exc).__name__}: {str(exc)[:120]} — retrying in {delay:.1f}s", flush=True)
            time.sleep(delay)
    assert last_exc is not None  # unreachable
    raise last_exc


def get_response_format(provider: str = "deepseek") -> dict | None:
    """Return the `response_format` to use, or None to omit.

    Defaults to `{"type": "json_object"}` for OpenRouter providers (baseline/
    CoT use JSON-mode to constrain output to a parseable shape). Some models
    (notably Kimi K2) misbehave under json_object on complex reasoning prompts
    — for those, return None and let the parser handle a mixed
    reasoning-text + JSON code block.
    """
    cfg = PROVIDERS[provider]
    if cfg.get("json_response_format") is False:
        return None
    # Default: JSON-object mode for any OpenRouter-style provider
    if cfg.get("base_url", "").startswith("https://openrouter.ai"):
        return {"type": "json_object"}
    return {"type": "json_object"}


def get_tool_choice(provider: str = "deepseek") -> str:
    """Return the right `tool_choice` for a provider.

    V3.2 via OpenRouter/AtlasCloud FP8 rejects `tool_choice="required"` (returns
    404), so the V3.2 entries default to "auto" — the system prompt + tool
    design steer the model reliably and the no-tool-call fuse handles edge
    cases. Cross-model entries (qwen3, gpt-oss) use the providers that DO
    support `required`, since that's a closer match to the original architectural
    intent (a forced tool call removes the "model emits plain text and refuses"
    failure mode entirely).
    """
    return PROVIDERS[provider].get("tool_choice", "auto")


def get_extra_body(provider: str = "deepseek") -> dict | None:
    """Return the `extra_body` for chat.completions.create, or None.

    For OpenRouter, this pins the inference backend so the served model stack
    can't vary between requests. Two pinning modes:
      - `or_provider`: single backend, no fallbacks (strict quantization parity).
      - `or_provider_order`: ordered list of acceptable backends. OpenRouter
        tries them in order on 429/503 etc., staying within the listed set
        (allow_fallbacks=False keeps us out of un-pinned providers). Used when
        a single backend rate-limits too aggressively for our concurrency.

    Direct providers return None.
    """
    config = PROVIDERS[provider]
    or_order = config.get("or_provider_order")
    if or_order:
        return {"provider": {"order": list(or_order), "allow_fallbacks": False}}
    or_provider = config.get("or_provider")
    if or_provider:
        return {"provider": {"only": [or_provider], "allow_fallbacks": False}}
    if config.get("thinking_disabled"):
        return {"thinking": {"type": "disabled"}}
    return None
