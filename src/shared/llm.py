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


def get_extra_body(provider: str = "deepseek") -> dict | None:
    """Return the `extra_body` for chat.completions.create, or None.

    For OpenRouter, this pins the inference backend so the served model stack
    can't vary between requests. Direct providers return None.
    """
    config = PROVIDERS[provider]
    or_provider = config.get("or_provider")
    if or_provider:
        return {"provider": {"only": [or_provider], "allow_fallbacks": False}}
    if config.get("thinking_disabled"):
        return {"thinking": {"type": "disabled"}}
    return None
