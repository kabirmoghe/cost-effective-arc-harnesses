"""Thin LLM client factory."""

import os
from openai import OpenAI, AsyncOpenAI

PROVIDERS = {
    "deepseek": {
        "env_key": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat",
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
        # stack request-to-request. Friendli serves V3.2 at native precision
        # (no quant), the closest available match to the first-party baseline.
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


def get_extra_body(provider: str = "deepseek") -> dict | None:
    """Return the `extra_body` for chat.completions.create, or None.

    For OpenRouter, this pins the inference backend so the served model stack
    can't vary between requests. Direct providers return None.
    """
    or_provider = PROVIDERS[provider].get("or_provider")
    if or_provider:
        return {"provider": {"only": [or_provider], "allow_fallbacks": False}}
    return None
