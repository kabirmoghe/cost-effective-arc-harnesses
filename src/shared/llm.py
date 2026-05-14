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
    }
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
