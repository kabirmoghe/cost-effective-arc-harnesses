"""Configuration and settings for ARC agent."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env file from current directory or parent directories
load_dotenv()
# Also try loading from src directory specifically
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Disable LangSmith tracing by default (can be overridden in .env)
if "LANGCHAIN_TRACING_V2" not in os.environ:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


class ModelConfig(BaseModel):
    """Configuration for a specific LLM model."""

    provider: str = Field(description="Provider name: 'deepseek' or 'openai'")
    model_name: str = Field(description="Model identifier")
    api_key: Optional[str] = Field(default=None, description="API key (can be set via env var)")
    base_url: Optional[str] = Field(default=None, description="Custom base URL for API")
    temperature: float = Field(default=0.0, description="Sampling temperature")

    # Only one of these should be set
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens in response")
    max_completion_tokens: Optional[int] = Field(default=None, description="Maximum completion tokens in response")


class AgentConfig(BaseModel):
    """Configuration for the ARC agent."""

    model: ModelConfig = Field(description="LLM model configuration")
    max_steps: int = Field(default=50, description="Maximum reasoning steps before timeout")
    max_retries: int = Field(default=3, description="Maximum retries on parse/API errors")


# Default configurations
DEEPSEEK_CONFIG = ModelConfig(
    provider="deepseek",
    model_name="deepseek-chat",
    base_url="https://api.deepseek.com",
    temperature=0.0,
    max_tokens=4096,
)

DEEPSEEK_REASONER_CONFIG = ModelConfig(
    provider="deepseek",
    model_name="deepseek-reasoner",
    base_url="https://api.deepseek.com",
    temperature=0.0,
    max_tokens=8192,
)

OPENAI_GPT4_CONFIG = ModelConfig(
    provider="openai",
    model_name="gpt-4-turbo",
    temperature=0.0,
    max_tokens=4096,
)

OPENAI_GPT4O_CONFIG = ModelConfig(
    provider="openai",
    model_name="gpt-4o",
    temperature=0.0,
    max_tokens=4096,
)

OPENAI_O1_CONFIG = ModelConfig(
    provider="openai",
    model_name="o1",
    temperature=1.0,  # o1 requires temperature=1
    max_completion_tokens=16384,
)


def get_api_key(provider: str) -> Optional[str]:
    """Get API key from environment variables."""
    env_vars = {
        "deepseek": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY",
    }
    env_var = env_vars.get(provider)
    if env_var:
        return os.environ.get(env_var)
    return None


def get_default_config(provider: str = "deepseek") -> AgentConfig:
    """Get default agent configuration for a provider."""
    if provider == "deepseek":
        model_config = DEEPSEEK_CONFIG.model_copy()
    elif provider == "deepseek-reasoner":
        model_config = DEEPSEEK_REASONER_CONFIG.model_copy()
    elif provider == "openai":
        model_config = OPENAI_GPT4O_CONFIG.model_copy()
    elif provider == "openai-o1":
        print("Using OpenAI O1 config")
        print(OPENAI_O1_CONFIG)

        model_config = OPENAI_O1_CONFIG.model_copy()
    else:
        raise ValueError(f"Unknown provider: {provider}")

    model_config.api_key = get_api_key(model_config.provider)

    return AgentConfig(model=model_config)
