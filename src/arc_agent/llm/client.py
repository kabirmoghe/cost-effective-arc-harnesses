"""LLM client implementations."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from openai import OpenAI

from ..config import ModelConfig, get_api_key


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Any:
        """Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions
            **kwargs: Additional provider-specific parameters

        Returns:
            Provider-specific response object
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name being used."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client."""

    def __init__(self, config: ModelConfig):
        """Initialize the OpenAI client.

        Args:
            config: Model configuration
        """
        self.config = config
        api_key = config.api_key or get_api_key("openai")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=api_key, base_url=config.base_url)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Any:
        """Send a chat completion request to OpenAI."""
        request_kwargs = {
            "model": self.config.model_name,
            "messages": messages,
        }

        # Handle token limits - use max_completion_tokens for o1 models
        if self.config.max_completion_tokens is not None:
            request_kwargs["max_completion_tokens"] = self.config.max_completion_tokens
        elif self.config.max_tokens is not None:
            request_kwargs["max_tokens"] = kwargs.get("max_tokens", self.config.max_tokens)

        # o1 models don't support temperature parameter
        if not self.config.model_name.startswith("o1"):
            request_kwargs["temperature"] = kwargs.get("temperature", self.config.temperature)

        if tools:
            request_kwargs["tools"] = tools
            # o1 models don't support tool_choice
            if not self.config.model_name.startswith("o1"):
                request_kwargs["tool_choice"] = kwargs.get("tool_choice", "auto")

        return self.client.chat.completions.create(**request_kwargs)

    def get_model_name(self) -> str:
        return self.config.model_name


class DeepSeekClient(LLMClient):
    """DeepSeek API client (OpenAI-compatible)."""

    def __init__(self, config: ModelConfig):
        """Initialize the DeepSeek client.

        Args:
            config: Model configuration
        """
        self.config = config
        api_key = config.api_key or get_api_key("deepseek")
        if not api_key:
            raise ValueError("DeepSeek API key not found. Set DEEPSEEK_API_KEY environment variable.")

        self.client = OpenAI(
            api_key=api_key,
            base_url=config.base_url or "https://api.deepseek.com",
        )

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Any:
        """Send a chat completion request to DeepSeek."""
        request_kwargs = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }

        if tools:
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = kwargs.get("tool_choice", "auto")

        return self.client.chat.completions.create(**request_kwargs)

    def get_model_name(self) -> str:
        return self.config.model_name


def create_client(config: ModelConfig) -> LLMClient:
    """Factory function to create the appropriate LLM client.

    Args:
        config: Model configuration

    Returns:
        LLMClient instance
    """
    if config.provider == "openai":
        return OpenAIClient(config)
    elif config.provider == "deepseek":
        return DeepSeekClient(config)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")
