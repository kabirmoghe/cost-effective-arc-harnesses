"""Base tool definitions."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from ..grid.state import GridState


class ToolResult(BaseModel):
    """Result of a tool execution."""

    success: bool = Field(description="Whether the tool executed successfully")
    message: str = Field(description="Human-readable result message")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Additional result data")
    is_terminal: bool = Field(default=False, description="Whether this ends the episode (submit)")

    def __str__(self) -> str:
        return self.message


class BaseTool(ABC):
    """Abstract base class for all tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name used in function calling."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does."""
        pass

    @property
    @abstractmethod
    def parameters_schema(self) -> Dict[str, Any]:
        """JSON Schema for tool parameters."""
        pass

    @abstractmethod
    def execute(self, grid_state: GridState, **kwargs) -> ToolResult:
        """Execute the tool with given parameters.

        Args:
            grid_state: The current grid state (may be modified)
            **kwargs: Tool-specific parameters

        Returns:
            ToolResult with success status and message
        """
        pass

    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    def to_anthropic_schema(self) -> Dict[str, Any]:
        """Convert to Anthropic tool calling format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters_schema,
        }
