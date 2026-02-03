"""Tool registry for registration and execution."""

from typing import Dict, List, Any, Optional

from ..grid.state import GridState
from .base import BaseTool, ToolResult


class ToolRegistry:
    """Registry for managing and executing tools."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def execute(self, tool_name: str, grid_state: GridState, **kwargs) -> ToolResult:
        """Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute
            grid_state: Current grid state
            **kwargs: Tool parameters

        Returns:
            ToolResult from execution
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(
                success=False,
                message=f"Unknown tool: {tool_name}. Available tools: {', '.join(self.list_tools())}",
            )

        try:
            return tool.execute(grid_state, **kwargs)
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Error executing {tool_name}: {str(e)}",
            )

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """Get all tools in OpenAI function calling format."""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def get_anthropic_tools(self) -> List[Dict[str, Any]]:
        """Get all tools in Anthropic tool calling format."""
        return [tool.to_anthropic_schema() for tool in self._tools.values()]

    def get_tools_description(self) -> str:
        """Get a human-readable description of all tools."""
        lines = ["Available Tools:", ""]
        for tool in self._tools.values():
            lines.append(f"- {tool.name}: {tool.description}")
        return "\n".join(lines)


# Singleton default registry
_default_registry: Optional[ToolRegistry] = None


def get_default_registry() -> ToolRegistry:
    """Get the default tool registry with all standard tools registered."""
    global _default_registry

    if _default_registry is None:
        from .select import SelectTool
        from .edit import EditTool
        from .flood_fill import FloodFillTool
        from .change_color import ChangeColorTool
        from .reset_grid import ResetGridTool
        from .resize_grid import ResizeGridTool
        from .submit import SubmitTool

        _default_registry = ToolRegistry()
        _default_registry.register(SelectTool())
        _default_registry.register(EditTool())
        _default_registry.register(FloodFillTool())
        _default_registry.register(ChangeColorTool())
        _default_registry.register(ResetGridTool())
        _default_registry.register(ResizeGridTool())
        _default_registry.register(SubmitTool())

    return _default_registry
