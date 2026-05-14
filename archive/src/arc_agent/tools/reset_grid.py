"""Reset grid tool - reset to original input state."""

from typing import Any, Dict

from ..grid.state import GridState
from .base import BaseTool, ToolResult


class ResetGridTool(BaseTool):
    """Reset the grid to its original input state."""

    @property
    def name(self) -> str:
        return "reset_grid"

    @property
    def description(self) -> str:
        return (
            "Reset the grid to its original input state, undoing all changes. "
            "Also clears any cell selection."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def execute(self, grid_state: GridState, **kwargs) -> ToolResult:
        grid_state.reset()

        return ToolResult(
            success=True,
            message="Grid reset to original input state. Selection cleared.",
        )
