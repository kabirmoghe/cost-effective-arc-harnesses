"""Edit tool - set a single cell to a specific color."""

from typing import Any, Dict

from ..grid.state import GridState
from ..data.types import COLOR_NAMES
from .base import BaseTool, ToolResult


class EditTool(BaseTool):
    """Edit a single cell to a specific color."""

    @property
    def name(self) -> str:
        return "edit"

    @property
    def description(self) -> str:
        return (
            "Set a single cell to a specific color. Colors are integers 0-9: "
            "0=black, 1=blue, 2=red, 3=green, 4=yellow, 5=gray, 6=magenta, 7=orange, 8=cyan, 9=maroon."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "row": {
                    "type": "integer",
                    "description": "Row index (0-indexed)",
                },
                "col": {
                    "type": "integer",
                    "description": "Column index (0-indexed)",
                },
                "color": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 9,
                    "description": "Color value (0-9)",
                },
            },
            "required": ["row", "col", "color"],
        }

    def execute(self, grid_state: GridState, **kwargs) -> ToolResult:
        row = kwargs.get("row")
        col = kwargs.get("col")
        color = kwargs.get("color")

        if row is None or col is None or color is None:
            return ToolResult(
                success=False,
                message="Missing required parameters: row, col, and color are all required.",
            )

        row = int(row)
        col = int(col)
        color = int(color)

        if not 0 <= color <= 9:
            return ToolResult(
                success=False,
                message=f"Invalid color {color}. Must be 0-9.",
            )

        old_value = grid_state.get_cell(row, col)
        if old_value is None:
            return ToolResult(
                success=False,
                message=f"Cell ({row}, {col}) is out of bounds. Grid is {grid_state.rows}x{grid_state.cols}.",
            )

        if old_value == color:
            return ToolResult(
                success=True,
                message=f"Cell ({row}, {col}) is already {COLOR_NAMES.get(color, color)}.",
            )

        success = grid_state.set_cell(row, col, color)

        if success:
            return ToolResult(
                success=True,
                message=f"Set cell ({row}, {col}) from {COLOR_NAMES.get(old_value, old_value)} to {COLOR_NAMES.get(color, color)}.",
                data={"row": row, "col": col, "old_color": old_value, "new_color": color},
            )
        else:
            return ToolResult(
                success=False,
                message=f"Failed to set cell ({row}, {col}).",
            )
