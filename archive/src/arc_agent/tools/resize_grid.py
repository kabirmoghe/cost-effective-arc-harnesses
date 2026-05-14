"""Resize grid tool - change grid dimensions."""

from typing import Any, Dict

from ..grid.state import GridState
from ..grid.operations import create_empty_grid
from .base import BaseTool, ToolResult


class ResizeGridTool(BaseTool):
    """Resize the grid to new dimensions."""

    @property
    def name(self) -> str:
        return "resize_grid"

    @property
    def description(self) -> str:
        return (
            "Resize the grid to new dimensions. Creates a new grid filled with zeros (black). "
            "Use this when the output needs to be a different size than the input. "
            "Any existing content will be lost."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "rows": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 30,
                    "description": "Number of rows in the new grid",
                },
                "cols": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 30,
                    "description": "Number of columns in the new grid",
                },
            },
            "required": ["rows", "cols"],
        }

    def execute(self, grid_state: GridState, **kwargs) -> ToolResult:
        rows = kwargs.get("rows")
        cols = kwargs.get("cols")

        if rows is None or cols is None:
            return ToolResult(
                success=False,
                message="Missing required parameters: rows and cols.",
            )

        rows = int(rows)
        cols = int(cols)

        if not (1 <= rows <= 30) or not (1 <= cols <= 30):
            return ToolResult(
                success=False,
                message=f"Invalid dimensions {rows}x{cols}. Must be between 1 and 30.",
            )

        old_rows, old_cols = grid_state.rows, grid_state.cols

        # Create new grid
        grid_state.grid = create_empty_grid(rows, cols, fill_value=0)
        grid_state.selected_cells.clear()

        return ToolResult(
            success=True,
            message=f"Resized grid from {old_rows}x{old_cols} to {rows}x{cols}. Grid is now filled with zeros.",
            data={"old_rows": old_rows, "old_cols": old_cols, "new_rows": rows, "new_cols": cols},
        )
