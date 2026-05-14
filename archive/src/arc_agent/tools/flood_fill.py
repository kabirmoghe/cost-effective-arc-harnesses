"""Flood fill tool - fill connected region with a color."""

from typing import Any, Dict

from ..grid.state import GridState
from ..data.types import COLOR_NAMES
from .base import BaseTool, ToolResult


class FloodFillTool(BaseTool):
    """Flood fill a connected region starting from a cell."""

    @property
    def name(self) -> str:
        return "flood_fill"

    @property
    def description(self) -> str:
        return (
            "Fill all cells connected to the starting cell (row, col) that have the same color "
            "with a new color. Uses 4-way connectivity (up, down, left, right)."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "row": {
                    "type": "integer",
                    "description": "Starting row index (0-indexed)",
                },
                "col": {
                    "type": "integer",
                    "description": "Starting column index (0-indexed)",
                },
                "color": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 9,
                    "description": "Color to fill with (0-9)",
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
                message=f"Cell ({row}, {col}) is already {COLOR_NAMES.get(color, color)}. No fill needed.",
            )

        filled_count = grid_state.flood_fill(row, col, color)

        return ToolResult(
            success=True,
            message=f"Flood filled {filled_count} cells from ({row}, {col}) with {COLOR_NAMES.get(color, color)}.",
            data={"filled_count": filled_count, "start_row": row, "start_col": col, "color": color},
        )
