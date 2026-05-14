"""Change color tool - change all selected cells to a new color."""

from typing import Any, Dict

from ..grid.state import GridState
from ..data.types import COLOR_NAMES
from .base import BaseTool, ToolResult


class ChangeColorTool(BaseTool):
    """Change the color of all currently selected cells."""

    @property
    def name(self) -> str:
        return "change_color"

    @property
    def description(self) -> str:
        return (
            "Change all currently selected cells to a new color. "
            "You must first use the select tool to select cells before using this tool."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "color": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 9,
                    "description": "Color value (0-9): 0=black, 1=blue, 2=red, 3=green, 4=yellow, 5=gray, 6=magenta, 7=orange, 8=cyan, 9=maroon",
                },
            },
            "required": ["color"],
        }

    def execute(self, grid_state: GridState, **kwargs) -> ToolResult:
        color = kwargs.get("color")

        if color is None:
            return ToolResult(
                success=False,
                message="Missing required parameter: color.",
            )

        color = int(color)

        if not 0 <= color <= 9:
            return ToolResult(
                success=False,
                message=f"Invalid color {color}. Must be 0-9.",
            )

        if not grid_state.selected_cells:
            return ToolResult(
                success=False,
                message="No cells are selected. Use the select tool first.",
            )

        changed_count = grid_state.change_selected_color(color)

        color_name = COLOR_NAMES.get(color, str(color))

        return ToolResult(
            success=True,
            message=f"Changed {changed_count} selected cells to {color_name}.",
            data={"changed_count": changed_count, "color": color},
        )
