"""Select tool - select cells for subsequent operations."""

from typing import Any, Dict, List

from ..grid.state import GridState
from .base import BaseTool, ToolResult


class SelectTool(BaseTool):
    """Select cells by coordinates for subsequent operations."""

    @property
    def name(self) -> str:
        return "select"

    @property
    def description(self) -> str:
        return (
            "Select cells by their (row, col) coordinates. Selected cells can then be "
            "modified using change_color. Replaces any existing selection."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "cells": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "description": "List of [row, col] coordinates to select",
                },
            },
            "required": ["cells"],
        }

    def execute(self, grid_state: GridState, **kwargs) -> ToolResult:
        cells_raw = kwargs.get("cells", [])

        if not cells_raw:
            return ToolResult(
                success=False,
                message="No cells provided to select.",
            )

        # Convert to list of tuples
        cells: List[tuple] = []
        for cell in cells_raw:
            if isinstance(cell, (list, tuple)) and len(cell) >= 2:
                cells.append((int(cell[0]), int(cell[1])))
            else:
                return ToolResult(
                    success=False,
                    message=f"Invalid cell format: {cell}. Expected [row, col].",
                )

        valid_count = grid_state.select(cells)

        if valid_count == 0:
            return ToolResult(
                success=False,
                message=f"No valid cells selected. All {len(cells)} coordinates were out of bounds.",
            )

        invalid_count = len(cells) - valid_count
        if invalid_count > 0:
            return ToolResult(
                success=True,
                message=f"Selected {valid_count} cells ({invalid_count} were out of bounds).",
                data={"selected_count": valid_count, "invalid_count": invalid_count},
            )

        return ToolResult(
            success=True,
            message=f"Selected {valid_count} cells.",
            data={"selected_count": valid_count},
        )
