"""Submit tool - submit current grid as solution."""

from typing import Any, Dict, Optional

from ..grid.state import GridState
from ..data.types import Grid
from ..grid.operations import grids_equal
from .base import BaseTool, ToolResult


class SubmitTool(BaseTool):
    """Submit the current grid as the solution."""

    def __init__(self, target_grid: Optional[Grid] = None):
        """Initialize with optional target grid for validation.

        Args:
            target_grid: Expected output grid to validate against
        """
        self._target_grid = target_grid

    def set_target(self, target_grid: Optional[Grid]) -> None:
        """Set the target grid for validation."""
        self._target_grid = target_grid

    @property
    def name(self) -> str:
        return "submit"

    @property
    def description(self) -> str:
        return (
            "Submit the current grid as your solution. This ends the episode. "
            "Make sure you have completed all transformations before submitting."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def execute(self, grid_state: GridState, **kwargs) -> ToolResult:
        # Check against target if available
        is_correct = None
        if self._target_grid is not None:
            is_correct = grids_equal(grid_state.grid, self._target_grid)

        if is_correct is True:
            return ToolResult(
                success=True,
                message="Solution submitted. Correct!",
                data={"correct": True, "grid": grid_state.grid},
                is_terminal=True,
            )
        elif is_correct is False:
            return ToolResult(
                success=True,  # Submission was successful, just wrong
                message="Solution submitted. Incorrect - grid does not match expected output.",
                data={"correct": False, "grid": grid_state.grid},
                is_terminal=True,
            )
        else:
            return ToolResult(
                success=True,
                message="Solution submitted.",
                data={"grid": grid_state.grid},
                is_terminal=True,
            )
