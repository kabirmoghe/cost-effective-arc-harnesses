"""GridState - Mutable grid with selection tracking."""

from typing import Set, Tuple, Optional, List
from pydantic import BaseModel, Field

from ..data.types import Grid
from .operations import copy_grid, grids_equal, is_valid_cell, flood_fill_region
from .serializer import grid_to_ascii


class GridState(BaseModel):
    """Mutable state representing the current working grid with selection."""

    grid: Grid = Field(description="Current 2D grid state")
    original_grid: Grid = Field(description="Original input grid for reset")
    selected_cells: Set[Tuple[int, int]] = Field(default_factory=set, description="Currently selected cells")

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_grid(cls, grid: Grid) -> "GridState":
        """Create a new GridState from an input grid."""
        return cls(
            grid=copy_grid(grid),
            original_grid=copy_grid(grid),
            selected_cells=set(),
        )

    @property
    def rows(self) -> int:
        """Number of rows in the grid."""
        return len(self.grid)

    @property
    def cols(self) -> int:
        """Number of columns in the grid."""
        return len(self.grid[0]) if self.grid else 0

    def get_cell(self, row: int, col: int) -> Optional[int]:
        """Get value at cell, or None if out of bounds."""
        if is_valid_cell(self.grid, row, col):
            return self.grid[row][col]
        return None

    def set_cell(self, row: int, col: int, color: int) -> bool:
        """Set a cell to a color. Returns True if successful."""
        if not is_valid_cell(self.grid, row, col):
            return False
        if not 0 <= color <= 9:
            return False
        self.grid[row][col] = color
        return True

    def select(self, cells: List[Tuple[int, int]]) -> int:
        """Select cells by coordinates. Returns number of valid cells selected."""
        self.selected_cells.clear()
        valid_count = 0
        for row, col in cells:
            if is_valid_cell(self.grid, row, col):
                self.selected_cells.add((row, col))
                valid_count += 1
        return valid_count

    def clear_selection(self) -> None:
        """Clear all selected cells."""
        self.selected_cells.clear()

    def change_selected_color(self, color: int) -> int:
        """Change all selected cells to a new color. Returns count of changed cells."""
        if not 0 <= color <= 9:
            return 0
        changed = 0
        for row, col in self.selected_cells:
            if is_valid_cell(self.grid, row, col):
                self.grid[row][col] = color
                changed += 1
        return changed

    def flood_fill(self, row: int, col: int, color: int) -> int:
        """Flood fill from a cell. Returns number of cells filled."""
        if not is_valid_cell(self.grid, row, col):
            return 0
        if not 0 <= color <= 9:
            return 0

        old_color = self.grid[row][col]
        if old_color == color:
            return 0

        new_grid = flood_fill_region(self.grid, row, col, color)

        # Count changes
        changed = sum(
            1 for r in range(self.rows) for c in range(self.cols)
            if self.grid[r][c] != new_grid[r][c]
        )

        self.grid = new_grid
        return changed

    def reset(self) -> None:
        """Reset grid to original state."""
        self.grid = copy_grid(self.original_grid)
        self.selected_cells.clear()

    def matches(self, target: Grid) -> bool:
        """Check if current grid matches the target."""
        return grids_equal(self.grid, target)

    def to_ascii(self, show_selection: bool = True, show_coordinates: bool = True) -> str:
        """Convert current state to ASCII for LLM context."""
        selected = self.selected_cells if show_selection else None
        return grid_to_ascii(self.grid, selected, show_coordinates)

    def copy(self) -> "GridState":
        """Create a deep copy of the state."""
        return GridState(
            grid=copy_grid(self.grid),
            original_grid=copy_grid(self.original_grid),
            selected_cells=set(self.selected_cells),
        )
