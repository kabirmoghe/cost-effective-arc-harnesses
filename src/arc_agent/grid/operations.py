"""Low-level grid operations."""

from typing import List, Set, Tuple
from collections import deque

from ..data.types import Grid


def copy_grid(grid: Grid) -> Grid:
    """Create a deep copy of a grid."""
    return [row[:] for row in grid]


def grids_equal(grid1: Grid, grid2: Grid) -> bool:
    """Check if two grids are identical."""
    if len(grid1) != len(grid2):
        return False
    for r1, r2 in zip(grid1, grid2):
        if len(r1) != len(r2):
            return False
        if r1 != r2:
            return False
    return True


def is_valid_cell(grid: Grid, row: int, col: int) -> bool:
    """Check if (row, col) is within grid bounds."""
    if not grid:
        return False
    return 0 <= row < len(grid) and 0 <= col < len(grid[0])


def get_neighbors(row: int, col: int, include_diagonal: bool = False) -> List[Tuple[int, int]]:
    """Get neighboring cell coordinates (4-connected or 8-connected)."""
    neighbors = [
        (row - 1, col),  # up
        (row + 1, col),  # down
        (row, col - 1),  # left
        (row, col + 1),  # right
    ]
    if include_diagonal:
        neighbors.extend([
            (row - 1, col - 1),  # up-left
            (row - 1, col + 1),  # up-right
            (row + 1, col - 1),  # down-left
            (row + 1, col + 1),  # down-right
        ])
    return neighbors


def flood_fill_region(
    grid: Grid,
    start_row: int,
    start_col: int,
    new_color: int,
    include_diagonal: bool = False
) -> Grid:
    """Flood fill from a starting cell, replacing connected cells of the same color.

    Args:
        grid: The grid to fill (will be copied, not modified)
        start_row: Starting row
        start_col: Starting column
        new_color: Color to fill with
        include_diagonal: Whether to include diagonal neighbors

    Returns:
        New grid with flood fill applied
    """
    if not is_valid_cell(grid, start_row, start_col):
        return copy_grid(grid)

    result = copy_grid(grid)
    original_color = result[start_row][start_col]

    if original_color == new_color:
        return result

    # BFS flood fill
    queue = deque([(start_row, start_col)])
    visited = set()

    while queue:
        r, c = queue.popleft()

        if (r, c) in visited:
            continue
        if not is_valid_cell(result, r, c):
            continue
        if result[r][c] != original_color:
            continue

        visited.add((r, c))
        result[r][c] = new_color

        for nr, nc in get_neighbors(r, c, include_diagonal):
            if (nr, nc) not in visited:
                queue.append((nr, nc))

    return result


def get_connected_region(
    grid: Grid,
    start_row: int,
    start_col: int,
    include_diagonal: bool = False
) -> Set[Tuple[int, int]]:
    """Get all cells connected to the starting cell with the same color."""
    if not is_valid_cell(grid, start_row, start_col):
        return set()

    original_color = grid[start_row][start_col]
    visited = set()
    queue = deque([(start_row, start_col)])

    while queue:
        r, c = queue.popleft()

        if (r, c) in visited:
            continue
        if not is_valid_cell(grid, r, c):
            continue
        if grid[r][c] != original_color:
            continue

        visited.add((r, c))

        for nr, nc in get_neighbors(r, c, include_diagonal):
            if (nr, nc) not in visited:
                queue.append((nr, nc))

    return visited


def create_empty_grid(rows: int, cols: int, fill_value: int = 0) -> Grid:
    """Create a new grid filled with a single value."""
    return [[fill_value for _ in range(cols)] for _ in range(rows)]
