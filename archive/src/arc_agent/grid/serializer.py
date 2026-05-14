"""Grid serialization for LLM context."""

from typing import List, Set, Tuple, Optional

from ..data.types import Grid, COLOR_NAMES


def grid_to_ascii(
    grid: Grid,
    selected_cells: Optional[Set[Tuple[int, int]]] = None,
    show_coordinates: bool = True
) -> str:
    """Convert a grid to ASCII representation for LLM context.

    Args:
        grid: 2D array of color values (0-9)
        selected_cells: Set of (row, col) tuples that are selected
        show_coordinates: Whether to show row/col indices

    Returns:
        ASCII string representation of the grid
    """
    if not grid or not grid[0]:
        return "(empty grid)"

    selected = selected_cells or set()
    rows = len(grid)
    cols = len(grid[0])

    lines = []

    # Column headers
    if show_coordinates:
        col_header = "   " + " ".join(f"{c:1d}" for c in range(cols))
        lines.append(col_header)
        lines.append("   " + "-" * (cols * 2 - 1))

    for r in range(rows):
        if show_coordinates:
            row_prefix = f"{r:2d}|"
        else:
            row_prefix = ""

        row_cells = []
        for c in range(cols):
            val = grid[r][c]
            cell_str = str(val)
            # Mark selected cells with brackets
            if (r, c) in selected:
                cell_str = f"[{val}]"
            row_cells.append(cell_str)

        # Join with spaces, handling bracketed cells
        if selected:
            line = row_prefix
            for i, cell in enumerate(row_cells):
                if i > 0:
                    line += " " if len(row_cells[i-1]) == 1 and len(cell) == 1 else ""
                line += cell
            lines.append(line)
        else:
            lines.append(row_prefix + " ".join(row_cells))

    return "\n".join(lines)


def grid_to_string(grid: Grid) -> str:
    """Convert grid to compact string (no coordinates)."""
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def grid_dimensions(grid: Grid) -> Tuple[int, int]:
    """Return (rows, cols) dimensions of grid."""
    if not grid:
        return (0, 0)
    return (len(grid), len(grid[0]) if grid[0] else 0)


def format_example_pair(input_grid: Grid, output_grid: Grid, example_num: int) -> str:
    """Format an input/output example pair for few-shot prompting."""
    lines = [f"Example {example_num}:"]
    lines.append("Input:")
    lines.append(grid_to_ascii(input_grid))
    lines.append("")
    lines.append("Output:")
    lines.append(grid_to_ascii(output_grid))
    return "\n".join(lines)


def describe_grid_changes(before: Grid, after: Grid) -> str:
    """Describe what changed between two grids."""
    if not before or not after:
        return "Grid changed"

    before_rows, before_cols = grid_dimensions(before)
    after_rows, after_cols = grid_dimensions(after)

    changes = []

    if (before_rows, before_cols) != (after_rows, after_cols):
        changes.append(f"Size changed from {before_rows}x{before_cols} to {after_rows}x{after_cols}")

    # Count cell changes (if same size)
    if (before_rows, before_cols) == (after_rows, after_cols):
        changed_cells = []
        for r in range(before_rows):
            for c in range(before_cols):
                if before[r][c] != after[r][c]:
                    changed_cells.append((r, c, before[r][c], after[r][c]))

        if changed_cells:
            if len(changed_cells) <= 5:
                for r, c, old, new in changed_cells:
                    changes.append(f"Cell ({r},{c}): {COLOR_NAMES.get(old, old)} -> {COLOR_NAMES.get(new, new)}")
            else:
                changes.append(f"{len(changed_cells)} cells changed")

    return "; ".join(changes) if changes else "No changes"
