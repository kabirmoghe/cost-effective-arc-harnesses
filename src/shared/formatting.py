"""Grid formatting utilities for LLM prompts."""

from typing import List

from .types import Grid, Task


def grid_to_ascii(grid: Grid, show_coordinates: bool = True) -> str:
    """Convert a grid to ASCII representation for LLM context."""
    if not grid or not grid[0]:
        return "(empty grid)"

    rows = len(grid)
    cols = len(grid[0])
    lines = []

    if show_coordinates:
        col_header = "   " + " ".join(f"{c:1d}" for c in range(cols))
        lines.append(col_header)
        lines.append("   " + "-" * (cols * 2 - 1))

    for r in range(rows):
        row_prefix = f"{r:2d}|" if show_coordinates else ""
        lines.append(row_prefix + " ".join(str(grid[r][c]) for c in range(cols)))

    return "\n".join(lines)


def format_example_pair(input_grid: Grid, output_grid: Grid, example_num: int) -> str:
    """Format an input/output example pair for few-shot prompting."""
    lines = [f"Example {example_num}:"]
    lines.append("Input:")
    lines.append(grid_to_ascii(input_grid))
    lines.append("")
    lines.append("Output:")
    lines.append(grid_to_ascii(output_grid))
    return "\n".join(lines)


def format_task_examples(task: Task) -> str:
    """Format all training pairs from a task."""
    parts = []
    for i, example in enumerate(task.train, 1):
        parts.append(format_example_pair(example.input, example.output, i))
    return "\n\n".join(parts)
