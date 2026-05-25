"""Train-set failure feedback rendering for the B5 refinement loop.

After the definer's `transform()` runs on the training pairs, any failing pair
is formatted into a user-message block via `render_train_feedback`:
    input grid + expected output + predicted output

Pre-computed diff (X-grid + cell list) is **off by default** — the model has
input/expected/predicted and can compute the discrepancy itself. The diff
helpers (`grid_diff_ascii`, `_list_diff_cells`) are kept in this module and
exposed via `render_train_feedback(..., include_diff=True)` for a B5
sub-ablation if richer feedback ever turns out to matter.
"""

from __future__ import annotations

from typing import Any

from shared.formatting import grid_to_ascii


def grid_diff_ascii(expected: list[list[int]], predicted: list[list[int]]) -> str:
    """Cell-level diff grid: '.' for matches, 'X' for mismatches.

    Falls back to a shape-mismatch line if the two grids differ in dimensions.
    """
    if not expected or not expected[0]:
        return "(expected grid is empty)"
    if not predicted or not predicted[0]:
        return "(predicted grid is empty)"
    e_rows, e_cols = len(expected), len(expected[0])
    p_rows, p_cols = len(predicted), len(predicted[0])
    if (e_rows, e_cols) != (p_rows, p_cols):
        return (
            f"(shape mismatch: expected {e_rows}x{e_cols}, "
            f"got {p_rows}x{p_cols} — cell-level diff not applicable)"
        )
    lines = ["   " + " ".join(f"{c:1d}" for c in range(e_cols))]
    lines.append("   " + "-" * (e_cols * 2 - 1))
    for r in range(e_rows):
        markers = " ".join(
            "." if expected[r][c] == predicted[r][c] else "X"
            for c in range(e_cols)
        )
        lines.append(f"{r:2d}|{markers}")
    return "\n".join(lines)


def _list_diff_cells(
    expected: list[list[int]],
    predicted: list[list[int]],
    limit: int = 12,
) -> str:
    """Compact textual listing of mismatched cells, capped to keep tokens bounded."""
    if not expected or not predicted:
        return ""
    if (len(expected), len(expected[0])) != (len(predicted), len(predicted[0])):
        return ""
    diffs = []
    for r in range(len(expected)):
        for c in range(len(expected[0])):
            if expected[r][c] != predicted[r][c]:
                diffs.append(
                    f"(r={r},c={c}) expected={expected[r][c]} got={predicted[r][c]}"
                )
    if not diffs:
        return ""
    if len(diffs) <= limit:
        return "Differing cells: " + "; ".join(diffs) + "."
    return (
        "Differing cells: "
        + "; ".join(diffs[:limit])
        + f"; ... ({len(diffs) - limit} more not shown)."
    )


def render_train_feedback(
    failing_pairs: list[dict[str, Any]],
    iteration: int,
    max_iters: int,
    include_diff: bool = False,
) -> str:
    """Build the user-message string for one refinement iteration.

    `failing_pairs` is a list of dicts, each with keys:
        pair_index : int                  (0-indexed train pair number)
        input      : Grid                 (the input grid from this train example)
        expected   : Grid                 (the expected output)
        predicted  : Grid | None          (your transform's output, if it executed)
        error      : str | None           (set if your transform errored on this pair)

    `iteration` is 1-indexed (1..max_iters). On the final iteration, prepend
    an explicit "this is your final refinement attempt" note so the model knows
    the selection contract.
    """
    is_final = iteration >= max_iters

    blocks: list[str] = []

    if is_final:
        blocks.append(
            "Note: this is your final refinement attempt for this transformation. "
            "After this, the version with the highest training-set score across all "
            "attempts will be selected as your final answer."
        )

    if failing_pairs:
        n_fail = len(failing_pairs)
        blocks.append(
            f"Your transform did not solve all training pairs "
            f"(refinement attempt {iteration} of {max_iters}; "
            f"{n_fail} pair{'s' if n_fail != 1 else ''} failed). "
            f"Failing pair details below:"
        )

    for fp in failing_pairs:
        pair_idx = fp.get("pair_index", 0)
        section: list[str] = [f"\n— Training pair {pair_idx + 1} —"]

        if fp.get("error"):
            section.append(f"Your transform raised an execution error on this pair:")
            section.append("```")
            section.append(str(fp["error"])[:800])
            section.append("```")
        else:
            input_grid = fp.get("input")
            expected = fp.get("expected")
            predicted = fp.get("predicted")
            if input_grid is not None:
                section.append("Input:")
                section.append(grid_to_ascii(input_grid))
                section.append("")
            if expected is not None:
                section.append("Expected output:")
                section.append(grid_to_ascii(expected))
                section.append("")
            if predicted is not None:
                section.append("Your transform produced:")
                section.append(grid_to_ascii(predicted))
                section.append("")
            if include_diff and expected is not None and predicted is not None:
                section.append("Diff (X = mismatched cell, . = matching):")
                section.append(grid_diff_ascii(expected, predicted))
                cell_list = _list_diff_cells(expected, predicted)
                if cell_list:
                    section.append("")
                    section.append(cell_list)

        blocks.append("\n".join(section))

    blocks.append(
        "\nRefine your transform() function to handle these failures. "
        "Use `think` to diagnose where the current logic goes wrong, then call "
        "`define_transformation` with the corrected code."
    )

    return "\n".join(blocks)
