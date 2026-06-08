"""Slide 2 + 11: render ARC task 3a301edc cleanly (no rule annotations).

Renders all 5 training input→output pairs + the test input + (faded) test
output. ARC colors are the official 10-color palette from the public viewer.
The same image is used for both slide 2 (task framing) and slide 11
(canonical-failure callback) — they reference the same visual.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from figures._style import save_paper, use_paper_style, REPO

TASK_ID = "3a301edc"
TASK_PATH = REPO / "data" / "evaluation" / f"{TASK_ID}.json"

# Official ARC 10-color palette (0..9)
ARC_COLORS = [
    "#000000",  # 0 black
    "#0074D9",  # 1 blue
    "#FF4136",  # 2 red
    "#2ECC40",  # 3 green
    "#FFDC00",  # 4 yellow
    "#AAAAAA",  # 5 gray
    "#F012BE",  # 6 magenta
    "#FF851B",  # 7 orange
    "#7FDBFF",  # 8 light blue
    "#870C25",  # 9 maroon
]
CMAP = ListedColormap(ARC_COLORS)


def render_grid(ax, grid: list[list[int]], *, title: str = "") -> None:
    g = np.array(grid)
    ax.imshow(g, cmap=CMAP, vmin=0, vmax=9, interpolation="nearest")
    # Thin grid lines between cells
    ax.set_xticks(np.arange(-0.5, g.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, g.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#333333", linewidth=0.5)
    ax.tick_params(which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, fontweight=500, pad=4, color="0.15")


def main() -> None:
    use_paper_style()
    task = json.loads(TASK_PATH.read_text())
    n_train = len(task["train"])
    n_test = len(task["test"])

    # Layout: top row = inputs, bottom row = outputs. Train pairs left of test.
    n_cols = n_train + n_test
    fig, axes = plt.subplots(2, n_cols, figsize=(2.2 * n_cols, 5.0))

    for i, pair in enumerate(task["train"]):
        render_grid(axes[0, i], pair["input"],
                    title=f"Train {i+1} — input" if i == 0 else f"Train {i+1}")
        render_grid(axes[1, i], pair["output"],
                    title="Train 1 — output" if i == 0 else "")

    for j, pair in enumerate(task["test"]):
        col = n_train + j
        render_grid(axes[0, col], pair["input"], title=f"Test {j+1} — input")
        render_grid(axes[1, col], pair["output"], title="Test — expected output")

    fig.suptitle(f"ARC-AGI-1 task {TASK_ID} — 5 training pairs + 1 test input",
                 fontsize=12.5, fontweight=600, y=1.0)
    fig.tight_layout(rect=(0, 0.0, 1, 0.94))

    # After tight_layout finalizes positions: wrap the entire test column
    # (test input + test output) in a single dashed border, and place a
    # faint italic "HELD OUT" label above the test column.
    from matplotlib.patches import FancyBboxPatch
    fig.canvas.draw()
    test_col_axes = [axes[0, n_train + j] for j in range(n_test)] + \
                    [axes[1, n_train + j] for j in range(n_test)]
    # Compute union bbox of all test-column axes (in figure coords)
    bboxes = [a.get_position() for a in test_col_axes]
    x0 = min(b.x0 for b in bboxes)
    x1 = max(b.x1 for b in bboxes)
    y0 = min(b.y0 for b in bboxes)
    y1 = max(b.y1 for b in bboxes)
    pad_x = 0.012
    pad_y_bot = 0.020
    pad_y_top = 0.050   # extra room above axis titles
    border = FancyBboxPatch(
        (x0 - pad_x, y0 - pad_y_bot),
        (x1 - x0) + 2 * pad_x,
        (y1 - y0) + pad_y_bot + pad_y_top,
        boxstyle="round,pad=0",
        fill=False, edgecolor="#888888",
        linewidth=1.4, linestyle=(0, (5, 3)),
        transform=fig.transFigure, zorder=20,
    )
    fig.patches.append(border)
    # Faint italic "HELD OUT" label centered above the dashed border
    fig.text((x0 + x1) / 2, y1 + pad_y_top + 0.014, "HELD OUT",
             fontsize=10.5, fontweight=400, color="0.45",
             ha="center", va="bottom", style="italic",
             transform=fig.transFigure, zorder=21)

    paths = save_paper(fig, "slides/slide02_arc_task_3a301edc")
    for p in paths:
        print(f"saved: {p}")


if __name__ == "__main__":
    main()
