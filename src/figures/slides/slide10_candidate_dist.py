"""Slide 10: bimodal candidate-distribution histogram.

For each of the 400 eval tasks in the locked Pipeline (M=5), counts how many
definers produced a correct candidate (correct = `test_results` all-correct).
Plots the distribution across tasks.

The intellectual centerpiece: most tasks are at 0/5 or at 4-5/5 — selection
can't recover from "no correct candidate" cases, hence "generation-bound."
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psycopg2

from figures._style import COLORS, save_paper, use_paper_style

PIPELINE_RUN = "019e7ec8-b83c-7ee2-96e8-12e3afd28d37"
DB_URL = "postgresql://kabirmoghe:postgres@localhost:5433/arc_evaluation"


def main() -> None:
    use_paper_style()
    conn = psycopg2.connect(DB_URL)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT task_id, output FROM definers WHERE run_id=%s",
            (PIPELINE_RUN,),
        )
        rows = cur.fetchall()
    by_task: dict[str, list[bool]] = defaultdict(list)
    for tid, output in rows:
        tr = output.get("test_results") or []
        ok = bool(tr) and all(t.get("correct") for t in tr)
        by_task[tid].append(ok)

    # Cap at M=5 to match the canonical Pipeline configuration
    counts = [sum(v[:5]) for v in by_task.values()]
    n_tasks = len(counts)
    print(f"n_tasks = {n_tasks}, distribution: " +
          ", ".join(f"{k}/{5}={counts.count(k)}" for k in range(6)))

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    bins = np.arange(7) - 0.5  # bin edges centered on integers 0..5
    counts_arr, _, patches = ax.hist(counts, bins=bins, color=COLORS["pipeline"],
                                      edgecolor="black", linewidth=0.8, zorder=3)
    # Highlight the two failure-mode endpoints
    patches[0].set_color("#B85450")  # 0/5: no correct definer (selection can't help)
    patches[5].set_color("#3D8B5E")  # 5/5: all correct (selection trivial)
    patches[0].set_edgecolor("black")
    patches[5].set_edgecolor("black")

    # Value labels above bars
    for i, c in enumerate(counts_arr):
        ax.text(i, c + 4, f"{int(c)}", ha="center", va="bottom",
                fontsize=10, color="0.2", fontweight=500)

    # Annotate the failure modes
    ax.annotate("Selection\nbottleneck\n(no correct candidate)",
                xy=(0, counts_arr[0]),
                xytext=(0.6, counts_arr[0] - 30),
                fontsize=9, color="#7A2E2E", fontweight=500, ha="left",
                arrowprops=dict(arrowstyle="->", color="#7A2E2E", lw=0.8))
    ax.annotate("Selection trivial\n(all correct)",
                xy=(5, counts_arr[5]),
                xytext=(3.6, counts_arr[5] + 6),
                fontsize=9, color="#2E7A4E", fontweight=500, ha="left",
                arrowprops=dict(arrowstyle="->", color="#2E7A4E", lw=0.8))

    ax.set_xlabel("Number of definers (out of M=5) producing a correct candidate")
    ax.set_ylabel("Number of tasks")
    ax.set_xticks(range(6))
    ax.set_title("Pipeline candidate distribution across the 400-task eval set",
                 fontsize=12.5, fontweight=600, pad=12)
    ax.set_ylim(0, max(counts_arr) * 1.18)
    ax.grid(axis="y", alpha=0.25)

    # Footnote framing the diagnostic
    fig.text(0.02, -0.02,
             "Bimodal: ~46% of tasks have 0/5 correct definers; ~23% have 5/5. "
             "The middle bins are sparse — confirming the system is generation-bound (no selection rule can recover the 0/5 mass).",
             fontsize=7.5, color="0.4", ha="left", va="top", style="italic")

    paths = save_paper(fig, "slides/slide10_candidate_distribution")
    for p in paths:
        print(f"saved: {p}")


if __name__ == "__main__":
    main()
