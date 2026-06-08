"""Slide 15: cumulative architectural-lift waterfall chart.

Each architectural addition is a floating Δ segment landing at the next
cumulative accuracy. End-state = 67.25% Orchestrator. Reads canonical
accuracies from docs/bootstrap_cis.json (single source of truth).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figures._style import COLORS, save_paper, use_paper_style, REPO

BOOT = REPO / "docs" / "bootstrap_cis.json"


def main() -> None:
    use_paper_style()
    boot = json.loads(BOOT.read_text())["per_architecture"]
    stages = [
        ("Baseline",                 boot["baseline_pass_at_1"]["point"] * 100, COLORS["baseline"]),
        ("+ CoT",                    boot["cot_pass_at_1"]["point"] * 100,      COLORS["cot"]),
        ("+ Pipeline",               boot["pipeline_pass_at_2"]["point"] * 100, COLORS["pipeline"]),
        ("+ Reflective Orchestrator", boot["orchestrator_pass_at_2"]["point"] * 100, COLORS["orchestrator"]),
    ]
    names = [s[0] for s in stages]
    cum = [s[1] for s in stages]
    colors = [s[2] for s in stages]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    # First bar = total (Baseline); subsequent bars are floating Δ segments
    bar_width = 0.62
    x = np.arange(len(stages))

    # Baseline: full bar from 0
    ax.bar(x[0], cum[0], width=bar_width, color=colors[0],
           edgecolor="black", linewidth=0.7, zorder=3)
    ax.text(x[0], cum[0] + 1.5, f"{cum[0]:.2f}%",
            ha="center", va="bottom", fontsize=10.5, fontweight=600,
            color="0.15")

    # Subsequent: floating Δ segment from prev to current
    for i in range(1, len(stages)):
        bottom = cum[i - 1]
        delta = cum[i] - bottom
        ax.bar(x[i], delta, bottom=bottom, width=bar_width, color=colors[i],
               edgecolor="black", linewidth=0.7, zorder=3)
        # Connector line from prev bar top → this bar bottom
        ax.plot([x[i - 1] + bar_width / 2, x[i] - bar_width / 2],
                [bottom, bottom], color="0.45", linestyle="--",
                linewidth=0.9, zorder=2)
        # Δ label centered on the floating segment
        ax.text(x[i], bottom + delta / 2,
                f"+{delta:.2f}pp",
                ha="center", va="center", fontsize=10, fontweight=600,
                color="white" if delta > 4 else "0.1")
        # Cumulative label above bar
        ax.text(x[i], cum[i] + 1.5, f"{cum[i]:.2f}%",
                ha="center", va="bottom", fontsize=10.5, fontweight=600,
                color="0.15")

    # End-to-end total annotation across the top
    total_lift = cum[-1] - cum[0]
    ax.annotate(
        f"End-to-end lift: +{total_lift:.2f}pp",
        xy=(x[-1], cum[-1]), xytext=(x[0] + 0.05, cum[-1] + 10),
        fontsize=11, fontweight=700, color=COLORS["orchestrator"],
        ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color=COLORS["orchestrator"],
                        lw=1.2, connectionstyle="arc3,rad=0.15"),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10.5)
    ax.set_ylabel("Accuracy on ARC-AGI-1 eval, pass@1 / pass@2 (%)")
    ax.set_ylim(0, 85)
    ax.set_title("Cumulative architectural lift on DeepSeek V3.2",
                 fontsize=13, fontweight=600, pad=14)
    ax.grid(axis="y", alpha=0.25)

    # Footnote on metric convention
    fig.text(0.02, -0.02,
             "Baseline + CoT report pass@1 (single attempt); Pipeline + Orchestrator report pass@2 (selected from M=5 candidates). "
             "All values task-level (any-test-pair-correct).",
             fontsize=7.5, color="0.4", ha="left", va="top", style="italic")

    paths = save_paper(fig, "slides/slide15_lift_waterfall")
    for p in paths:
        print(f"saved: {p}")


if __name__ == "__main__":
    main()
