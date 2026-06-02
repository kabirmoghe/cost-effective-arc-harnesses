"""M-sweep: Pipeline vs Reflective Orchestrator across M ∈ {1..5}.

One figure = one claim: "Orchestrator Pareto-dominates Pipeline at every M."

Reads docs/pareto_orchestrator.json. Output: docs/figures/m_sweep.{pdf,png}.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figures._style import COLORS, save_paper, use_paper_style, REPO

DATA = REPO / "docs" / "pareto_orchestrator.json"


def main() -> None:
    use_paper_style()
    data = json.loads(DATA.read_text())
    b7 = data["b7"]["m_sweep"]  # Pipeline
    b8 = data["b8"]["m_sweep"]  # Orchestrator

    def cost_y(points):
        # x: cents per task; y: pass@2 if available else pass@1 (only at M=1)
        x = [p["usd_per_task"] * 100 for p in points]
        y = [(p["pass_at_2"] if p["pass_at_2"] is not None else p["pass_at_1"]) * 100 for p in points]
        return x, y

    pipeline_x, pipeline_y = cost_y(b7)
    orch_x, orch_y = cost_y(b8)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))

    ax.plot(pipeline_x, pipeline_y, "-", color=COLORS["pipeline"],
            linewidth=2.2, marker="D", markersize=8,
            markeredgecolor="black", markeredgewidth=0.5,
            label="Pipeline", zorder=3)
    ax.plot(orch_x, orch_y, "-", color=COLORS["orchestrator"],
            linewidth=2.6, marker="D", markersize=9,
            markeredgecolor="black", markeredgewidth=0.5,
            label="Reflective Orchestrator", zorder=4)

    # Annotate M values inline next to each marker — direct labels, no legend
    # entry needed for them.
    # Pipeline: M=1 anchor stays below-right of its hollow circle (bottom
    # of the chart, no eclipse risk). M=2..5 placed ABOVE-LEFT of their
    # diamonds — opposite side from the rising curve, so the label sits in
    # clear space rather than getting eclipsed by the next marker.
    for p, x, y in zip(b7, pipeline_x, pipeline_y):
        if p["M"] == 1:
            off, ha, va = (8, -8), "left", "top"
        else:
            off, ha, va = (-8, 10), "right", "bottom"
        ax.annotate(f"M={p['M']}", xy=(x, y), xytext=off,
                    textcoords="offset points", fontsize=8.5,
                    color=COLORS["pipeline"], fontweight=500,
                    ha=ha, va=va)
    # Orchestrator: consistently below-right of every marker (curve rises
    # up-right, so below-right keeps labels in the empty area below the line).
    for p, x, y in zip(b8, orch_x, orch_y):
        ax.annotate(f"M={p['M']}", xy=(x, y), xytext=(8, -10),
                    textcoords="offset points", fontsize=8.5,
                    color=COLORS["orchestrator"], fontweight=500,
                    ha="left", va="top")

    # Highlight the M=1 anchor on each line with a hollow circle (pass@1 only)
    ax.scatter(pipeline_x[0], pipeline_y[0], s=120, facecolor="white",
               edgecolor=COLORS["pipeline"], linewidth=1.8, zorder=5)
    ax.scatter(orch_x[0], orch_y[0], s=140, facecolor="white",
               edgecolor=COLORS["orchestrator"], linewidth=1.8, zorder=5)

    ax.set_xlabel("Cost per task (¢)")
    ax.set_ylabel("Accuracy on ARC-AGI-1 eval (%)")
    ax.set_title("Pipeline vs Reflective Orchestrator across M",
                 fontsize=13, fontweight=600, pad=14)
    ax.set_ylim(40, 75)

    # Build legend — color encodes architecture, hollow marker encodes M=1 (pass@1)
    legend_handles = [
        Line2D([0], [0], marker="D", linestyle="-", color=COLORS["pipeline"],
               markersize=8, markeredgecolor="black", markeredgewidth=0.5,
               label="Pipeline"),
        Line2D([0], [0], marker="D", linestyle="-", color=COLORS["orchestrator"],
               markersize=8, markeredgecolor="black", markeredgewidth=0.5,
               label="Reflective Orchestrator"),
        Line2D([0], [0], marker="o", linestyle="", color="white",
               markerfacecolor="white", markeredgecolor="0.4", markeredgewidth=1.5,
               markersize=8, label="M=1 (pass@1 only)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)

    paths = save_paper(fig, "m_sweep_pipeline_vs_orchestrator")
    for p in paths:
        print(f"saved: {p}")


if __name__ == "__main__":
    main()
