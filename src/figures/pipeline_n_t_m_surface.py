"""Pipeline N×t×M Pareto surface — the dense sweep.

One figure = one claim: "The Pipeline cost-accuracy front is convex; the
locked cell (N=5, t=0.5, M=5) is a Pareto-optimal pick on it."

Reads docs/pareto_pipeline_sweep_v32.json (produced by
src/scripts/pareto_analysis.py). Output: docs/figures/pipeline_n_t_m_surface.{pdf,png}.

Styling principles (per paper.mplstyle + COLORS):
- Color encodes N (explorers); marker shape encodes t (temperature).
- Dominated points are muted (low alpha); frontier is highlighted.
- M is annotated inline on every point.
- Ablation (act-only) plotted as a distinct red X with explicit label.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figures._style import COLORS, save_paper, use_paper_style, REPO

DATA = REPO / "docs" / "pareto_pipeline_sweep_v32.json"

# Color encodes N — three-step ramp (light → muted → strong) within the
# pipeline-blue family.  Frontier points get drawn at full alpha; dominated
# points fade to 35% for context-without-clutter.
N_COLORS = {
    1: "#9BBFE0",
    2: "#5B9BD5",  # matches COLORS["pipeline"]
    5: "#1F4E79",
}
T_MARKERS = {0.0: "o", 0.5: "s", 1.0: "^"}
DOMINATED_ALPHA = 0.35
FRONTIER_ALPHA = 0.95


def main() -> None:
    use_paper_style()
    data = json.loads(DATA.read_text())
    surface = data["surface"]
    front = data["pareto_front"]
    ablations = data.get("ablations", [])

    # Set of (N, t, M) tuples on the Pareto front for visual highlighting
    front_keys = {(p["N"], p["t"], p["M"]) for p in front}

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    # Plot all surface points; dominated ones faded. Skip M=1 cells (pass@2
    # undefined — you need ≥2 candidates to pick the top-2).
    for p in surface:
        if p["pass_at_2"] is None:
            continue
        key = (p["N"], p["t"], p["M"])
        on_front = key in front_keys
        ax.scatter(
            p["usd_per_task"] * 100,
            p["pass_at_2"] * 100,
            c=N_COLORS[p["N"]],
            marker=T_MARKERS[p["t"]],
            s=60 + p["M"] * 14,
            edgecolor="black", linewidth=0.5,
            alpha=FRONTIER_ALPHA if on_front else DOMINATED_ALPHA,
            zorder=4 if on_front else 2,
        )
        # Label M only on frontier points to reduce clutter
        if on_front:
            ax.annotate(f"M={p['M']}",
                        xy=(p["usd_per_task"] * 100, p["pass_at_2"] * 100),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=7.5, color="0.25", fontweight=500)

    # Pareto front dashed line
    front_x = [p["usd_per_task"] * 100 for p in front]
    front_y = [p["pass_at_2"] * 100 for p in front]
    ax.plot(front_x, front_y, "--", color="0.4", linewidth=1.2, alpha=0.65,
            zorder=3, label="Pareto front")

    # Ablation overlay — distinct red X with text label
    for ab in ablations:
        if ab.get("pass_at_2") is None:
            continue
        ax.scatter(
            ab["usd_per_task"] * 100, ab["pass_at_2"] * 100,
            marker="X", color=COLORS["ablation"], s=170,
            edgecolor="black", linewidth=0.8, zorder=6,
        )
        ax.annotate(ab["label"],
                    xy=(ab["usd_per_task"] * 100, ab["pass_at_2"] * 100),
                    xytext=(8, -14), textcoords="offset points",
                    fontsize=9, color=COLORS["ablation"], fontweight=600)

    # Legends — color = N, marker = t. Two stacked legends in lower right.
    n_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=c,
               markersize=8, markeredgecolor="black", markeredgewidth=0.5,
               label=f"N={N}")
        for N, c in N_COLORS.items()
    ]
    t_handles = [
        Line2D([0], [0], marker=m, linestyle="", markerfacecolor="0.6",
               markersize=8, markeredgecolor="black", markeredgewidth=0.5,
               label=f"t={t}")
        for t, m in T_MARKERS.items()
    ]
    leg_n = ax.legend(handles=n_handles, loc="lower right",
                      title="N (explorers)", fontsize=8.5, title_fontsize=9)
    ax.add_artist(leg_n)
    ax.legend(handles=t_handles, loc="lower right",
              bbox_to_anchor=(1.0, 0.27),
              title="t (explorer temp)", fontsize=8.5, title_fontsize=9)

    ax.set_xlabel("Cost per task (¢)")
    ax.set_ylabel("Accuracy on ARC-AGI-1 eval (pass@2, %)")
    ax.set_title("Pipeline N × t × M cost-accuracy surface",
                 fontsize=13, fontweight=600, pad=14)
    ax.set_xlim(left=0)

    paths = save_paper(fig, "pipeline_n_t_m_surface")
    for p in paths:
        print(f"saved: {p}")


if __name__ == "__main__":
    main()
