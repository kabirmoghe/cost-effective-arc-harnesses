"""Pipeline N×M curves faceted by explorer temperature t.

Three small subplots side-by-side, one per t ∈ {0.0, 0.5, 1.0}. Each subplot
shows three N-curves (N ∈ {1, 2, 5}) connecting M=1..5 at that temperature.
Color encodes N only; no shape encoding.

The canonical Pipeline cell (N=5, t=0.5, M=5) is highlighted with a star
overlay on the t=0.5 panel — it's the Pareto-optimal pick across the full
N×t×M surface (not just within its panel). Ablations are NOT shown here;
they live in a separate ablations table.

Reads docs/pareto_pipeline_sweep_v32.json. Output:
docs/figures/pipeline_n_t_m_faceted.{pdf,png}.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figures._style import COLORS, save_paper, use_paper_style, REPO

DATA = REPO / "docs" / "pareto_pipeline_sweep_v32.json"

# Color encodes N within the Pipeline-blue family
N_COLORS = {
    1: "#9BBFE0",
    2: "#5B9BD5",
    5: "#1F4E79",
}
T_VALUES = [0.0, 0.5, 1.0]


def main() -> None:
    use_paper_style()
    data = json.loads(DATA.read_text())
    surface = data["surface"]

    # Index by (N, t) → list of {M, x, y} sorted by M, skipping M=1 (no pass@2)
    by_nt: dict[tuple[int, float], list[dict]] = defaultdict(list)
    for p in surface:
        if p["pass_at_2"] is None:
            continue
        by_nt[(p["N"], p["t"])].append({
            "M": p["M"],
            "x": p["usd_per_task"] * 100,
            "y": p["pass_at_2"] * 100,
        })
    for k in by_nt:
        by_nt[k].sort(key=lambda d: d["M"])

    # Shared axis limits so panels are comparable
    all_x = [d["x"] for pts in by_nt.values() for d in pts]
    all_y = [d["y"] for pts in by_nt.values() for d in pts]
    x_lim = (0, max(all_x) * 1.1)
    y_lim = (min(all_y) - 4, max(all_y) + 4)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0), sharey=True)

    for ax, t in zip(axes, T_VALUES):
        for N in (1, 2, 5):
            pts = by_nt.get((N, t), [])
            if not pts:
                continue
            xs = [d["x"] for d in pts]
            ys = [d["y"] for d in pts]
            ax.plot(xs, ys, "-", color=N_COLORS[N],
                    marker="o", markersize=7,
                    markeredgecolor="black", markeredgewidth=0.5,
                    linewidth=1.8, label=f"N={N}", zorder=3)
            # M annotations on the M=5 endpoint only (less clutter)
            last = pts[-1]
            ax.annotate(f"M=5", xy=(last["x"], last["y"]), xytext=(6, 4),
                        textcoords="offset points", fontsize=7.5,
                        color=N_COLORS[N], fontweight=500)

        # Highlight the canonical cell (N=5, t=0.5, M=5) — the Pareto-optimal
        # pick across the FULL N×t×M surface. Star overlay on its panel.
        if t == 0.5:
            locked = next((d for d in by_nt.get((5, 0.5), []) if d["M"] == 5), None)
            if locked:
                ax.scatter(locked["x"], locked["y"], s=240, marker="*",
                           color=N_COLORS[5], edgecolor="black", linewidth=1.0,
                           zorder=7)
                ax.annotate("Canonical\nPipeline",
                            xy=(locked["x"], locked["y"]),
                            xytext=(-12, 8), textcoords="offset points",
                            fontsize=8.5, color=N_COLORS[5], fontweight=700,
                            ha="right", va="bottom")

        ax.set_title(f"t = {t}", fontsize=11, fontweight=500, pad=8)
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_xlabel("Cost per task (¢)")

    axes[0].set_ylabel("Accuracy on ARC-AGI-1 eval (pass@2, %)")

    # Shared legend: N + locked-cell star
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="-", color=N_COLORS[N],
               markersize=7, markeredgecolor="black", markeredgewidth=0.5,
               label=f"N={N} explorers")
        for N in (1, 2, 5)
    ]
    legend_handles.append(
        Line2D([0], [0], marker="*", linestyle="", color=N_COLORS[5],
               markersize=12, markeredgecolor="black", markeredgewidth=0.8,
               label="Canonical Pipeline (Pareto-optimal across surface)")  # noqa: E501
    )
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=len(legend_handles),
               fontsize=9, frameon=False)

    fig.suptitle("Pipeline N × M cost-accuracy curves, faceted by explorer temperature t",
                 fontsize=13, fontweight=600, y=1.0)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))

    paths = save_paper(fig, "pipeline_n_t_m_faceted")
    for p in paths:
        print(f"saved: {p}")


if __name__ == "__main__":
    main()
