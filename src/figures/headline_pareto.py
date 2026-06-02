"""Sample refactored paper figure: headline Pareto.

One figure = one claim: "Architectural progression Baseline → CoT → Pipeline →
Orchestrator lifts ARC-AGI-1 eval accuracy by ~5× over a tractable cost range."

Reads cached numbers from docs/pareto_orchestrator.json (produced by
src/scripts/pareto_orchestrator.py); does NOT recompute. Run after a fresh
data regen.

Output: docs/figures/headline_pareto.{pdf,png}

Style: src/figures/paper.mplstyle (Times, no top/right spines, vector export).
Colors: src/figures/_style.py COLORS dict.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from figures._style import COLORS, save_paper, use_paper_style, REPO

DATA = REPO / "docs" / "pareto_orchestrator.json"
BOOT = REPO / "docs" / "bootstrap_cis.json"


def main() -> None:
    use_paper_style()
    with DATA.open() as f:
        data = json.load(f)

    # Override Baseline + CoT accuracies with the canonical task-level numbers
    # from bootstrap_cis.json (per-task any-correct convention; matches the
    # pipeline pass@k convention so Baseline → CoT → Pipeline deltas are
    # apples-to-apples). The legacy `accuracy` in pareto_orchestrator.json is
    # per-test-pair for baseline/CoT, which mixes conventions on the same axis.
    boot = json.loads(BOOT.read_text())["per_architecture"]
    refs = data["reference_systems"]
    refs["baseline"]["accuracy"] = boot["baseline_pass_at_1"]["point"]
    refs["CoT"]["accuracy"] = boot["cot_pass_at_1"]["point"]

    b7 = data["b7"]["m_sweep"]
    b8 = data["b8"]["m_sweep"]

    # Metric kind is encoded by marker shape (○ pass@1, ◇ pass@2) so the
    # text label can be terse (name + accuracy + cost only).
    MARK_PASS_AT_1 = "o"
    MARK_PASS_AT_2 = "D"
    points = [
        {
            "label": "Baseline",
            "metric": "pass@1", "marker": MARK_PASS_AT_1,
            "x": refs["baseline"]["usd_per_task_atlas"] * 100,
            "y": refs["baseline"]["accuracy"] * 100,
            "color": COLORS["baseline"],
            "annot_offset": (10, -8),
        },
        {
            "label": "CoT",
            "metric": "pass@1", "marker": MARK_PASS_AT_1,
            "x": refs["CoT"]["usd_per_task_atlas"] * 100,
            "y": refs["CoT"]["accuracy"] * 100,
            "color": COLORS["cot"],
            "annot_offset": (10, -8),
        },
        {
            "label": "Pipeline",
            "metric": "pass@2", "marker": MARK_PASS_AT_2,
            "x": b7[-1]["usd_per_task"] * 100,
            "y": b7[-1]["pass_at_2"] * 100,
            "color": COLORS["pipeline"],
            "annot_offset": (-10, -8),
            "annot_ha": "right",
        },
        {
            "label": "Reflective Orchestrator",
            "metric": "pass@2", "marker": MARK_PASS_AT_2,
            "x": b8[-1]["usd_per_task"] * 100,
            "y": b8[-1]["pass_at_2"] * 100,
            "color": COLORS["orchestrator"],
            "annot_offset": (-10, 8),
            "annot_ha": "right",
            "is_hero": True,
        },
    ]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    # Frontier line (gray, subtle, sorted by cost)
    front = sorted(points, key=lambda p: p["x"])
    ax.plot([p["x"] for p in front], [p["y"] for p in front],
            "--", color=COLORS["frontier"], linewidth=1.0, zorder=1, alpha=0.6)

    # Plot each point with direct label (metric kind = marker shape, see legend)
    for p in points:
        size = 130 if p.get("is_hero") else 80
        ax.scatter(p["x"], p["y"], s=size, color=p["color"],
                   edgecolor="black", linewidth=0.6, zorder=4,
                   marker=p["marker"])
        cost_str = f"{p['x']:.2f}¢" if p["x"] < 1 else f"{p['x']:.0f}¢"
        label_text = f"{p['label']}\n{p['y']:.1f}% · {cost_str}"
        ax.annotate(label_text,
                    xy=(p["x"], p["y"]),
                    xytext=p["annot_offset"],
                    textcoords="offset points",
                    fontsize=9, color=COLORS["annotation"],
                    ha=p.get("annot_ha", "left"),
                    va="bottom" if p["annot_offset"][1] > 0 else "top")

    # Tiny legend keyed only to marker shape (gray, neutral — does NOT
    # encode color, which is reserved for architectural identity).
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker=MARK_PASS_AT_1, linestyle="",
               color="0.4", markersize=7, markeredgecolor="black",
               markeredgewidth=0.5, label="pass@1 (single attempt)"),
        Line2D([0], [0], marker=MARK_PASS_AT_2, linestyle="",
               color="0.4", markersize=7, markeredgecolor="black",
               markeredgewidth=0.5, label="pass@2 (selected from M)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8.5)

    ax.set_xscale("log")
    ax.set_xlabel("Cost per task (log scale)")
    ax.set_ylabel("Accuracy on ARC-AGI-1 eval (%)")
    ax.set_title("Architectural progression: harnesses on DeepSeek V3.2",
                 fontsize=13, fontweight=600, pad=14)
    ax.set_ylim(0, 78)
    ax.set_xlim(0.1, 200)

    # Dense, human-readable cost ticks. Major in dark, minor in light gray.
    major_ticks = [0.1, 1, 10, 100]
    minor_ticks = [0.2, 0.5, 2, 5, 20, 50]
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_xticklabels(["0.1¢", "1¢", "10¢", "100¢"])
    ax.set_xticklabels(["0.2¢", "0.5¢", "2¢", "5¢", "20¢", "50¢"], minor=True)
    ax.tick_params(axis="x", which="minor", labelsize=7.5, colors="0.45")

    paths = save_paper(fig, "headline_pareto")
    for p in paths:
        print(f"saved: {p}")


if __name__ == "__main__":
    main()
