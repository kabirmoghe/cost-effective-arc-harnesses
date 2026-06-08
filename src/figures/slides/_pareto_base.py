"""Shared cost-vs-accuracy Pareto renderer used across the slide deck.

Parameterized over which series to include (external comparators, our
architectures, callouts). Each slide-specific script just configures and calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figures._style import COLORS, save_paper, use_paper_style, REPO
from figures.slides._comparators import COMPARATORS

BOOT = REPO / "docs" / "bootstrap_cis.json"
PARETO_ORCH = REPO / "docs" / "pareto_orchestrator.json"
SLIDES_OUT = REPO / "docs" / "figures" / "slides"


def load_canonical() -> dict[str, dict]:
    """Returns our 4 architecture points (Baseline, CoT, Pipeline, Orchestrator)
    with task-level accuracy + measured cost."""
    boot = json.loads(BOOT.read_text())["per_architecture"]
    orch = json.loads(PARETO_ORCH.read_text())
    refs = orch["reference_systems"]
    b7_locked = orch["b7"]["m_sweep"][-1]      # M=5
    b8_locked = orch["b8"]["m_sweep"][-1]      # M=5
    return {
        "baseline": {
            "label": "Baseline",
            "accuracy": boot["baseline_pass_at_1"]["point"] * 100,
            "usd_per_task": refs["baseline"]["usd_per_task_atlas"],
            "color": COLORS["baseline"],
        },
        "cot": {
            "label": "CoT",
            "accuracy": boot["cot_pass_at_1"]["point"] * 100,
            "usd_per_task": refs["CoT"]["usd_per_task_atlas"],
            "color": COLORS["cot"],
        },
        "pipeline": {
            "label": "Pipeline",
            "accuracy": boot["pipeline_pass_at_2"]["point"] * 100,
            "usd_per_task": b7_locked["usd_per_task"],
            "color": COLORS["pipeline"],
        },
        "orchestrator": {
            "label": "Reflective Orchestrator",
            "accuracy": boot["orchestrator_pass_at_2"]["point"] * 100,
            "usd_per_task": b8_locked["usd_per_task"],
            "color": COLORS["orchestrator"],
        },
    }


def render_pareto(
    name: str,
    *,
    include_external: bool = True,
    include_ours: tuple[str, ...] = (),    # subset of ("baseline","cot","pipeline","orchestrator")
    callouts: list[dict] | None = None,    # list of {"text": "...", "anchor": (x,y), "offset": (dx,dy)}
    title: str | None = None,
) -> Path:
    use_paper_style()
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    # External comparators — neutral gray for visual hierarchy (our points pop)
    if include_external:
        for c in COMPARATORS:
            x = c["usd_per_task"] * 100  # cents
            ax.scatter(x, c["accuracy"], s=90, color="#999999",
                       edgecolor="black", linewidth=0.6, marker="s", zorder=3)
            ax.annotate(c["short"], xy=(x, c["accuracy"]),
                        xytext=(8, 5), textcoords="offset points",
                        fontsize=9, color="0.25", fontweight=500,
                        ha="left", va="bottom")

    # Our architectures
    canonical = load_canonical()
    for key in include_ours:
        p = canonical[key]
        x = p["usd_per_task"] * 100
        is_hero = key == "orchestrator"
        size = 180 if is_hero else 110
        ax.scatter(x, p["accuracy"], s=size, color=p["color"],
                   edgecolor="black", linewidth=0.8,
                   marker="o", zorder=5)
        offset = (-10, 8) if key in ("orchestrator", "pipeline") else (10, -6)
        ha = "right" if offset[0] < 0 else "left"
        va = "bottom" if offset[1] > 0 else "top"
        ax.annotate(f"{p['label']}\n{p['accuracy']:.1f}%",
                    xy=(x, p["accuracy"]),
                    xytext=offset, textcoords="offset points",
                    fontsize=10, color=p["color"], fontweight=600,
                    ha=ha, va=va)

    # Optional speaker-aid callouts (cost-multiplier comparisons)
    if callouts:
        for co in callouts:
            ax.annotate(co["text"],
                        xy=co["anchor"], xytext=co["offset"],
                        textcoords="offset points",
                        fontsize=9, color=COLORS["annotation"],
                        fontweight=500, fontstyle="italic",
                        arrowprops=dict(arrowstyle="->", color="0.5",
                                        connectionstyle="arc3,rad=-0.2",
                                        lw=0.9))

    ax.set_xscale("log")
    ax.set_xlabel("Cost per task (log scale)")
    ax.set_ylabel("Accuracy on ARC-AGI-1 eval (%)")
    if title:
        ax.set_title(title, fontsize=13, fontweight=600, pad=14)
    ax.set_ylim(0, 100)
    ax.set_xlim(0.05, 100_000)  # 0.05¢ to $1000

    major = [0.1, 1, 10, 100, 1000, 10_000]
    ax.set_xticks(major)
    ax.set_xticklabels(["0.1¢", "1¢", "10¢", "$1", "$10", "$100"])
    ax.grid(axis="y", alpha=0.25)

    # Tiny legend: only when both series present
    if include_external and include_ours:
        legend_handles = [
            Line2D([0], [0], marker="o", linestyle="", color=COLORS["orchestrator"],
                   markersize=8, markeredgecolor="black", markeredgewidth=0.5,
                   label="This work (V3.2)"),
            Line2D([0], [0], marker="s", linestyle="", color="#999999",
                   markersize=7, markeredgecolor="black", markeredgewidth=0.5,
                   label="Published systems"),
        ]
        ax.legend(handles=legend_handles, loc="lower right", fontsize=9, frameon=False)

    SLIDES_OUT.mkdir(parents=True, exist_ok=True)
    paths = save_paper(fig, f"slides/{name}")
    plt.close(fig)
    return paths[0]
