"""Render the Track A Pareto scaffold as a paper-quality figure.

Reads `output/track_a_preview.json` (produced by `stats.preview --json`) and
writes a cost-accuracy scatter to the same directory in both PNG (150 DPI) and
PDF (vector) so the figure is ready for thesis insertion.

Design choices:
- log-x for cost (range spans ~70× across architectures)
- color by architecture family (single-call vs pipeline) so the qualitative
  gap is visible before reading the labels
- Pareto-frontier overlay across the non-dominated points
- annotation offsets tuned to avoid overlap; if you add points and labels
  collide, tune them in `LABEL_OFFSETS` below.

Usage:
    uv run python -m stats.plot_pareto
    uv run python -m stats.plot_pareto --in path.json --out-stem out/path
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

FAMILY = {
    "arc_base": "single-call",
    "baseline": "single-call",
    "CoT": "single-call",
    "pipeline_M1": "pipeline",
    "pipeline_M5": "pipeline",
}

PRETTY_LABEL = {
    "arc_base": "arc_base (published harness)",
    "baseline": "Baseline (no-CoT)",
    "CoT": "CoT",
    "pipeline_M1": "Pipeline (M=1)",
    "pipeline_M5": "Pipeline (M=5, pass@2)",
}

# (dx_pts, dy_pts) per label — manual de-overlap tuning.
LABEL_OFFSETS = {
    "arc_base": (10, -4),
    "baseline": (10, 6),
    "CoT": (10, -10),
    "pipeline_M1": (-10, 12),
    "pipeline_M5": (-12, -16),
}


def _pareto_frontier(points: list[dict]) -> list[dict]:
    """Lower-cost, higher-accuracy non-dominated points, sorted by cost."""
    pts = sorted(points, key=lambda p: p["cost_per_task"])
    frontier: list[dict] = []
    best_acc = -1.0
    for p in pts:
        if p["accuracy"] > best_acc:
            frontier.append(p)
            best_acc = p["accuracy"]
    return frontier


def render(points: list[dict], out_stem: Path, title: str, subtitle: str) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    palette = {"single-call": "#4C72B0", "pipeline": "#C44E52"}

    for fam in ("single-call", "pipeline"):
        fam_pts = [p for p in points if FAMILY.get(p["label"]) == fam]
        if not fam_pts:
            continue
        ax.scatter(
            [p["cost_per_task"] for p in fam_pts],
            [p["accuracy"] * 100 for p in fam_pts],
            s=110, color=palette[fam], edgecolor="white", linewidth=1.2,
            label=fam, zorder=3,
        )

    # Pareto frontier
    frontier = _pareto_frontier(points)
    ax.plot(
        [p["cost_per_task"] for p in frontier],
        [p["accuracy"] * 100 for p in frontier],
        linestyle="--", color="#555555", linewidth=1.0, alpha=0.7,
        zorder=2, label="Pareto frontier",
    )

    for p in points:
        dx, dy = LABEL_OFFSETS.get(p["label"], (8, 6))
        ax.annotate(
            PRETTY_LABEL.get(p["label"], p["label"]),
            (p["cost_per_task"], p["accuracy"] * 100),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=10, color="#222",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Cost per task (USD, DeepSeek V3.2 list rates)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title, fontsize=13, loc="left", pad=14)
    if subtitle:
        ax.text(
            0.0, 1.02, subtitle, transform=ax.transAxes,
            fontsize=9, color="#666", ha="left", va="bottom",
        )

    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    ax.set_ylim(0, max(p["accuracy"] for p in points) * 100 * 1.18)
    ax.margins(x=0.18)

    sns.despine(ax=ax, left=False, bottom=False)
    fig.tight_layout()

    png_path = out_stem.with_suffix(".png")
    pdf_path = out_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in", dest="in_path", type=str,
        default="output/track_a_preview.json",
    )
    parser.add_argument(
        "--out-stem", type=str,
        default="output/track_a_pareto",
        help="Output path without extension; .png and .pdf both written.",
    )
    parser.add_argument(
        "--title", type=str,
        default="Cost vs. accuracy across architectures",
    )
    parser.add_argument(
        "--subtitle", type=str,
        default="DeepSeek V3.2 (May 2026 preview); ARC-AGI-1 evaluation split, n=400 tasks",
    )
    args = parser.parse_args()

    with open(args.in_path) as f:
        data = json.load(f)
    render(data["a3"], Path(args.out_stem), args.title, args.subtitle)


if __name__ == "__main__":
    main()
