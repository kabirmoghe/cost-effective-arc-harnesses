"""Shared paper-figure style: matplotlib stylesheet + color palette.

Usage:
    from figures._style import use_paper_style, COLORS, save_paper
    use_paper_style()
    fig, ax = plt.subplots()
    # ... plot ...
    save_paper(fig, "headline_pareto")  # writes pdf + png

Rationale:
- Single source of truth so all paper figures look uniform.
- Visual hierarchy: Reflective Orchestrator is the hero (full pop color);
  Pipeline is the comparison baseline (muted same-family color).
- Color-blind safe (Wong palette, adjusted for hierarchy).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
MPLSTYLE = Path(__file__).parent / "paper.mplstyle"
FIG_DIR = REPO / "docs" / "figures"

COLORS = {
    # Baselines (gray — context, not claim)
    "baseline":     "#777777",
    "cot":          "#444444",

    # Pipeline (muted blue — the comparison baseline, not the hero)
    "pipeline":     "#5B9BD5",
    "pipeline_dim": "#B8D4EC",  # for dominated points / surfaces

    # Reflective Orchestrator (full pop — the hero)
    "orchestrator":     "#D55E00",
    "orchestrator_dim": "#F2C2A0",

    # Ablations / supporting
    "ablation":     "#CC3311",
    "frontier":     "#999999",
    "annotation":   "#333333",
}


def use_paper_style() -> None:
    plt.style.use(str(MPLSTYLE))


def save_paper(fig, name: str, formats=("pdf", "png")) -> list[Path]:
    """Save a figure to docs/figures/ in both vector + raster formats.

    PDF is canonical for the paper; PNG is for screen preview / slides.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = FIG_DIR / f"{name}.{fmt}"
        fig.savefig(path)
        paths.append(path)
    return paths
