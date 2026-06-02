"""Cross-model Pareto: V3.2 vs Qwen3 across all four architectures on n=99.

One figure = one claim: "The architectural progression (Baseline → CoT → Pipeline
→ Orchestrator) transfers to a second open-weight model — in direction and
magnitude — although Qwen3's absolute level trails V3.2 throughout."

Two curves on shared cost / accuracy axes:
  V3.2  (filled markers, blue/orange family + grays for baselines)
  Q3    (hollow markers, same color family)

Reads docs/cross_model_pareto_n99.json. Output: docs/figures/cross_model_pareto.{pdf,png}.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figures._style import COLORS, save_paper, use_paper_style, REPO

DATA = REPO / "docs" / "cross_model_pareto_n99.json"


ARCH_ORDER = ["baseline", "cot", "pipeline", "orchestrator"]
ARCH_LABEL = {
    "baseline": "Baseline",
    "cot": "CoT",
    "pipeline": "Pipeline",
    "orchestrator": "Reflective Orchestrator",
}
ARCH_COLOR = {
    "baseline": COLORS["baseline"],
    "cot": COLORS["cot"],
    "pipeline": COLORS["pipeline"],
    "orchestrator": COLORS["orchestrator"],
}
# pass@1 for single-call architectures, pass@2 for ensembled ones.
ARCH_METRIC = {"baseline": "pass_at_1", "cot": "pass_at_1",
               "pipeline": "pass_at_2", "orchestrator": "pass_at_2"}


def main() -> None:
    use_paper_style()
    data = json.loads(DATA.read_text())
    by_model: dict[str, dict[str, dict]] = {"v32": {}, "q3": {}}
    for cell in data["cells"]:
        by_model[cell["model"]][cell["architecture"]] = cell

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    # Plot each model's frontier
    for model, marker_filled in [("v32", True), ("q3", False)]:
        xs, ys = [], []
        for arch in ARCH_ORDER:
            c = by_model[model][arch]
            x = c["usd_per_task"] * 100  # cents
            y = c[ARCH_METRIC[arch]] * 100
            xs.append(x); ys.append(y)

        # Dashed frontier line for each model
        ax.plot(xs, ys, "--", color="0.6" if model == "v32" else "0.75",
                linewidth=1.0, alpha=0.7, zorder=1)

        # Markers + labels
        for i, arch in enumerate(ARCH_ORDER):
            color = ARCH_COLOR[arch]
            size = 140 if (model == "v32" and arch == "orchestrator") else 90
            if marker_filled:
                ax.scatter(xs[i], ys[i], s=size, color=color,
                           edgecolor="black", linewidth=0.7,
                           marker="o", zorder=4)
            else:
                ax.scatter(xs[i], ys[i], s=size, facecolor="white",
                           edgecolor=color, linewidth=1.8,
                           marker="o", zorder=4)

    # Per-architecture labels: the model that's HIGHER on the y-axis at this
    # architecture gets its label above its own marker; the lower model gets
    # its label below. This avoids collisions when the two markers are close
    # in cost (e.g., baseline cluster at ~0.15¢, pipeline cluster at ~15¢).
    for arch in ARCH_ORDER:
        v = by_model["v32"][arch]
        q = by_model["q3"][arch]
        vy = v[ARCH_METRIC[arch]] * 100
        vx = v["usd_per_task"] * 100
        qy = q[ARCH_METRIC[arch]] * 100
        qx = q["usd_per_task"] * 100

        # The label for the HIGHER model goes above-and-LEFT (so it doesn't
        # collide with the next-architecture marker, which is up and to the
        # right on a cost-vs-accuracy plot). The lower model's label goes
        # below-and-right (clear empty space).
        v_is_higher = vy >= qy
        v_offset = (-8, 6) if v_is_higher else (8, -6)
        q_offset = (8, -6) if v_is_higher else (-8, 6)
        # Baseline points sit at the leftmost edge — above-left would fall
        # off the panel. Nudge the higher model's label up-and-right of marker.
        # Baseline labels are forced ha="center" so the x-offset is a true
        # pixel nudge of the label's center (not an anchor flip).
        baseline_center = arch == "baseline"
        if baseline_center:
            if v_is_higher:
                v_offset = (3, 14)
            else:
                q_offset = (3, 14)

        def _ha(off, center=False):
            if center:
                return "center"
            return "center" if off[0] == 0 else ("right" if off[0] < 0 else "left")
        ax.annotate(f"{ARCH_LABEL[arch]} (V3.2)\n{vy:.1f}%",
                    xy=(vx, vy), xytext=v_offset,
                    textcoords="offset points",
                    fontsize=8.5, color=ARCH_COLOR[arch], fontweight=500,
                    ha=_ha(v_offset, center=baseline_center and v_is_higher),
                    va="bottom" if v_offset[1] > 0 else "top")
        ax.annotate(f"Qwen3: {qy:.1f}%", xy=(qx, qy), xytext=q_offset,
                    textcoords="offset points",
                    fontsize=8.5, color=ARCH_COLOR[arch], fontweight=400,
                    ha=_ha(q_offset, center=baseline_center and not v_is_higher),
                    va="bottom" if q_offset[1] > 0 else "top",
                    fontstyle="italic")

    ax.set_xscale("log")
    ax.set_xlabel("Cost per task (log scale)")
    ax.set_ylabel("Accuracy on ARC-AGI-1 eval, n=99 matched subset (%)")
    ax.set_title(
        "Cross-model Pareto: DeepSeek V3.2 vs Qwen3-235B, n = 99",
        fontsize=12.5, fontweight=600, pad=12,
    )
    ax.set_ylim(0, 80)
    ax.set_xlim(0.08, 60)
    major = [0.1, 1, 10]
    minor = [0.2, 0.5, 2, 5, 20, 50]
    ax.set_xticks(major)
    ax.set_xticks(minor, minor=True)
    ax.set_xticklabels(["0.1¢", "1¢", "10¢"])
    ax.set_xticklabels(["0.2¢", "0.5¢", "2¢", "5¢", "20¢", "50¢"], minor=True)
    ax.tick_params(axis="x", which="minor", labelsize=7.5, colors="0.45")

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", color="0.3",
               markersize=8, markeredgecolor="black", markeredgewidth=0.6,
               label="DeepSeek V3.2 (filled)"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white",
               color="white", markeredgecolor="0.3", markeredgewidth=1.5,
               markersize=8, label="Qwen3-235B (hollow)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9, frameon=False)

    # Methodology footnote
    fig.text(0.02, -0.02,
             "Pipeline/Orchestrator: M=3 for parity with Q3's ensemble. "
             "All cost values directly measured (V3.2 CoT uses the same fully-measured "
             "run as the headline DeepSeek Pareto).",
             fontsize=7, color="0.4", ha="left", va="top", style="italic")

    paths = save_paper(fig, "cross_model_pareto")
    for p in paths:
        print(f"saved: {p}")


if __name__ == "__main__":
    main()
