"""Slide 16: full Pareto with cost-multiplier callouts vs nearest comparators.

The annotations are speaker-aids — quantify "we are X× cheaper than Y at
comparable accuracy" without forcing the audience to do mental math.
"""

from figures.slides._pareto_base import render_pareto
from figures.slides._comparators import COMPARATORS

# Find anchor points programmatically so callouts stay attached if numbers shift
PANG = next(c for c in COMPARATORS if c["short"] == "Pang")
BERMAN24 = next(c for c in COMPARATORS if c["short"] == "Berman '24")
BERMAN25 = next(c for c in COMPARATORS if c["short"] == "Berman '25")


def main() -> None:
    # Orchestrator at ~67% / ~$0.62/task vs Pang at 77.1% / $2.56 (~4× cheaper)
    # Orchestrator at ~67% / ~$0.62/task vs Berman 2024 at 53.6% / $29 (~47× cheaper at higher acc)
    callouts = [
        {
            "text": "~4× cheaper than Pang (similar regime)",
            "anchor": (PANG["usd_per_task"] * 100, PANG["accuracy"]),
            "offset": (-130, 40),
        },
        {
            "text": "~47× cheaper than Berman '24,\nat +13.7pp higher accuracy",
            "anchor": (BERMAN24["usd_per_task"] * 100, BERMAN24["accuracy"]),
            "offset": (-180, -50),
        },
    ]
    p = render_pareto(
        "slide16_pareto_annotated",
        include_external=True,
        include_ours=("baseline", "cot", "pipeline", "orchestrator"),
        callouts=callouts,
        title="The Pareto-frontier story (with cost-multiplier comparisons)",
    )
    print(f"saved: {p}")


if __name__ == "__main__":
    main()
