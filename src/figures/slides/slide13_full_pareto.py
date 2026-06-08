"""Slide 13: full Pareto — all 4 of our architectures + external comparators.

This is the canonical chart from the paper, expanded with published systems for
the talk.
"""

from figures.slides._pareto_base import render_pareto


def main() -> None:
    p = render_pareto(
        "slide13_full_pareto",
        include_external=True,
        include_ours=("baseline", "cot", "pipeline", "orchestrator"),
        title="Cost vs accuracy: this work + published ARC-AGI-1 systems",
    )
    print(f"saved: {p}")


if __name__ == "__main__":
    main()
