"""Slide 8: cost-vs-accuracy with Pipeline added to the external comparators.

Second beat of the reveal sequence (external → +Pipeline → +Orchestrator → full).
"""

from figures.slides._pareto_base import render_pareto


def main() -> None:
    p = render_pareto(
        "slide08_pipeline_added",
        include_external=True,
        include_ours=("pipeline",),
        title="Adding the Pipeline (V3.2) to the published landscape",
    )
    print(f"saved: {p}")


if __name__ == "__main__":
    main()
