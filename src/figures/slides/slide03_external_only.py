"""Slide 3: cost-vs-accuracy chart with only external published systems.

Sets the stage for the visual reveal across slides 8, 13, 16.
"""

from figures.slides._pareto_base import render_pareto


def main() -> None:
    p = render_pareto(
        "slide03_external_only",
        include_external=True,
        include_ours=(),
        title="Published ARC-AGI-1 systems: cost vs accuracy",
    )
    print(f"saved: {p}")


if __name__ == "__main__":
    main()
