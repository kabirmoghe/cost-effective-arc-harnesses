"""Bootstrap confidence intervals on per-task accuracy.

Pure-stdlib resampling. For binary 0/1 outcomes (correct/incorrect) on N tasks,
draws B resamples of N tasks with replacement and reports the requested
percentile interval on the mean.
"""

from __future__ import annotations

import random
from statistics import mean


def bootstrap_ci(
    values: list[int | float],
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (point_estimate, lo, hi) for the mean of `values`.

    `alpha=0.05` → 95% CI (2.5th and 97.5th percentiles).
    """
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    point = mean(values)
    means = []
    for _ in range(n_resamples):
        sample_mean = sum(rng.choice(values) for _ in range(n)) / n
        means.append(sample_mean)
    means.sort()
    lo_idx = int(n_resamples * (alpha / 2))
    hi_idx = int(n_resamples * (1 - alpha / 2)) - 1
    return point, means[lo_idx], means[hi_idx]
