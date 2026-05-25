"""Paired bootstrap on accuracy deltas between two runs over a matched task set.

Both inputs are dicts {task_id: 0|1}. Tasks present in both are matched; the
delta is mean(b) - mean(a) and the bootstrap resamples task IDs (not
independent draws), so within-task correlation is preserved.
"""

from __future__ import annotations

import random


def paired_bootstrap_ci(
    a: dict[str, int | float],
    b: dict[str, int | float],
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float, int]:
    """Return (point_delta, lo, hi, n_matched) for mean(b) - mean(a)."""
    matched = sorted(set(a) & set(b))
    if not matched:
        return 0.0, 0.0, 0.0, 0
    a_vals = [a[t] for t in matched]
    b_vals = [b[t] for t in matched]
    n = len(matched)
    point = sum(b_vals) / n - sum(a_vals) / n
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_resamples):
        idxs = [rng.randrange(n) for _ in range(n)]
        deltas.append(
            sum(b_vals[i] for i in idxs) / n - sum(a_vals[i] for i in idxs) / n
        )
    deltas.sort()
    lo_idx = int(n_resamples * (alpha / 2))
    hi_idx = int(n_resamples * (1 - alpha / 2)) - 1
    return point, deltas[lo_idx], deltas[hi_idx], n
