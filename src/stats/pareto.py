"""Build (accuracy, $/task) Pareto points from DB runs.

Uses DeepSeek list pricing from `metrics.compute_costs.PRICING`. Pricing here
is approximate: April first-party V3.2 and May Friendli V3.2 hit different
real backends but list rates were similar; use a single rate for the preview
and switch to provider-specific rates once AtlasCloud numbers land.
"""

from __future__ import annotations

# DeepSeek V3.2 list pricing (cache-miss, on-peak). Source: api-docs.
PER_M_IN = 0.28
PER_M_OUT = 0.42


def cost_per_task(tokens: dict, num_tasks: int) -> float:
    if not num_tasks:
        return 0.0
    # Pipeline runs store {total_prompt, total_completion}; baseline runs store
    # {prompt_tokens, completion_tokens}. Try both.
    in_tok = tokens.get("total_prompt", tokens.get("prompt_tokens", 0))
    out_tok = tokens.get("total_completion", tokens.get("completion_tokens", 0))
    cost = in_tok / 1_000_000 * PER_M_IN + out_tok / 1_000_000 * PER_M_OUT
    return cost / num_tasks


def pareto_point(label: str, accuracy: float, tokens: dict, num_tasks: int) -> dict:
    return {
        "label": label,
        "accuracy": accuracy,
        "cost_per_task": cost_per_task(tokens, num_tasks),
        "num_tasks": num_tasks,
    }
