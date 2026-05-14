"""Compute cost per task across approaches using published DeepSeek rates.

Reads metrics/*_metrics.json files, applies current DeepSeek list pricing,
and writes metrics/cost_per_task.json.

Update PRICING below if DeepSeek changes their rates.
Source: https://api-docs.deepseek.com/quick_start/pricing
"""

import json
from pathlib import Path

# Current DeepSeek list pricing (per 1M tokens), on-peak, cache-miss.
# deepseek-chat currently resolves to DeepSeek V3.2.
# Cache-hit input is ~$0.028/M (90% discount); off-peak is ~50% off.
PRICING = {
    "deepseek-chat": {
        "input_per_million": 0.28,
        "output_per_million": 0.42,
    },
}


def cost_for(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = PRICING[model]
    return (
        prompt_tokens / 1_000_000 * rates["input_per_million"]
        + completion_tokens / 1_000_000 * rates["output_per_million"]
    )


def main():
    metrics_dir = Path(__file__).parent
    rows = []

    # Dedupe: prefer summary file over *_full when both exist
    files = {}
    for f in sorted(metrics_dir.glob("*_metrics*.json")):
        if f.name == "cost_per_task.json":
            continue
        key = f.name.replace("_full.json", ".json")
        if key in files and not files[key].name.endswith("_full.json"):
            continue
        files[key] = f

    for f in sorted(files.values()):
        d = json.load(open(f))
        model = d["model"]
        usage = d["token_usage"]
        tasks = d.get("total_tasks") or d.get("total")
        total_cost = cost_for(model, usage["prompt_tokens"], usage["completion_tokens"])
        rows.append({
            "file": f.name,
            "split": d["split"],
            "approach": d.get("approach") or d.get("baseline"),
            "model": model,
            "tasks": tasks,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_cost_usd": round(total_cost, 4),
            "cost_per_task_usd": round(total_cost / tasks, 6) if tasks else None,
        })

    out = {
        "pricing_source": "https://api-docs.deepseek.com/quick_start/pricing",
        "pricing": PRICING,
        "notes": "On-peak, cache-miss list rates. Actual cost may be lower with cache hits / off-peak.",
        "rows": rows,
    }

    out_path = metrics_dir / "cost_per_task.json"
    out_path.write_text(json.dumps(out, indent=2))

    print(f"{'file':<40} {'tasks':>6} {'$/task':>10} {'total $':>10}")
    print("-" * 70)
    for r in rows:
        print(f"{r['file']:<40} {r['tasks']:>6} {r['cost_per_task_usd']:>10.4f} {r['total_cost_usd']:>10.2f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
