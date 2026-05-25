"""Track A preview: bootstrap CIs + paired deltas + Pareto scaffold on the
stored V3.2 May data.

Runs A1 (bootstrap CIs), A2 (paired bootstrap), A3 (Pareto scaffold) over the
existing canonical V3.2 runs in the DB. Pre-headline V3.2/AtlasCloud B4 lands;
results here will be rerun on that data once B4 finishes.

Usage:
    uv run python -m stats.preview                 # print to stdout
    uv run python -m stats.preview --json out.json # also save
"""

from __future__ import annotations

import argparse
import json

from stats.bootstrap import bootstrap_ci
from stats.load import load_baseline_results, load_eval_meta, load_pipeline_pass_at_k
from stats.paired import paired_bootstrap_ci
from stats.pareto import pareto_point

# Canonical historical V3.2 runs (eval split).
RUNS = {
    "baseline": "019e1ac3-28aa-72c7-8eab-39ff23ef3da7",   # April first-party V3.2 baseline eval (14.5%)
    "CoT": "019e1ac3-28ba-7700-a193-a6ba7ff3239a",        # April first-party V3.2 CoT eval (15.5%)
    "arc_base": "019e2c5b-77a5-7beb-b79c-f88c99646568",   # May Friendli V3.2 arc_base eval (4.25%)
    "pipeline_M1": "019e289c-95d2-77ce-af93-46240142069f",  # May Friendli V3.2 M=1 control (40.6%)
    "pipeline_M5": "019e28b9-6a90-7242-89a7-91a4e7680fb8",  # May Friendli V3.2 M=5 (53.5%)
}


def _fmt_ci(point: float, lo: float, hi: float) -> str:
    return f"{point*100:5.2f}%  [{lo*100:5.2f}%, {hi*100:5.2f}%]"


def load_all() -> dict[str, dict[str, int]]:
    """Returns {name: {task_id: 0|1}} for each canonical run."""
    out: dict[str, dict[str, int]] = {}
    # Baselines: per-task = any test pair correct
    for name in ("baseline", "CoT", "arc_base"):
        per_task, _ = load_baseline_results(RUNS[name])
        out[name] = per_task
    # Pipeline: pass@1 and pass@2 reconstructions
    out["pipeline_M1_pass@1"] = load_pipeline_pass_at_k(RUNS["pipeline_M1"], 1)
    out["pipeline_M5_pass@1"] = load_pipeline_pass_at_k(RUNS["pipeline_M5"], 1)
    out["pipeline_M5_pass@2"] = load_pipeline_pass_at_k(RUNS["pipeline_M5"], 2)
    return out


def run_a1(arms: dict[str, dict[str, int]]) -> list[dict]:
    rows = []
    for name, per_task in arms.items():
        vals = list(per_task.values())
        p, lo, hi = bootstrap_ci(vals)
        rows.append({"arm": name, "n": len(vals), "point": p, "lo": lo, "hi": hi})
    return rows


def run_a2(arms: dict[str, dict[str, int]]) -> list[dict]:
    # Matched-task deltas worth reporting.
    pairs = [
        ("baseline", "CoT"),
        ("CoT", "pipeline_M1_pass@1"),
        ("pipeline_M1_pass@1", "pipeline_M5_pass@1"),
        ("pipeline_M5_pass@1", "pipeline_M5_pass@2"),
        ("baseline", "pipeline_M5_pass@2"),
        ("arc_base", "baseline"),
    ]
    rows = []
    for a, b in pairs:
        if a not in arms or b not in arms:
            continue
        p, lo, hi, n = paired_bootstrap_ci(arms[a], arms[b])
        rows.append({"a": a, "b": b, "delta": p, "lo": lo, "hi": hi, "n_matched": n})
    return rows


def _explorer_tokens_for(d: dict) -> dict:
    """Pull explorer-stage tokens from the source explorer run if this is a
    definer-only re-run (explorer prompt=0)."""
    src = d.get("source_explorer_run_id")
    if not src:
        return {"total_prompt": 0, "total_completion": 0}
    src_meta = load_eval_meta(src)
    src_tokens = (src_meta["data"].get("tokens") or {}).get("by_agent", {}).get("explorer", {})
    return {
        "total_prompt": src_tokens.get("prompt", 0),
        "total_completion": src_tokens.get("completion", 0),
    }


def run_a3(arms: dict[str, dict[str, int]]) -> list[dict]:
    points = []
    for name, run_id in RUNS.items():
        meta = load_eval_meta(run_id)
        d = meta["data"]
        tokens = dict(d.get("tokens") or d.get("token_usage") or {})
        # Fold in explorer-stage cost for definer-only re-runs.
        if "source_explorer_run_id" in d and d["source_explorer_run_id"]:
            exp = _explorer_tokens_for(d)
            tokens["total_prompt"] = tokens.get("total_prompt", 0) + exp["total_prompt"]
            tokens["total_completion"] = tokens.get("total_completion", 0) + exp["total_completion"]
        num_tasks = d.get("num_tasks") or len({r["task_id"] for r in d.get("results", [])})
        # Use the canonical headline accuracy from the eval record.
        if name == "pipeline_M5":
            acc_keys = ("accuracy_pass_at_2", "accuracy")
        elif name == "pipeline_M1":
            acc_keys = ("accuracy_pass_at_1", "accuracy")
        else:
            acc_keys = ("accuracy",)
        acc = next((d[k] for k in acc_keys if k in d and d[k] is not None), None)
        points.append(pareto_point(name, acc, tokens, num_tasks))
    points.sort(key=lambda p: p["cost_per_task"])
    return points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default=None, help="optional output path")
    args = parser.parse_args()

    arms = load_all()

    a1 = run_a1(arms)
    print("=" * 72)
    print("A1. Bootstrap 95% CI on accuracy (per-task vectors)")
    print("=" * 72)
    for r in a1:
        print(f"  {r['arm']:28s} n={r['n']:4d}  {_fmt_ci(r['point'], r['lo'], r['hi'])}")

    a2 = run_a2(arms)
    print()
    print("=" * 72)
    print("A2. Paired bootstrap 95% CI on accuracy delta (mean(b) - mean(a))")
    print("=" * 72)
    for r in a2:
        sig = "*" if (r["lo"] > 0 or r["hi"] < 0) else " "
        print(
            f"  {sig} {r['a']:24s} -> {r['b']:24s}  n={r['n_matched']:4d}  "
            f"Δ={r['delta']*100:+6.2f}pp  [{r['lo']*100:+6.2f}, {r['hi']*100:+6.2f}]"
        )
    print("  (* = 95% CI excludes 0)")

    a3 = run_a3(arms)
    print()
    print("=" * 72)
    print("A3. Pareto scaffold (cost/task in USD vs accuracy)")
    print("=" * 72)
    for p in a3:
        print(f"  {p['label']:14s}  acc={p['accuracy']*100:5.2f}%  $/task={p['cost_per_task']:7.4f}  n_tasks={p['num_tasks']}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"a1": a1, "a2": a2, "a3": a3}, f, indent=2, default=str)
        print(f"\nSaved to {args.json}")


if __name__ == "__main__":
    main()
