"""Generate standardized metrics from pipeline output folder.

Reads transformation_definer JSON files and writes metrics files
matching the baseline/CoT schema.

Usage:
    uv run -m metrics.generate_pipeline_metrics --split training
    uv run -m metrics.generate_pipeline_metrics --split evaluation --output-dir output/pipeline_eval
"""

import argparse
import json
import glob
from pathlib import Path


def main(split: str, output_dir: str, metrics_dir: str = "metrics"):
    output_path = Path(output_dir)
    metrics_path = Path(metrics_dir)

    all_results = []
    token_map = {}

    for summary_file in sorted(output_path.glob("run_*_summary.json")):
        s = json.load(open(summary_file))
        for r in s["results"]:
            if not r.get("error"):
                token_map[r["task_id"]] = {
                    "prompt_tokens": r["prompt_tokens"],
                    "completion_tokens": r["completion_tokens"],
                }

    for f in sorted(glob.glob(str(output_path / "*/transformation_definer_*.json"))):
        d = json.load(open(f))
        task_id = d["task_id"]
        tokens = token_map.get(task_id, {"prompt_tokens": 0, "completion_tokens": 0})

        if "test_results" in d and d["test_results"]:
            for tr in d["test_results"]:
                all_results.append({
                    "task_id": task_id,
                    "test_index": tr["test_index"],
                    "correct": tr.get("correct", False) or False,
                    "error": tr.get("error"),
                    "prompt_tokens": tokens["prompt_tokens"] if tr["test_index"] == 0 else 0,
                    "completion_tokens": tokens["completion_tokens"] if tr["test_index"] == 0 else 0,
                })
        else:
            all_results.append({
                "task_id": task_id,
                "test_index": 0,
                "correct": d.get("correct", False) or False,
                "error": d.get("final_error"),
                "prompt_tokens": tokens["prompt_tokens"],
                "completion_tokens": tokens["completion_tokens"],
            })

    total_prompt = sum(r["prompt_tokens"] for r in all_results)
    total_completion = sum(r["completion_tokens"] for r in all_results)
    correct = sum(1 for r in all_results if r["correct"])
    errors = sum(1 for r in all_results if r["error"])
    total_tests = len(all_results)
    unique_tasks = len(set(r["task_id"] for r in all_results))

    summary = {
        "split": split,
        "model": "deepseek-chat",
        "approach": "pipeline",
        "total": total_tests,
        "total_tasks": unique_tasks,
        "correct": correct,
        "errors": errors,
        "accuracy": correct / total_tests if total_tests else 0,
        "token_usage": {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
        },
    }

    full = {**summary, "results": all_results}

    metrics_path.mkdir(exist_ok=True)
    label = "eval" if split == "evaluation" else "train"
    summary_path = metrics_path / f"pipeline_{label}_metrics.json"
    full_path = metrics_path / f"pipeline_{label}_metrics_full.json"

    summary_path.write_text(json.dumps(summary, indent=2))
    full_path.write_text(json.dumps(full, indent=2))

    print(f"Tasks: {unique_tasks}")
    print(f"Test pairs: {total_tests}")
    print(f"Correct: {correct}/{total_tests} ({correct/total_tests*100:.1f}%)")
    print(f"Errors: {errors}")
    print(f"Tokens: {total_prompt + total_completion:,}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {full_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="training", choices=["training", "evaluation"])
    parser.add_argument("--output-dir", default=None, help="Pipeline output directory")
    parser.add_argument("--metrics-dir", default="metrics")
    args = parser.parse_args()

    output_dir = args.output_dir or ("output/pipeline_eval" if args.split == "evaluation" else "output/pipeline")
    main(split=args.split, output_dir=output_dir, metrics_dir=args.metrics_dir)
