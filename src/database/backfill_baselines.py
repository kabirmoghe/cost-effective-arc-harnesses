"""
backfill_baselines.py

Reads the baseline_* and CoT_* metrics_full.json files in src/metrics/ and
inserts one evals row per (system, dataset) with the full results in `data`.

Usage:
    python backfill_baselines.py             # apply
    python backfill_baselines.py --dry-run
"""

import argparse
import json
import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import Json

from consolidate import uuid7

load_dotenv()

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://kabirmoghe:postgres@localhost:5433/arc_evaluation",
)

METRICS_DIR = Path(__file__).resolve().parents[1] / "metrics"

SOURCES = [
    ("baseline", "training",   "baseline_train_metrics_full.json"),
    ("baseline", "evaluation", "baseline_eval_metrics_full.json"),
    ("CoT",      "training",   "CoT_train_metrics_full.json"),
    ("CoT",      "evaluation", "CoT_eval_metrics_full.json"),
]


def build_data(payload: dict) -> dict:
    results = payload.get("results") or []
    total_pairs = len(results)
    correct_pairs = sum(1 for r in results if r.get("correct"))
    errors = sum(1 for r in results if r.get("error"))
    distinct_tasks = len({r["task_id"] for r in results})
    file_level = {}
    for r in results:
        tid = r["task_id"]
        file_level.setdefault(tid, True)
        file_level[tid] = file_level[tid] and bool(r.get("correct"))
    correct_files = sum(1 for v in file_level.values() if v)
    tokens = payload.get("token_usage") or {}
    return {
        "num_tasks": distinct_tasks,
        "test_pairs_total": total_pairs,
        "test_pairs_correct": correct_pairs,
        "errors": errors,
        "correct_file_level": correct_files,
        "accuracy_file_level": (correct_files / distinct_tasks) if distinct_tasks else None,
        "accuracy_test_pair_level": (correct_pairs / total_pairs) if total_pairs else None,
        "tokens": {
            "total_prompt": tokens.get("prompt_tokens", 0),
            "total_completion": tokens.get("completion_tokens", 0),
            "total": tokens.get("total_tokens", 0),
        },
        "results": results,
    }


def main(dry_run: bool):
    conn = psycopg2.connect(DB_URL)
    try:
        with conn, conn.cursor() as cur:
            for system, dataset, filename in SOURCES:
                path = METRICS_DIR / filename
                if not path.exists():
                    print(f"  WARN: {path} missing, skipping")
                    continue
                payload = json.loads(path.read_text())
                data = build_data(payload)
                new_run_id = str(uuid7())
                print(f"\n[{system}/{dataset}] {new_run_id}")
                print(f"  tasks={data['num_tasks']}  test_pairs={data['test_pairs_total']}  "
                      f"correct_pairs={data['test_pairs_correct']}  correct_files={data['correct_file_level']}  "
                      f"acc_file={data['accuracy_file_level']:.4f}  "
                      f"acc_pair={data['accuracy_test_pair_level']:.4f}  "
                      f"tokens_total={data['tokens']['total']:,}")
                if dry_run:
                    continue
                cur.execute(
                    "INSERT INTO evals (run_id, system, dataset, data) VALUES (%s, %s, %s, %s)",
                    (new_run_id, system, dataset, Json(data)),
                )
            if dry_run:
                print("\n--- DRY RUN ---")
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(args.dry_run)
