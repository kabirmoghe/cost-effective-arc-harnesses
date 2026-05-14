"""
backfill.py

Walk src/output/pipeline (training) and src/output/pipeline_eval (evaluation),
load every pattern_explorer / transformation_definer JSON, and populate the
evals / explorers / definers tables.

Usage:
    python backfill.py            # apply
    python backfill.py --dry-run  # report what would be inserted, no writes
"""

import argparse
import json
import os
import re
import uuid
from collections import defaultdict
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import Json, execute_values

load_dotenv()

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://kabirmoghe:postgres@localhost:5433/arc_evaluation",
)

SRC_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRS = {
    "training": SRC_ROOT / "output" / "pipeline",
    "evaluation": SRC_ROOT / "output" / "pipeline_eval",
}

EXPLORER_RE = re.compile(
    r"^pattern_explorer_([0-9a-fA-F]+)_(\d{8}-\d{6})_(\d+)\.json$"
)
DEFINER_RE = re.compile(
    r"^transformation_definer_([0-9a-fA-F]+)_(\d{8}-\d{6})\.json$"
)

SYSTEM = "pipeline"


def collect_files():
    """Return (explorers, definers) lists of dicts."""
    explorers = []
    definers = []
    skipped = []

    for dataset, root in OUTPUT_DIRS.items():
        if not root.exists():
            print(f"  warning: {root} does not exist, skipping")
            continue
        for task_dir in sorted(root.iterdir()):
            if not task_dir.is_dir():
                continue
            for f in sorted(task_dir.iterdir()):
                if not f.name.endswith(".json"):
                    continue
                m = EXPLORER_RE.match(f.name)
                if m:
                    explorers.append({
                        "dataset": dataset,
                        "task_id": m.group(1),
                        "run_id": m.group(2),
                        "agent_idx": int(m.group(3)),
                        "path": f,
                    })
                    continue
                m = DEFINER_RE.match(f.name)
                if m:
                    definers.append({
                        "dataset": dataset,
                        "task_id": m.group(1),
                        "run_id": m.group(2),
                        "path": f,
                    })
                    continue
                skipped.append(f)

    return explorers, definers, skipped


def aggregate_eval_metrics(explorers, definers):
    """Return {(dataset, run_id): metrics_dict}."""
    runs = defaultdict(lambda: {
        "task_ids": set(),
        "num_explorer_rows": 0,
        "num_definer_rows": 0,
        "correct_definers": 0,
        "test_pairs_total": 0,
        "test_pairs_correct": 0,
        "explorer_prompt_tokens": 0,
        "explorer_completion_tokens": 0,
        "definer_prompt_tokens": 0,
        "definer_completion_tokens": 0,
    })

    for e in explorers:
        key = (e["dataset"], e["run_id"])
        d = json.loads(e["path"].read_text())
        runs[key]["task_ids"].add(e["task_id"])
        runs[key]["num_explorer_rows"] += 1
        usage = d.get("usage") or {}
        runs[key]["explorer_prompt_tokens"] += usage.get("prompt_tokens", 0)
        runs[key]["explorer_completion_tokens"] += usage.get("completion_tokens", 0)

    for df in definers:
        key = (df["dataset"], df["run_id"])
        d = json.loads(df["path"].read_text())
        runs[key]["task_ids"].add(df["task_id"])
        runs[key]["num_definer_rows"] += 1
        if d.get("correct"):
            runs[key]["correct_definers"] += 1
        for tr in (d.get("test_results") or []):
            runs[key]["test_pairs_total"] += 1
            if tr.get("correct"):
                runs[key]["test_pairs_correct"] += 1
        usage = d.get("usage") or {}
        runs[key]["definer_prompt_tokens"] += usage.get("prompt_tokens", 0)
        runs[key]["definer_completion_tokens"] += usage.get("completion_tokens", 0)

    out = {}
    for key, m in runs.items():
        num_tasks = len(m["task_ids"])
        out[key] = {
            "num_tasks": num_tasks,
            "num_explorer_rows": m["num_explorer_rows"],
            "num_definer_rows": m["num_definer_rows"],
            "correct_definers": m["correct_definers"],
            "test_pairs_total": m["test_pairs_total"],
            "test_pairs_correct": m["test_pairs_correct"],
            "accuracy_file_level": (m["correct_definers"] / num_tasks) if num_tasks else None,
            "accuracy_test_pair_level": (m["test_pairs_correct"] / m["test_pairs_total"]) if m["test_pairs_total"] else None,
            "tokens": {
                "explorer": {
                    "prompt": m["explorer_prompt_tokens"],
                    "completion": m["explorer_completion_tokens"],
                },
                "definer": {
                    "prompt": m["definer_prompt_tokens"],
                    "completion": m["definer_completion_tokens"],
                },
                "total_prompt": m["explorer_prompt_tokens"] + m["definer_prompt_tokens"],
                "total_completion": m["explorer_completion_tokens"] + m["definer_completion_tokens"],
            },
        }
    return out


def backfill(dry_run: bool):
    explorers, definers, skipped = collect_files()

    print(f"Found {len(explorers)} explorer files, {len(definers)} definer files")
    if skipped:
        print(f"Skipped {len(skipped)} non-matching .json files (e.g. *_test.json):")
        for s in skipped[:5]:
            print(f"  - {s}")
        if len(skipped) > 5:
            print(f"  ... and {len(skipped) - 5} more")

    eval_metrics = aggregate_eval_metrics(explorers, definers)
    print(f"Distinct (dataset, run_id) eval rows: {len(eval_metrics)}")

    if dry_run:
        print("\n--- DRY RUN ---")
        for (dataset, run_id), m in sorted(eval_metrics.items()):
            print(f"  {dataset:10} {run_id}  tasks={m['num_tasks']:>3}  "
                  f"correct={m['correct_definers']:>3}  acc={m['accuracy']}")
        return

    conn = psycopg2.connect(DB_URL)
    try:
        with conn, conn.cursor() as cur:
            # Insert evals
            evals_rows = [
                (run_id, SYSTEM, dataset, Json(metrics))
                for (dataset, run_id), metrics in eval_metrics.items()
            ]
            execute_values(
                cur,
                "INSERT INTO evals (run_id, system, dataset, data) VALUES %s "
                "ON CONFLICT (run_id) DO NOTHING",
                evals_rows,
            )
            print(f"Inserted/skipped {len(evals_rows)} evals rows")

            # Insert explorers, capture UUIDs by (run_id, task_id, agent_idx)
            explorer_uuids = {}
            explorer_rows = []
            for e in explorers:
                payload = json.loads(e["path"].read_text())
                aid = uuid.uuid4()
                explorer_uuids[(e["run_id"], e["task_id"], e["agent_idx"])] = aid
                agent = f"pattern_explorer_{e['agent_idx']}"
                output = {
                    "patterns": payload.get("patterns"),
                    "synthesis": payload.get("synthesis"),
                    "trace": payload.get("trace"),
                }
                metadata = {
                    "model": payload.get("model"),
                    "provider": payload.get("provider"),
                    "created_at": payload.get("created_at"),
                    "usage": payload.get("usage"),
                    "source_file": str(e["path"].relative_to(SRC_ROOT)),
                }
                explorer_rows.append(
                    (str(aid), e["run_id"], e["task_id"], agent, Json(output), Json(metadata))
                )

            execute_values(
                cur,
                "INSERT INTO explorers (agent_id, run_id, task_id, agent, output, metadata) "
                "VALUES %s ON CONFLICT (run_id, task_id, agent) DO NOTHING",
                explorer_rows,
            )
            print(f"Inserted/skipped {len(explorer_rows)} explorer rows")

            # Insert definers with parent_explorer_ids
            definer_rows = []
            for df in definers:
                payload = json.loads(df["path"].read_text())
                aid = uuid.uuid4()
                parents = [
                    str(explorer_uuids[k])
                    for k in explorer_uuids
                    if k[0] == df["run_id"] and k[1] == df["task_id"]
                ]
                agent = "transformation_definer"
                output = {
                    "code": payload.get("code"),
                    "reasoning": payload.get("reasoning"),
                    "transformation_summary": payload.get("transformation_summary"),
                    "trace": payload.get("trace"),
                    "test_results": payload.get("test_results"),
                    "correct": payload.get("correct"),
                    "success": payload.get("success"),
                    "num_correct": payload.get("num_correct"),
                    "final_error": payload.get("final_error"),
                }
                metadata = {
                    "model": payload.get("model"),
                    "provider": payload.get("provider"),
                    "created_at": payload.get("created_at"),
                    "usage": payload.get("usage"),
                    "repair_attempts": payload.get("repair_attempts"),
                    "max_repairs": payload.get("max_repairs"),
                    "source_file": str(df["path"].relative_to(SRC_ROOT)),
                }
                definer_rows.append(
                    (str(aid), df["run_id"], df["task_id"], agent,
                     parents, Json(output), Json(metadata))
                )

            execute_values(
                cur,
                "INSERT INTO definers "
                "(agent_id, run_id, task_id, agent, parent_explorer_ids, output, metadata) "
                "VALUES %s ON CONFLICT (run_id, task_id, agent) DO NOTHING",
                definer_rows,
                template="(%s, %s, %s, %s, %s::uuid[], %s, %s)",
            )
            print(f"Inserted/skipped {len(definer_rows)} definer rows")
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    backfill(args.dry_run)
