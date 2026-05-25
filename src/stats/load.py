"""Load per-task pass/fail vectors from DB for any run.

For baseline/CoT/arc_base: pulls from `evals.data.results` (one row per test
pair), aggregates to task-level (any test pair passes → task passes), and also
returns the raw test-pair-level vector. For pipeline: reconstructs per-task
pass@k booleans from the `definers` table using the same selection logic as
`pipeline.selection`.
"""

from __future__ import annotations

import os
from collections import defaultdict

import psycopg2
from dotenv import load_dotenv

from pipeline.selection import candidate_from_output, select_pass_at_k

load_dotenv()

DEFAULT_DB_URL = "postgresql://kabirmoghe:postgres@localhost:5433/arc_evaluation"


def _conn():
    return psycopg2.connect(os.getenv("DATABASE_URL", DEFAULT_DB_URL))


def load_baseline_results(run_id: str) -> tuple[dict[str, int], dict[str, int]]:
    """Return (per_task_dict, per_pair_dict). Per-task: any-pair-correct."""
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT data FROM evals WHERE run_id::text = %s", (run_id,))
        d = cur.fetchone()[0]
    per_task: dict[str, list[int]] = defaultdict(list)
    per_pair: dict[str, int] = {}
    for r in d["results"]:
        tid = r["task_id"]
        idx = r.get("test_index", 0)
        ok = 1 if r["correct"] else 0
        per_task[tid].append(ok)
        per_pair[f"{tid}#{idx}"] = ok
    per_task_agg = {tid: int(any(vs)) for tid, vs in per_task.items()}
    return per_task_agg, per_pair


def load_pipeline_pass_at_k(run_id: str, k: int) -> dict[str, int]:
    """Reconstruct per-task pass@k for a pipeline run from `definers`."""
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT task_id, agent, output FROM definers WHERE run_id::text = %s",
            (run_id,),
        )
        rows = cur.fetchall()
    by_task: dict[str, list] = defaultdict(list)
    for task_id, agent, output in rows:
        try:
            idx = int(agent.rsplit("_", 1)[-1])
        except Exception:
            idx = 0
        by_task[task_id].append(candidate_from_output(idx, output))
    return {
        tid: int(select_pass_at_k(cands, k)["pass_at_k"])
        for tid, cands in by_task.items()
    }


def load_eval_meta(run_id: str) -> dict:
    """Return the full evals.data blob for a run."""
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT system, dataset, data FROM evals WHERE run_id::text = %s",
            (run_id,),
        )
        sys, ds, d = cur.fetchone()
    return {"system": sys, "dataset": ds, "data": d}
