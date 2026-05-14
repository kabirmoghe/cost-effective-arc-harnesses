"""
consolidate.py

Merges the partial per-timestamp run rows in `evals` into one canonical run per
dataset (training, evaluation). Produces UUID v7 run_ids, repoints all
explorers/definers, recomputes aggregate metrics, and deletes the old eval rows.

Idempotent only in the loose sense: running it twice will mint *new* canonical
UUIDs each time. Don't run twice on already-consolidated data.

Usage:
    python consolidate.py             # apply
    python consolidate.py --dry-run   # report, no writes
"""

import argparse
import json
import os
import secrets
import time
import uuid

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import Json

load_dotenv()

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://kabirmoghe:postgres@localhost:5433/arc_evaluation",
)



def uuid7() -> uuid.UUID:
    """Minimal RFC 9562 UUID v7 generator (Python stdlib gains this in 3.14)."""
    ts_ms = int(time.time() * 1000) & ((1 << 48) - 1)
    rand_a = secrets.randbits(12)
    rand_b = secrets.randbits(62)
    n = ts_ms << 80
    n |= 0x7 << 76
    n |= rand_a << 64
    n |= 0x2 << 62
    n |= rand_b
    return uuid.UUID(int=n)


def compute_metrics(cur, run_id: str) -> dict:
    cur.execute(
        "SELECT COUNT(DISTINCT task_id) FROM definers WHERE run_id = %s",
        (run_id,),
    )
    num_tasks = cur.fetchone()[0]

    cur.execute(
        """
        SELECT COUNT(*),
               COUNT(*) FILTER (WHERE (output->>'correct')::bool),
               COALESCE(SUM((metadata->'usage'->>'prompt_tokens')::int), 0),
               COALESCE(SUM((metadata->'usage'->>'completion_tokens')::int), 0)
        FROM definers WHERE run_id = %s
        """,
        (run_id,),
    )
    num_def, correct_def, def_p, def_c = cur.fetchone()

    cur.execute(
        """
        SELECT COUNT(*),
               COALESCE(SUM((metadata->'usage'->>'prompt_tokens')::int), 0),
               COALESCE(SUM((metadata->'usage'->>'completion_tokens')::int), 0)
        FROM explorers WHERE run_id = %s
        """,
        (run_id,),
    )
    num_expl, expl_p, expl_c = cur.fetchone()

    cur.execute(
        """
        SELECT COUNT(*) FILTER (WHERE (tr->>'correct')::bool),
               COUNT(*)
        FROM definers d,
             jsonb_array_elements(
               CASE WHEN jsonb_typeof(d.output->'test_results') = 'array'
                    THEN d.output->'test_results'
                    ELSE '[]'::jsonb END
             ) tr
        WHERE d.run_id = %s
        """,
        (run_id,),
    )
    tp_correct, tp_total = cur.fetchone()

    return {
        "num_tasks": num_tasks,
        "num_explorer_rows": num_expl,
        "num_definer_rows": num_def,
        "correct_definers": correct_def,
        "test_pairs_total": tp_total,
        "test_pairs_correct": tp_correct,
        "accuracy_file_level": (correct_def / num_tasks) if num_tasks else None,
        "accuracy_test_pair_level": (tp_correct / tp_total) if tp_total else None,
        "tokens": {
            "total_prompt": expl_p + def_p,
            "total_completion": expl_c + def_c,
            "total": expl_p + def_p + expl_c + def_c,
            "by_agent": {
                "explorer": {"prompt": expl_p, "completion": expl_c},
                "definer": {"prompt": def_p, "completion": def_c},
            },
        },
    }


def main(dry_run: bool):
    conn = psycopg2.connect(DB_URL)
    try:
        with conn, conn.cursor() as cur:
            print("=== Pre-state ===")
            cur.execute("SELECT dataset, COUNT(*) FROM evals GROUP BY dataset ORDER BY dataset")
            for row in cur.fetchall():
                print(f"  evals rows for {row[0]}: {row[1]}")

            # 1. Cleanup: for each task, keep only rows from the latest run that
            #    has a definer for it. Drop everything else (older duplicate
            #    attempts + orphan explorers from runs whose definer didn't
            #    complete).
            print("\n=== Cleanup: drop older / orphan rows ===")
            for dataset in ("evaluation", "training"):
                # canonical run per task = max run_id among runs with a definer
                # (run_ids are YYYYMMDD-HHMMSS strings, so lex max = latest)
                cur.execute(
                    """
                    SELECT d.task_id, MAX(d.run_id)
                    FROM definers d JOIN evals e ON d.run_id = e.run_id
                    WHERE e.dataset = %s
                    GROUP BY d.task_id
                    """,
                    (dataset,),
                )
                canonical = cur.fetchall()

                # Drop non-canonical (task, run) rows
                drop_expl = drop_def = 0
                for task_id, canon_run in canonical:
                    if dry_run:
                        cur.execute(
                            """
                            SELECT COUNT(*) FROM explorers x JOIN evals e ON x.run_id=e.run_id
                            WHERE e.dataset=%s AND x.task_id=%s AND x.run_id != %s
                            """,
                            (dataset, task_id, canon_run),
                        )
                        drop_expl += cur.fetchone()[0]
                        cur.execute(
                            """
                            SELECT COUNT(*) FROM definers d JOIN evals e ON d.run_id=e.run_id
                            WHERE e.dataset=%s AND d.task_id=%s AND d.run_id != %s
                            """,
                            (dataset, task_id, canon_run),
                        )
                        drop_def += cur.fetchone()[0]
                    else:
                        cur.execute(
                            """
                            DELETE FROM explorers x USING evals e
                            WHERE x.run_id=e.run_id AND e.dataset=%s
                              AND x.task_id=%s AND x.run_id != %s
                            """,
                            (dataset, task_id, canon_run),
                        )
                        drop_expl += cur.rowcount
                        cur.execute(
                            """
                            DELETE FROM definers d USING evals e
                            WHERE d.run_id=e.run_id AND e.dataset=%s
                              AND d.task_id=%s AND d.run_id != %s
                            """,
                            (dataset, task_id, canon_run),
                        )
                        drop_def += cur.rowcount

                # Drop orphan explorers (tasks with no definer anywhere)
                orphan_q = """
                    FROM explorers x USING evals e
                    WHERE x.run_id=e.run_id AND e.dataset=%s
                      AND NOT EXISTS (
                        SELECT 1 FROM definers d JOIN evals e2 ON d.run_id=e2.run_id
                        WHERE d.task_id=x.task_id AND e2.dataset=%s
                      )
                """
                if dry_run:
                    cur.execute(
                        "SELECT COUNT(*) " + orphan_q.replace("FROM explorers x USING evals e",
                                                              "FROM explorers x JOIN evals e ON x.run_id=e.run_id"),
                        (dataset, dataset),
                    )
                    orphan = cur.fetchone()[0]
                else:
                    cur.execute("DELETE " + orphan_q, (dataset, dataset))
                    orphan = cur.rowcount

                print(f"  [{dataset}] dropped {drop_expl} non-canonical explorer rows, "
                      f"{drop_def} non-canonical definer rows, {orphan} orphan explorer rows")

            new_ids = {}
            for dataset in ("evaluation", "training"):
                cur.execute(
                    "SELECT run_id FROM evals WHERE dataset=%s ORDER BY run_id",
                    (dataset,),
                )
                old_run_ids = [r[0] for r in cur.fetchall()]
                new_run_id = str(uuid7())
                new_ids[dataset] = new_run_id

                print(f"\n[{dataset}] consolidating {len(old_run_ids)} runs -> {new_run_id}")
                print(f"  old run_ids: {old_run_ids}")

                if dry_run:
                    continue

                # Insert canonical eval row with placeholder data (we'll fill after repoint)
                cur.execute(
                    "INSERT INTO evals (run_id, system, dataset, data) "
                    "VALUES (%s, 'pipeline', %s, %s)",
                    (new_run_id, dataset, Json({})),
                )
                # Repoint children
                cur.execute(
                    "UPDATE explorers SET run_id=%s WHERE run_id = ANY(%s)",
                    (new_run_id, old_run_ids),
                )
                expl_moved = cur.rowcount
                cur.execute(
                    "UPDATE definers SET run_id=%s WHERE run_id = ANY(%s)",
                    (new_run_id, old_run_ids),
                )
                def_moved = cur.rowcount
                # Delete old eval rows
                cur.execute(
                    "DELETE FROM evals WHERE run_id = ANY(%s)",
                    (old_run_ids,),
                )
                print(f"  repointed {expl_moved} explorers, {def_moved} definers; "
                      f"deleted {len(old_run_ids)} old eval rows")

            if dry_run:
                print("\n--- DRY RUN: no changes applied ---")
                return

            # Recompute aggregated metrics for each new eval row
            print("\n=== Recomputing canonical eval metrics ===")
            for dataset, new_run_id in new_ids.items():
                metrics = compute_metrics(cur, new_run_id)
                cur.execute(
                    "UPDATE evals SET data=%s WHERE run_id=%s",
                    (Json(metrics), new_run_id),
                )
                print(f"\n[{dataset}] {new_run_id}")
                print(json.dumps(metrics, indent=2))
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(args.dry_run)
