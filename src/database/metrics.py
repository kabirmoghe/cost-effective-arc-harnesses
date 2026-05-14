"""Aggregate metrics computed from explorers/definers rows or baseline results.

Two flavors:
- compute_pipeline_metrics(cur, run_id): queries the DB. Supports M>1 definers
  per task via pass@1 (transformation_definer_0 only) and pass@M (any definer)
  accuracy. The top-level `accuracy` field is pass@1 for back-compat.
- compute_baseline_metrics(results, token_usage): pure-Python over a baseline /
  CoT results list. File-level (task-level) accuracy is the top-level metric.
"""

from typing import Any


def compute_pipeline_metrics(cur, run_id: str) -> dict[str, Any]:
    cur.execute(
        "SELECT COUNT(DISTINCT task_id) FROM definers WHERE run_id = %s",
        (run_id,),
    )
    num_tasks = cur.fetchone()[0]

    cur.execute(
        """
        SELECT COUNT(*),
               COALESCE(SUM((metadata->'usage'->>'prompt_tokens')::int), 0),
               COALESCE(SUM((metadata->'usage'->>'completion_tokens')::int), 0)
        FROM definers WHERE run_id = %s
        """,
        (run_id,),
    )
    num_def, def_p, def_c = cur.fetchone()

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
        SELECT COUNT(DISTINCT task_id) FROM definers
        WHERE run_id = %s AND agent = 'transformation_definer_0'
          AND (output->>'correct')::bool
        """,
        (run_id,),
    )
    pass_at_1 = cur.fetchone()[0]

    cur.execute(
        """
        SELECT COUNT(DISTINCT task_id) FROM definers
        WHERE run_id = %s AND (output->>'correct')::bool
        """,
        (run_id,),
    )
    pass_at_m = cur.fetchone()[0]

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
        WHERE d.run_id = %s AND d.agent = 'transformation_definer_0'
        """,
        (run_id,),
    )
    tp_correct, tp_total = cur.fetchone()

    acc_p1 = (pass_at_1 / num_tasks) if num_tasks else None
    acc_pm = (pass_at_m / num_tasks) if num_tasks else None

    return {
        "num_tasks": num_tasks,
        "num_explorer_rows": num_expl,
        "num_definer_rows": num_def,
        "pass_at_1_tasks": pass_at_1,
        "pass_at_m_tasks": pass_at_m,
        "accuracy_pass_at_1": acc_p1,
        "accuracy_pass_at_m": acc_pm,
        "accuracy": acc_p1,
        "test_pairs_total": tp_total,
        "test_pairs_correct": tp_correct,
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


def compute_baseline_metrics(
    results: list[dict], token_usage: dict | None = None
) -> dict[str, Any]:
    tokens = token_usage or {}
    total_pairs = len(results)
    correct_pairs = sum(1 for r in results if r.get("correct"))
    errors = sum(1 for r in results if r.get("error"))
    distinct_tasks = len({r["task_id"] for r in results})

    file_level: dict[str, bool] = {}
    for r in results:
        tid = r["task_id"]
        file_level[tid] = file_level.get(tid, True) and bool(r.get("correct"))
    correct_files = sum(1 for v in file_level.values() if v)

    acc_file = (correct_files / distinct_tasks) if distinct_tasks else None
    return {
        "num_tasks": distinct_tasks,
        "test_pairs_total": total_pairs,
        "test_pairs_correct": correct_pairs,
        "errors": errors,
        "correct_file_level": correct_files,
        "accuracy_file_level": acc_file,
        "accuracy_test_pair_level": (correct_pairs / total_pairs) if total_pairs else None,
        "accuracy": acc_file,
        "tokens": {
            "total_prompt": tokens.get("prompt_tokens", 0),
            "total_completion": tokens.get("completion_tokens", 0),
            "total": tokens.get("total_tokens", 0),
        },
        "results": results,
    }
