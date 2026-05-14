"""Aggregate metrics computed from explorers/definers rows or baseline results.

Two flavors:
- compute_pipeline_metrics(cur, run_id): queries the DB. Supports M>1 definers
  per task. Per task it runs the same pass@k selection as the pipeline runner
  (see pipeline.selection): drop non-executing candidates, dedup by predicted
  test grids, rank by training-set score, take top k. Reports accuracy at
  pass@1, pass@2, and pass@M (the ceiling). Top-level `accuracy` is pass@2.
- compute_baseline_metrics(results, token_usage): pure-Python over a baseline /
  CoT results list. File-level (task-level) accuracy is the top-level metric.
"""

from collections import defaultdict
from typing import Any


def compute_pipeline_metrics(cur, run_id: str, ks: tuple[int, ...] = (1, 2)) -> dict[str, Any]:
    # Imported here to keep the database layer free of an import-time
    # dependency on the pipeline package.
    from pipeline.selection import candidate_from_output, select_pass_at_k

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
        "SELECT task_id, agent, output FROM definers WHERE run_id = %s",
        (run_id,),
    )
    by_task: dict[str, list] = defaultdict(list)
    for task_id, agent, output in cur.fetchall():
        try:
            agent_idx = int(agent.rsplit("_", 1)[-1])
        except ValueError:
            agent_idx = 0
        by_task[task_id].append(candidate_from_output(agent_idx, output or {}))

    num_tasks = len(by_task)
    pass_counts = {k: 0 for k in ks}
    pass_at_m = 0
    tp_correct = tp_total = 0

    for cands in by_task.values():
        for k in ks:
            if select_pass_at_k(cands, k)["pass_at_k"]:
                pass_counts[k] += 1
        if select_pass_at_k(cands, max(len(cands), 1))["pass_at_k"]:
            pass_at_m += 1
        # test-pair-level accuracy is measured over the pass@1-selected candidate
        for sel in select_pass_at_k(cands, 1)["selected"]:
            tp_correct += sel["num_test_correct"]
            tp_total += sel["num_test_total"]

    accuracies = {
        f"accuracy_pass_at_{k}": (pass_counts[k] / num_tasks if num_tasks else None)
        for k in ks
    }
    acc_pm = (pass_at_m / num_tasks) if num_tasks else None
    primary = accuracies.get("accuracy_pass_at_2", accuracies.get("accuracy_pass_at_1"))

    return {
        "num_tasks": num_tasks,
        "num_explorer_rows": num_expl,
        "num_definer_rows": num_def,
        "pass_at_k_tasks": pass_counts,
        "pass_at_m_tasks": pass_at_m,
        **accuracies,
        "accuracy_pass_at_m": acc_pm,
        "accuracy": primary,
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
