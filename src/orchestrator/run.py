"""B8 — Reflective Orchestrator CLI entrypoint.

Definer-only re-run only: orchestrator always consumes a pre-existing explorer
cell via `--from-explorers <source_run_id>`. This keeps the architecture as a
clean single-axis ablation against B7 (same Phase 1 explorers, only the definer
loop differs).

Reuses pipeline scaffolding via library imports — `load_task`, `get_task_ids`,
`EvalClient`, `_hydrate_exploration`, `candidate_from_result`,
`select_pass_at_k`, output-saving helpers. Does NOT add any orchestrator-aware
flags to `pipeline/run.py`.
"""

import asyncio
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from database.client import EvalClient
from shared.loader import load_task, get_task_ids
from shared.llm import create_async_client, get_default_model, get_extra_body
from pipeline.io import new_run_id, run_dir, task_dir, save_transformation_result
from pipeline.selection import candidate_from_result, select_pass_at_k
from pipeline.run import _hydrate_exploration, TaskLogger, _console

from orchestrator.core import run_orchestrator
from orchestrator.types import OrchestratorResult

load_dotenv()


async def _run_task(
    client,
    task_id: str,
    split: str,
    model: str,
    provider: str,
    output_dir: Path,
    run_id: str,
    num_definers: int,
    log_fn,
    db: EvalClient,
    exploration,
    parent_explorer_ids: list[str],
    extra_body: dict | None,
) -> dict:
    """Run M orchestrator definers on one task, write to DB and disk."""
    task = load_task(task_id, split)
    saved_paths: list[str] = []
    explorer_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    log_fn(f"🔨 Running {num_definers} Orchestrator definer(s)...")

    # Each definer gets a private deep-copy of exploration so spawn calls in
    # one definer don't contaminate sibling definers' context.
    raw_definer_results = await asyncio.gather(
        *[
            run_orchestrator(
                task, deepcopy(exploration), client, model,
                log_fn=log_fn, extra_body=extra_body,
            )
            for _ in range(num_definers)
        ],
        return_exceptions=True,
    )
    definer_results: list[OrchestratorResult] = []
    for idx, r in enumerate(raw_definer_results):
        if isinstance(r, BaseException):
            log_fn(f"  ❌ definer {idx} failed after retries: {type(r).__name__}: {str(r)[:120]}")
            continue
        definer_results.append(r)
    if not definer_results:
        raise next(r for r in raw_definer_results if isinstance(r, BaseException))

    definer_prompt = 0
    definer_completion = 0
    for idx, definer_result in enumerate(definer_results):
        definer_result.run_id = run_id
        definer_result.agent_idx = idx
        definer_result.model = model
        definer_result.provider = provider
        definer_result.created_at = datetime.now(timezone.utc).isoformat()
        saved_paths.append(str(save_transformation_result(definer_result, output_dir)))
        definer_prompt += definer_result.usage["prompt_tokens"]
        definer_completion += definer_result.usage["completion_tokens"]
        if db is not None:
            output = {
                "code": definer_result.code,
                "reasoning": definer_result.reasoning,
                "transformation_summary": definer_result.transformation_summary,
                "trace": [{"kind": t.kind, "content": t.content, "args": t.args} for t in definer_result.trace],
                "test_results": [
                    {
                        "test_index": tr.test_index,
                        "predicted_output": tr.predicted_output,
                        "expected_output": tr.expected_output,
                        "correct": tr.correct,
                        "error": tr.error,
                    }
                    for tr in definer_result.test_results
                ],
                "correct": definer_result.correct,
                "success": definer_result.success,
                "num_correct": definer_result.num_correct,
                "train_num_correct": definer_result.train_num_correct,
                "train_num_total": definer_result.train_num_total,
                "final_error": definer_result.final_error,
                "attempts": [
                    {
                        "iter": a.iter,
                        "phase": a.phase,
                        "code": a.code,
                        "transformation_summary": a.transformation_summary,
                        "reasoning": a.reasoning,
                        "test_results": [
                            {
                                "test_index": tr.test_index,
                                "predicted_output": tr.predicted_output,
                                "expected_output": tr.expected_output,
                                "correct": tr.correct,
                                "error": tr.error,
                            }
                            for tr in a.test_results
                        ],
                        "train_results": [
                            {
                                "pair_index": tr.pair_index,
                                "input_grid": tr.input_grid,
                                "expected_output": tr.expected_output,
                                "predicted_output": tr.predicted_output,
                                "correct": tr.correct,
                                "error": tr.error,
                            }
                            for tr in a.train_results
                        ],
                        "train_num_correct": a.train_num_correct,
                        "train_num_total": a.train_num_total,
                        "final_error": a.final_error,
                    }
                    for a in definer_result.attempts
                ],
                # Orchestrator-specific telemetry.
                "exit_reason": definer_result.exit_reason,
                "done_reason": definer_result.done_reason,
                "iterations_used": definer_result.iterations_used,
                "spawn_calls": definer_result.spawn_calls,
            }
            metadata = {
                "model": definer_result.model,
                "provider": definer_result.provider,
                "created_at": definer_result.created_at,
                "usage": definer_result.usage,
                "repair_attempts": definer_result.repair_attempts,
                "max_repairs": definer_result.max_repairs,
            }
            db.record_definer(run_id, task_id, idx, parent_explorer_ids, output, metadata)

    candidates = [candidate_from_result(r) for r in definer_results]
    sel_1 = select_pass_at_k(candidates, 1)
    sel_2 = select_pass_at_k(candidates, 2)

    total_prompt = explorer_usage["prompt_tokens"] + definer_prompt
    total_completion = explorer_usage["completion_tokens"] + definer_completion

    return {
        "task_id": task_id,
        "run_id": run_id,
        "num_explorers": len(exploration.documents),
        "num_definers": len(definer_results),
        "patterns_per_explorer": [len(d.patterns) for d in exploration.documents],
        "transformation_correct": sel_2["pass_at_k"],
        "pass_at_1": sel_1["pass_at_k"],
        "pass_at_2": sel_2["pass_at_k"],
        "selection": sel_2,
        "num_test_pairs": len(definer_results[0].test_results) if definer_results else 0,
        "transformation_error": None,
        "files": saved_paths,
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "error": None,
    }


async def run(
    split: str,
    from_explorers: str,
    limit: Optional[int] = None,
    task_ids: Optional[list[str]] = None,
    num_definers: int = 5,
    max_concurrent_tasks: int = 4,
    output_dir: str = "output/orchestrator",
    provider: str = "openrouter",
    model: Optional[str] = None,
    run_id: Optional[str] = None,
    quiet: bool = False,
    resume: bool = False,
):
    """Run B8 orchestrator. Always definer-only re-run from a saved explorer cell."""
    if not from_explorers:
        raise ValueError(
            "Orchestrator requires --from-explorers <source_run_id>. The orchestrator "
            "is defined as a definer-side architecture; initial explorers come from a "
            "pre-existing explorer cell so the comparison vs B7 is single-axis."
        )
    if resume and not run_id:
        raise ValueError("--resume requires --run-id.")
    if from_explorers == run_id:
        raise ValueError("--from-explorers must differ from --run-id.")

    client = create_async_client(provider)
    model_name = model or get_default_model(provider)
    extra_body = get_extra_body(provider)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    run_id = run_id or new_run_id()

    this_run_dir = run_dir(out_path, run_id)
    log_dir = this_run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    db = EvalClient()
    db.register_run(run_id, "orchestrator", split)

    # Hydrate explorers from the source cell.
    explorers_by_task = db.fetch_explorers(from_explorers)
    if not explorers_by_task:
        raise ValueError(
            f"--from-explorers {from_explorers}: no explorer rows found in the DB."
        )
    hydrated: dict[str, tuple] = {
        tid: _hydrate_exploration(tid, rows) for tid, rows in explorers_by_task.items()
    }

    if task_ids:
        ids = task_ids
    else:
        ids = get_task_ids(split)
        if limit:
            ids = ids[:limit]

    ids = [tid for tid in ids if tid in hydrated]

    if resume:
        already_done = set()
        for tid in ids:
            td = task_dir(out_path, run_id, tid)
            if td.is_dir() and list(td.glob("transformation_definer_*.json")):
                already_done.add(tid)
        if already_done:
            ids = [tid for tid in ids if tid not in already_done]
            _console(f"Resuming: skipping {len(already_done)} already-completed tasks in run {run_id}")

    _console(
        f"Run {run_id} | ORCHESTRATOR | definer-only re-run from {from_explorers} | "
        f"{model_name} | {len(ids)} tasks | {num_definers} definers/task | "
        f"max_concurrent={max_concurrent_tasks}"
    )
    _console(f"Logs: {log_dir.resolve()}")
    _console()

    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    completed = {"correct": 0, "incorrect": 0, "error": 0}

    async def _guarded(task_id: str) -> dict:
        task_log = TaskLogger(task_id, log_dir, quiet=quiet)
        try:
            async with semaphore:
                _console(f"  ▶ {task_id}")
                exploration, parent_ids = hydrated[task_id]
                result = await _run_task(
                    client, task_id, split, model_name, provider, out_path, run_id,
                    num_definers, task_log, db=db,
                    exploration=exploration, parent_explorer_ids=parent_ids,
                    extra_body=extra_body,
                )
                tokens = result["prompt_tokens"] + result["completion_tokens"]
                if result["transformation_correct"]:
                    completed["correct"] += 1
                    _console(f"  ✅ {task_id} | {tokens:,} tokens")
                else:
                    completed["incorrect"] += 1
                    _console(f"  ❌ {task_id} | {tokens:,} tokens")
                return result
        except Exception as e:
            completed["error"] += 1
            _console(f"  💥 {task_id}: {e}")
            return {
                "task_id": task_id, "run_id": run_id, "num_explorers": 0,
                "num_definers": 0, "patterns_per_explorer": [], "files": [],
                "prompt_tokens": 0, "completion_tokens": 0, "error": str(e),
                "transformation_correct": False, "pass_at_1": False, "pass_at_2": False,
                "selection": None, "num_test_pairs": 0,
                "transformation_error": str(e),
            }
        finally:
            task_log.close()

    all_results = await asyncio.gather(*[_guarded(tid) for tid in ids])

    total_prompt = sum(r["prompt_tokens"] for r in all_results)
    total_completion = sum(r["completion_tokens"] for r in all_results)

    _console()
    _console(f"Done: {completed['correct']}✅ {completed['incorrect']}❌ {completed['error']}💥  ({len(all_results)} tasks)")
    _console(f"Tokens: {total_prompt + total_completion:,} ({total_prompt:,} prompt + {total_completion:,} completion)")

    summary = {
        "run_id": run_id,
        "split": split,
        "model": model_name,
        "provider": provider,
        "architecture": "orchestrator",
        "num_definers": num_definers,
        "source_explorer_run_id": from_explorers,
        "total_tasks": len(all_results),
        "correct": completed["correct"],
        "incorrect": completed["incorrect"],
        "errors": completed["error"],
        "pass_at_1": sum(1 for r in all_results if r.get("pass_at_1")),
        "pass_at_2": sum(1 for r in all_results if r.get("pass_at_2")),
        "token_usage": {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
        },
        "results": list(all_results),
    }
    summary_path = this_run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    _console(f"Summary: {summary_path}")

    extra = {
        "num_definers": num_definers,
        "mode": "definer_only",
        "architecture": "orchestrator",
        "source_explorer_run_id": from_explorers,
    }
    db.finalize_pipeline_run(run_id, extra=extra)
    db.close()

    return list(all_results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="B8 Reflective Orchestrator")
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--from-explorers", type=str, required=True,
                        help="Source run_id whose saved explorer cell to consume (definer-only re-run).")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task", type=str, default=None, help="Single task ID")
    parser.add_argument("--num-definers", type=int, default=5)
    parser.add_argument("--max-concurrent-tasks", type=int, default=4)
    parser.add_argument("--output", type=str, default="output/orchestrator")
    parser.add_argument("--provider", type=str, default="openrouter",
                        choices=["deepseek", "openai", "openrouter", "openrouter-friendli"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    task_ids = [args.task] if args.task else None
    asyncio.run(run(
        split=args.split,
        from_explorers=args.from_explorers,
        limit=args.limit,
        task_ids=task_ids,
        num_definers=args.num_definers,
        max_concurrent_tasks=args.max_concurrent_tasks,
        output_dir=args.output,
        provider=args.provider,
        model=args.model,
        run_id=args.run_id,
        quiet=args.quiet,
        resume=args.resume,
    ))
