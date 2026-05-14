"""CLI entry point for the pipeline."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from database.client import EvalClient
from shared.loader import load_task, get_task_ids
from shared.llm import create_async_client, get_default_model
from pipeline.agents.pattern_explorer.core import run_parallel_explorers
from pipeline.agents.transformation_definer.core import define_transformation
from pipeline.agents.pattern_explorer.types import ExplorationResult, PatternDocument
from pipeline.agents.transformation_definer.types import TransformationResult
from pipeline.io import new_run_id, run_dir, task_dir, save_pattern_document, save_transformation_result

load_dotenv()


class TaskLogger:
    """Per-task logger that writes to a file and optionally to console."""

    def __init__(self, task_id: str, log_dir: Path, quiet: bool = False):
        self.task_id = task_id
        self.quiet = quiet
        self.log_file = log_dir / f"{task_id}.log"
        self._fh = open(self.log_file, "w")

    def __call__(self, msg: str):
        self._fh.write(msg + "\n")
        self._fh.flush()
        if not self.quiet:
            print(f"[{self.task_id}] {msg}", flush=True)

    def close(self):
        self._fh.close()


def _console(msg: str = ""):
    print(msg, flush=True)


def _stamp_metadata(doc: PatternDocument, run_id: str, agent_idx: int, model: str, provider: str):
    doc.run_id = run_id
    doc.agent_idx = agent_idx
    doc.model = model
    doc.provider = provider
    doc.created_at = datetime.now(timezone.utc).isoformat()


async def _run_task(
    client,
    task_id: str,
    split: str,
    model: str,
    provider: str,
    output_dir: Path,
    run_id: str,
    num_explorers: int,
    max_steps: int,
    log_fn,
    db: EvalClient | None = None,
) -> dict:
    """Run N parallel explorers then the transformation definer on one task."""
    task = load_task(task_id, split)

    exploration: ExplorationResult = await run_parallel_explorers(
        task, client, model,
        num_explorers=num_explorers,
        max_steps=max_steps,
        log_fn=log_fn,
    )
    exploration.run_id = run_id

    saved_paths = []
    explorer_ids: list[str] = []
    for i, doc in enumerate(exploration.documents):
        _stamp_metadata(doc, run_id, i, model, provider)
        saved_paths.append(str(save_pattern_document(doc, output_dir)))
        if db is not None:
            output = {
                "patterns": [{"id": p.id, "text": p.text} for p in doc.patterns],
                "synthesis": doc.synthesis,
                "trace": [{"kind": t.kind, "content": t.content, "pattern_id": t.pattern_id} for t in doc.trace],
            }
            metadata = {
                "model": doc.model,
                "provider": doc.provider,
                "created_at": doc.created_at,
                "usage": doc.usage,
            }
            exp_id = db.record_explorer(run_id, task_id, i, output, metadata)
            if exp_id is not None:
                explorer_ids.append(exp_id)

    explorer_usage = exploration.total_usage

    log_fn("🔨 Running TransformationDefiner...")
    definer_result: TransformationResult = await define_transformation(
        task, exploration, client, model,
        max_steps=max_steps,
        log_fn=log_fn,
    )
    definer_result.run_id = run_id
    definer_result.agent_idx = 0
    definer_result.model = model
    definer_result.provider = provider
    definer_result.created_at = datetime.now(timezone.utc).isoformat()
    saved_paths.append(str(save_transformation_result(definer_result, output_dir)))
    if db is not None:
        output = {
            "code": definer_result.code,
            "reasoning": definer_result.reasoning,
            "transformation_summary": definer_result.transformation_summary,
            "trace": [{"kind": t.kind, "content": t.content} for t in definer_result.trace],
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
            "final_error": definer_result.final_error,
        }
        metadata = {
            "model": definer_result.model,
            "provider": definer_result.provider,
            "created_at": definer_result.created_at,
            "usage": definer_result.usage,
            "repair_attempts": definer_result.repair_attempts,
            "max_repairs": definer_result.max_repairs,
        }
        db.record_definer(run_id, task_id, 0, explorer_ids, output, metadata)

    total_prompt = explorer_usage["prompt_tokens"] + definer_result.usage["prompt_tokens"]
    total_completion = explorer_usage["completion_tokens"] + definer_result.usage["completion_tokens"]

    return {
        "task_id": task_id,
        "run_id": run_id,
        "num_explorers": len(exploration.documents),
        "patterns_per_explorer": [len(d.patterns) for d in exploration.documents],
        "transformation_correct": definer_result.correct,
        "num_test_pairs": len(definer_result.test_results),
        "num_correct": definer_result.num_correct,
        "transformation_error": definer_result.final_error,
        "files": saved_paths,
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "error": None,
    }


async def run(
    split: str = "training",
    limit: Optional[int] = None,
    task_ids: Optional[list[str]] = None,
    num_explorers: int = 3,
    max_concurrent_tasks: int = 4,
    output_dir: str = "output/pipeline",
    provider: str = "deepseek",
    model: Optional[str] = None,
    max_steps: int = 10,
    run_id: Optional[str] = None,
    quiet: bool = False,
    resume: bool = False,
):
    """Run pipeline on tasks."""
    if resume and not run_id:
        raise ValueError(
            "--resume requires --run-id: pass the UUID of the run to continue. "
            "Resume scans that run's output dir on disk for completed tasks."
        )

    client = create_async_client(provider)
    model_name = model or get_default_model(provider)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    run_id = run_id or new_run_id()

    this_run_dir = run_dir(out_path, run_id)
    log_dir = this_run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    db = EvalClient()
    db.register_run(run_id, "pipeline", split)

    if task_ids:
        ids = task_ids
    else:
        ids = get_task_ids(split)
        if limit:
            ids = ids[:limit]

    if resume:
        already_done = set()
        for tid in ids:
            td = task_dir(out_path, run_id, tid)
            if td.is_dir() and list(td.glob("transformation_definer_*.json")):
                already_done.add(tid)
        if already_done:
            ids = [tid for tid in ids if tid not in already_done]
            _console(f"Resuming: skipping {len(already_done)} already-completed tasks in run {run_id}")

    _console(f"Run {run_id} | {model_name} | {len(ids)} tasks | {num_explorers} explorers/task | max_concurrent={max_concurrent_tasks}")
    _console(f"Logs: {log_dir.resolve()}")
    _console()

    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    completed = {"correct": 0, "incorrect": 0, "error": 0, "no_output": 0}

    async def _guarded(task_id: str) -> dict:
        task_log = TaskLogger(task_id, log_dir, quiet=quiet)
        try:
            async with semaphore:
                _console(f"  ▶ {task_id}")
                result = await _run_task(
                    client, task_id, split, model_name, provider, out_path, run_id,
                    num_explorers, max_steps, task_log, db=db,
                )
                if result["transformation_correct"]:
                    tag, completed["correct"] = "✅", completed["correct"] + 1
                elif result["transformation_correct"] is False:
                    tag, completed["incorrect"] = "❌", completed["incorrect"] + 1
                else:
                    tag, completed["no_output"] = "⚠️", completed["no_output"] + 1
                tokens = result["prompt_tokens"] + result["completion_tokens"]
                _console(f"  {tag} {task_id} | {tokens:,} tokens")
                return result
        except Exception as e:
            completed["error"] = completed["error"] + 1
            _console(f"  💥 {task_id}: {e}")
            return {
                "task_id": task_id, "run_id": run_id, "num_explorers": 0,
                "patterns_per_explorer": [], "files": [],
                "prompt_tokens": 0, "completion_tokens": 0, "error": str(e),
                "transformation_correct": None, "num_test_pairs": 0, "num_correct": 0,
                "transformation_error": str(e),
            }
        finally:
            task_log.close()

    all_results = await asyncio.gather(*[_guarded(tid) for tid in ids])

    total_prompt = sum(r["prompt_tokens"] for r in all_results)
    total_completion = sum(r["completion_tokens"] for r in all_results)

    _console()
    _console(f"Done: {completed['correct']}✅ {completed['incorrect']}❌ {completed['no_output']}⚠️ {completed['error']}💥  ({len(all_results)} tasks)")
    _console(f"Tokens: {total_prompt + total_completion:,} ({total_prompt:,} prompt + {total_completion:,} completion)")

    summary = {
        "run_id": run_id,
        "split": split,
        "model": model_name,
        "provider": provider,
        "num_explorers": num_explorers,
        "max_steps": max_steps,
        "total_tasks": len(all_results),
        "correct": completed["correct"],
        "incorrect": completed["incorrect"],
        "no_output": completed["no_output"],
        "errors": completed["error"],
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

    db.finalize_pipeline_run(run_id)
    db.close()

    return list(all_results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ARC pipeline")
    parser.add_argument("--split", default="training")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task", type=str, default=None, help="Single task ID")
    parser.add_argument("--num-explorers", type=int, default=3)
    parser.add_argument("--max-concurrent-tasks", type=int, default=4)
    parser.add_argument("--output", type=str, default="output/pipeline")
    parser.add_argument("--provider", type=str, default="deepseek", choices=["deepseek", "openai"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--quiet", action="store_true", help="Only show per-task results on console (detailed logs still written to files)")
    parser.add_argument("--resume", action="store_true", help="Skip tasks that already have transformation_definer output (requires --run-id)")
    args = parser.parse_args()

    task_ids = [args.task] if args.task else None
    asyncio.run(run(
        split=args.split,
        limit=args.limit,
        task_ids=task_ids,
        num_explorers=args.num_explorers,
        max_concurrent_tasks=args.max_concurrent_tasks,
        output_dir=args.output,
        provider=args.provider,
        model=args.model,
        max_steps=args.max_steps,
        run_id=args.run_id,
        quiet=args.quiet,
        resume=args.resume,
    ))
