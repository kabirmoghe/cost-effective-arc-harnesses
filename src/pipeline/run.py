"""CLI entry point for the pipeline."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from database.client import EvalClient
from shared.loader import load_task, get_task_ids
from shared.llm import create_async_client, get_default_model, get_extra_body, get_tool_choice
from pipeline.agents.pattern_explorer.core import run_parallel_explorers
from pipeline.agents.transformation_definer.core import define_transformation
from pipeline.agents.pattern_explorer.types import ExplorationResult, PatternDocument
from pipeline.agents.transformation_definer.types import TransformationResult
from pipeline.io import new_run_id, run_dir, task_dir, save_pattern_document, save_transformation_result
from pipeline.selection import candidate_from_result, select_pass_at_k

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


def _hydrate_exploration(task_id: str, rows: list[dict]) -> tuple[ExplorationResult, list[str]]:
    """Rebuild an ExplorationResult from DB explorer rows (definer-only re-run).

    Returns the reconstructed result plus the source explorer agent_ids, so the
    new run's definers can record honest `parent_explorer_ids` provenance.
    """
    rows = sorted(rows, key=lambda r: r["agent_idx"])
    docs = []
    for row in rows:
        data = {"task_id": task_id, "agent_idx": row["agent_idx"], **row["output"], **row["metadata"]}
        docs.append(PatternDocument.from_dict(data))
    parent_ids = [row["agent_id"] for row in rows]
    return ExplorationResult(task_id=task_id, documents=docs), parent_ids


async def _run_task(
    client,
    task_id: str,
    split: str,
    model: str,
    provider: str,
    output_dir: Path,
    run_id: str,
    num_explorers: int,
    num_definers: int,
    max_steps: int,
    log_fn,
    db: EvalClient | None = None,
    exploration: ExplorationResult | None = None,
    parent_explorer_ids: list[str] | None = None,
    extra_body: dict | None = None,
    tool_choice: str = "auto",
    enable_refinement: bool = False,
    explorers_only: bool = False,
    explorer_temperature: float = 0.0,
    ablation: str | None = None,
) -> dict:
    """Run N parallel explorers (or reuse hydrated findings) then M parallel definers.

    If `explorers_only` is True, the definer phase is skipped — explorer findings
    are saved to disk + DB and the task returns immediately. Used for cheap
    `--from-explorers` source runs in N×t sweeps where we want to share one
    explorer pass across multiple downstream definer configurations.
    """
    task = load_task(task_id, split)
    reused = exploration is not None

    saved_paths: list[str] = []
    if reused:
        # definer-only re-run: explorer findings reused from a prior run, so
        # nothing is re-explored, re-saved, or re-recorded for the explorer phase.
        explorer_ids: list[str] = list(parent_explorer_ids or [])
        explorer_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    else:
        exploration = await run_parallel_explorers(
            task, client, model,
            num_explorers=num_explorers,
            max_steps=max_steps,
            log_fn=log_fn,
            extra_body=extra_body,
            temperature=explorer_temperature,
            tool_choice=tool_choice,
        )
        exploration.run_id = run_id
        explorer_ids = []
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

    if explorers_only:
        # Definer phase skipped — explorer findings are now in disk + DB ready
        # for downstream `--from-explorers <run_id>` consumers. Return a result
        # shape that lets the caller distinguish this from a full pipeline run.
        log_fn(f"📚 Explorer-only mode: {len(exploration.documents)} explorer(s) saved, "
               f"skipping definer phase")
        return {
            "task_id": task_id,
            "run_id": run_id,
            "num_explorers": len(exploration.documents),
            "num_definers": 0,
            "patterns_per_explorer": [len(d.patterns) for d in exploration.documents],
            "files": saved_paths,
            "prompt_tokens": explorer_usage["prompt_tokens"],
            "completion_tokens": explorer_usage["completion_tokens"],
            "explorer_only": True,
            # Definer-related fields kept present (with neutral values) so the
            # outer summary aggregation logic doesn't need explorer-only branches.
            "transformation_correct": None,
            "pass_at_1": False,
            "pass_at_2": False,
            "selection": None,
            "num_test_pairs": 0,
            "transformation_error": None,
            "error": None,
        }

    log_fn(f"🔨 Running {num_definers} TransformationDefiner(s)...")
    # Map --ablation flag (CLI-friendly hyphens) to the definer_variant value.
    # Default: "react" (standard B4/B5b). Currently the only ablation is
    # "act-only" (think tool removed). Add new ablations here as they land.
    _ablation_to_variant = {None: "react", "act-only": "act_only"}
    if ablation not in _ablation_to_variant:
        raise ValueError(f"Unknown --ablation value: {ablation!r}. Choices: {list(_ablation_to_variant)}")
    definer_variant = _ablation_to_variant[ablation]
    raw_definer_results = await asyncio.gather(
        *[
            define_transformation(
                task, exploration, client, model,
                max_steps=max_steps, log_fn=log_fn, extra_body=extra_body,
                enable_refinement=enable_refinement,
                definer_variant=definer_variant,
                tool_choice=tool_choice,
            )
            for _ in range(num_definers)
        ],
        return_exceptions=True,
    )
    definer_results: list[TransformationResult] = []
    for idx, r in enumerate(raw_definer_results):
        if isinstance(r, BaseException):
            log_fn(f"  ❌ definer {idx} failed after retries: {type(r).__name__}: {str(r)[:120]}")
            continue
        definer_results.append(r)
    if not definer_results:
        # All M definers failed: surface the first exception so the task is
        # flagged 💥 and re-runnable, instead of silently producing a no-result.
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
                # B5: full attempt history. Top-level fields above mirror the
                # best-by-train attempt (selected by the definer driver) for
                # backward compat; consumers wanting the full trajectory read
                # `attempts`. Empty list for definers that produced no clean
                # define (degenerate) or for B4-style single-attempt runs.
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
            }
            metadata = {
                "model": definer_result.model,
                "provider": definer_result.provider,
                "created_at": definer_result.created_at,
                "usage": definer_result.usage,
                "repair_attempts": definer_result.repair_attempts,
                "max_repairs": definer_result.max_repairs,
            }
            db.record_definer(run_id, task_id, idx, explorer_ids, output, metadata)

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
    split: str = "training",
    limit: Optional[int] = None,
    task_ids: Optional[list[str]] = None,
    num_explorers: int = 3,
    num_definers: int = 1,
    max_concurrent_tasks: int = 4,
    output_dir: str = "output/pipeline",
    provider: str = "deepseek",
    model: Optional[str] = None,
    max_steps: int = 10,
    run_id: Optional[str] = None,
    quiet: bool = False,
    resume: bool = False,
    from_explorers: Optional[str] = None,
    enable_refinement: bool = False,
    explorers_only: bool = False,
    explorer_temperature: float = 0.0,
    ablation: str | None = None,
):
    """Run pipeline on tasks."""
    if resume and not run_id:
        raise ValueError(
            "--resume requires --run-id: pass the UUID of the run to continue. "
            "Resume scans that run's output dir on disk for completed tasks."
        )
    if from_explorers and from_explorers == run_id:
        raise ValueError(
            "--from-explorers must differ from --run-id: reusing a run's explorer "
            "findings must produce a separate run, never write back into the source."
        )
    if explorers_only and from_explorers:
        raise ValueError(
            "--explorers-only and --from-explorers are mutually exclusive: "
            "explorer-only mode produces explorer findings, definer-only mode consumes them."
        )

    client = create_async_client(provider)
    model_name = model or get_default_model(provider)
    extra_body = get_extra_body(provider)
    tool_choice = get_tool_choice(provider)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    run_id = run_id or new_run_id()

    this_run_dir = run_dir(out_path, run_id)
    log_dir = this_run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    db = EvalClient()
    db.register_run(run_id, "pipeline", split)

    # Definer-only re-run: hydrate the exploration phase from a prior run instead
    # of re-exploring. Holds the explorer phase fixed so any delta is attributable
    # to definer fan-out + pass@k selection alone.
    hydrated: dict[str, tuple] = {}
    if from_explorers:
        explorers_by_task = db.fetch_explorers(from_explorers)
        if not explorers_by_task:
            raise ValueError(
                f"--from-explorers {from_explorers}: no explorer rows found in the DB for that run_id."
            )
        for tid, rows in explorers_by_task.items():
            hydrated[tid] = _hydrate_exploration(tid, rows)

    if task_ids:
        ids = task_ids
    else:
        ids = get_task_ids(split)
        if limit:
            ids = ids[:limit]

    if from_explorers:
        ids = [tid for tid in ids if tid in hydrated]

    if resume:
        # Different run modes write different per-task artifacts; check the right one.
        # Explorer-only runs write `pattern_explorer_*.json`; full and definer-only
        # runs write `transformation_definer_*.json`.
        artifact_glob = "pattern_explorer_*.json" if explorers_only else "transformation_definer_*.json"
        already_done = set()
        for tid in ids:
            td = task_dir(out_path, run_id, tid)
            if td.is_dir() and list(td.glob(artifact_glob)):
                already_done.add(tid)
        if already_done:
            ids = [tid for tid in ids if tid not in already_done]
            _console(f"Resuming: skipping {len(already_done)} already-completed tasks in run {run_id}")

    if from_explorers:
        _console(f"Run {run_id} | definer-only re-run from {from_explorers} | {model_name} | {len(ids)} tasks | {num_definers} definers/task | max_concurrent={max_concurrent_tasks}")
    elif explorers_only:
        _console(f"Run {run_id} | EXPLORER-ONLY | {model_name} | {len(ids)} tasks | N={num_explorers} explorers @ t={explorer_temperature} | max_concurrent={max_concurrent_tasks}")
    else:
        _console(f"Run {run_id} | {model_name} | {len(ids)} tasks | {num_explorers} explorers @ t={explorer_temperature} + {num_definers} definers/task | max_concurrent={max_concurrent_tasks}")
    _console(f"Logs: {log_dir.resolve()}")
    _console()

    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    completed = {"correct": 0, "incorrect": 0, "error": 0, "explored": 0}

    async def _guarded(task_id: str) -> dict:
        task_log = TaskLogger(task_id, log_dir, quiet=quiet)
        try:
            async with semaphore:
                _console(f"  ▶ {task_id}")
                exploration, parent_ids = hydrated.get(task_id, (None, None))
                result = await _run_task(
                    client, task_id, split, model_name, provider, out_path, run_id,
                    num_explorers, num_definers, max_steps, task_log, db=db,
                    exploration=exploration, parent_explorer_ids=parent_ids,
                    extra_body=extra_body,
                    tool_choice=tool_choice,
                    enable_refinement=enable_refinement,
                    explorers_only=explorers_only,
                    explorer_temperature=explorer_temperature,
                    ablation=ablation,
                )
                tokens = result["prompt_tokens"] + result["completion_tokens"]
                if result.get("explorer_only"):
                    completed["explored"] += 1
                    _console(f"  📚 {task_id} | {tokens:,} tokens")
                elif result["transformation_correct"]:
                    completed["correct"] += 1
                    _console(f"  ✅ {task_id} | {tokens:,} tokens")
                else:
                    completed["incorrect"] += 1
                    _console(f"  ❌ {task_id} | {tokens:,} tokens")
                return result
        except Exception as e:
            completed["error"] = completed["error"] + 1
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
    if explorers_only:
        _console(f"Done: {completed['explored']}📚 {completed['error']}💥  ({len(all_results)} tasks)")
    else:
        _console(f"Done: {completed['correct']}✅ {completed['incorrect']}❌ {completed['error']}💥  ({len(all_results)} tasks)")
    _console(f"Tokens: {total_prompt + total_completion:,} ({total_prompt:,} prompt + {total_completion:,} completion)")

    mode = "explorer_only" if explorers_only else ("definer_only" if from_explorers else "full")

    summary = {
        "run_id": run_id,
        "split": split,
        "model": model_name,
        "provider": provider,
        "num_explorers": num_explorers,
        "num_definers": 0 if explorers_only else num_definers,
        "explorer_temperature": explorer_temperature,
        "max_steps": max_steps,
        "mode": mode,
        "source_explorer_run_id": from_explorers,
        "total_tasks": len(all_results),
        "correct": completed["correct"],
        "incorrect": completed["incorrect"],
        "explored": completed["explored"],
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
        "num_definers": 0 if explorers_only else num_definers,
        "num_explorers": num_explorers,
        "explorer_temperature": explorer_temperature,
        "mode": mode,
    }
    if from_explorers:
        extra["source_explorer_run_id"] = from_explorers
    if ablation is not None:
        extra["ablation"] = ablation
    db.finalize_pipeline_run(run_id, extra=extra)
    db.close()

    return list(all_results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ARC pipeline")
    parser.add_argument("--split", default="training")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task", type=str, default=None, help="Single task ID")
    parser.add_argument("--num-explorers", type=int, default=3)
    parser.add_argument("--num-definers", type=int, default=1, help="M definers per task (default 1); pass@k selection ranks them by training-set score")
    parser.add_argument("--from-explorers", type=str, default=None, help="Definer-only re-run: reuse explorer findings from this source run_id, skipping the exploration phase (must differ from --run-id)")
    parser.add_argument("--max-concurrent-tasks", type=int, default=4)
    parser.add_argument("--output", type=str, default="output/pipeline")
    parser.add_argument("--provider", type=str, default="deepseek", choices=["deepseek", "openai", "openrouter", "openrouter-friendli", "openrouter-qwen3", "openrouter-llama-3.3", "openrouter-llama-4-maverick", "openrouter-kimi-k2"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--quiet", action="store_true", help="Only show per-task results on console (detailed logs still written to files)")
    parser.add_argument("--resume", action="store_true", help="Skip tasks that already have transformation_definer output (requires --run-id)")
    parser.add_argument("--definer-refinement", action="store_true", help="B5: enable train-feedback refinement loop in the definer (up to 2 refinements after the initial define, with train-failure feedback shown to the model). Off by default (B4 behavior).")
    parser.add_argument("--explorers-only", action="store_true", help="Run only the explorer phase: N explorers per task, save findings to disk + DB, then exit (no definer fan-out). Used to produce reusable explorer sets for downstream --from-explorers definer-only re-runs. Mutually exclusive with --from-explorers.")
    parser.add_argument("--explorer-temperature", type=float, default=0.0, help="Sampling temperature for the explorer phase. Default 0.0 (deterministic). Higher values inject diversity across N parallel explorers.")
    parser.add_argument("--ablation", type=str, default=None, choices=["act-only"], help="Run a definer ablation variant. 'act-only' = remove `think` tool from the definer (both Phase 1 and Phase 2). Stored in evals.data['ablation'] for downstream filtering.")
    args = parser.parse_args()

    task_ids = [args.task] if args.task else None
    asyncio.run(run(
        split=args.split,
        limit=args.limit,
        task_ids=task_ids,
        num_explorers=args.num_explorers,
        num_definers=args.num_definers,
        max_concurrent_tasks=args.max_concurrent_tasks,
        output_dir=args.output,
        provider=args.provider,
        model=args.model,
        max_steps=args.max_steps,
        run_id=args.run_id,
        quiet=args.quiet,
        resume=args.resume,
        from_explorers=args.from_explorers,
        enable_refinement=args.definer_refinement,
        explorers_only=args.explorers_only,
        explorer_temperature=args.explorer_temperature,
        ablation=args.ablation,
    ))
