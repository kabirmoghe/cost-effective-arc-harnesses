"""Faithful runner for the ARC Prize "Base LLM" harness.

For each test pair, runs `num_attempts` independent model calls with the same
context and same params (no temperature variation, no seed change, no
context-append between attempts) — relying on API non-determinism for diversity
between attempts. Pass@2 = pair is correct if ANY attempt matches. Mirrors the
upstream `ARCTester.generate_task_solution` loop in `main.py`.

Defaults come from the upstream `models.yml` `deepseek_chat` entry: temp 0.0,
max_tokens 4024, 2 attempts/pair. The model + OpenRouter provider pin (Friendli)
are inherited from `shared/llm.py`'s `openrouter` provider config — same V3.2
backend the pipeline runs against, so an arc_base number is directly comparable
to the pipeline's M=1 control on the same stack.
"""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from arc_base.prompt import build_prompt
from arc_base.extract_response import extract_response
from database.client import EvalClient
from shared.llm import create_client, get_default_model, get_extra_body
from shared.loader import load_task, get_task_ids
from shared.types import Grid, Task

load_dotenv()

_print_lock = threading.Lock()

# Upstream `models.yml` `deepseek_chat` config (verified 2026-05-14).
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 4024
DEFAULT_NUM_ATTEMPTS = 2


def log(msg: str = ""):
    with _print_lock:
        print(msg, flush=True)


def _one_attempt(
    client: OpenAI,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    extra_body: dict | None,
) -> tuple[Grid | None, str | None, dict]:
    """Issue one model call. Returns (predicted_grid, error, usage_dict).

    A single user message carries the full ARC template — the upstream file is
    named `system_prompt.txt` but contains the data placeholders, so it functions
    as the whole prompt. User-role keeps this provider-portable.
    """
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        raw = resp.choices[0].message.content or ""
        if resp.usage:
            usage["prompt_tokens"] = resp.usage.prompt_tokens
            usage["completion_tokens"] = resp.usage.completion_tokens
        return extract_response(raw), None, usage
    except Exception as e:
        return None, str(e), usage


def _run_one_pair(
    client: OpenAI,
    task: Task,
    test_index: int,
    model: str,
    temperature: float,
    max_tokens: int,
    num_attempts: int,
    extra_body: dict | None,
) -> dict:
    """N attempts on one test pair, scored pass@N (any attempt correct)."""
    prompt = build_prompt(task, test_index)
    expected = task.test[test_index].output

    attempts = []
    for _ in range(num_attempts):
        predicted, err, usage = _one_attempt(
            client, prompt, model, temperature, max_tokens, extra_body
        )
        attempts.append({
            "answer": predicted,
            "correct": (err is None and predicted == expected),
            "error": err,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
        })

    any_correct = any(a["correct"] for a in attempts)
    all_errored = all(a["error"] is not None for a in attempts)
    return {
        "task_id": task.task_id,
        "test_index": test_index,
        "correct": any_correct,
        "error": "all attempts errored" if all_errored else None,
        "attempts": attempts,
        "prompt_tokens": sum(a["prompt_tokens"] for a in attempts),
        "completion_tokens": sum(a["completion_tokens"] for a in attempts),
    }


def _run_single(
    client: OpenAI,
    task_id: str,
    split: str,
    model: str,
    temperature: float,
    max_tokens: int,
    num_attempts: int,
    extra_body: dict | None,
) -> list[dict]:
    """Run all test pairs of one task."""
    task = load_task(task_id, split)
    return [
        _run_one_pair(
            client, task, i, model, temperature, max_tokens, num_attempts, extra_body
        )
        for i in range(task.num_test)
    ]


def evaluate(
    split: str = "evaluation",
    limit: Optional[int] = None,
    task_ids: Optional[list[str]] = None,
    workers: int = 1,
    provider: str = "openrouter",
    model: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    num_attempts: int = DEFAULT_NUM_ATTEMPTS,
    output_file: Optional[str] = None,
):
    """Run the ARC Base LLM harness on a set of tasks."""
    log(f"Initializing {provider} client...")
    client = create_client(provider)
    model_name = model or get_default_model(provider)
    extra_body = get_extra_body(provider)
    log(f"Model: {model_name} | temp={temperature} | max_tokens={max_tokens} | "
        f"attempts/pair={num_attempts}")
    log("")

    if task_ids:
        ids = task_ids
    else:
        ids = get_task_ids(split)
        if limit:
            ids = ids[:limit]

    log(f"Evaluating {len(ids)} tasks from '{split}' (workers={workers})")
    log("=" * 60)

    all_results: list[dict] = []

    if workers <= 1:
        for task_id in ids:
            for r in _run_single(
                client, task_id, split, model_name, temperature,
                max_tokens, num_attempts, extra_body,
            ):
                all_results.append(r)
                tag = "CORRECT" if r["correct"] else ("ERROR" if r["error"] else "WRONG")
                log(f"  {task_id}[{r['test_index']}]: {tag}")
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(
                    _run_single, client, tid, split, model_name, temperature,
                    max_tokens, num_attempts, extra_body,
                ): tid
                for tid in ids
            }
            tasks_done = 0
            for fut in as_completed(futures):
                tid = futures[fut]
                tasks_done += 1
                try:
                    pair_results = fut.result()
                except Exception as e:
                    log(f"  [{tasks_done}/{len(ids)}] {tid}: EXCEPTION {e}")
                    all_results.append({
                        "task_id": tid, "test_index": 0, "correct": False,
                        "error": str(e), "attempts": [],
                        "prompt_tokens": 0, "completion_tokens": 0,
                    })
                    continue
                all_results.extend(pair_results)
                # Task-level tag: correct iff ALL its pairs are correct.
                task_correct = all(r["correct"] for r in pair_results)
                running_correct = sum(1 for r in all_results if r.get("correct"))
                tag = "✅" if task_correct else "❌"
                log(f"  [{tasks_done}/{len(ids)}] {tid} {tag} | "
                    f"running pair-correct: {running_correct}/{len(all_results)} "
                    f"({running_correct/len(all_results)*100:.1f}%)")

    total_pairs = len(all_results)
    pair_correct = sum(1 for r in all_results if r["correct"])
    errors = sum(1 for r in all_results if r["error"])
    total_prompt = sum(r["prompt_tokens"] for r in all_results)
    total_completion = sum(r["completion_tokens"] for r in all_results)

    # File-level: a task is correct iff every test pair has at least one
    # correct attempt. That's the ARC scoring convention.
    by_task: dict[str, bool] = {}
    for r in all_results:
        tid = r["task_id"]
        by_task[tid] = by_task.get(tid, True) and bool(r["correct"])
    correct_files = sum(1 for v in by_task.values() if v)

    log("")
    log("=" * 60)
    log(f"Pair-level pass@{num_attempts}: {pair_correct}/{total_pairs} "
        f"({pair_correct/total_pairs*100:.1f}%)  ({errors} errors)")
    log(f"File-level pass@{num_attempts}: {correct_files}/{len(by_task)} "
        f"({correct_files/len(by_task)*100:.1f}%)")
    log(f"Tokens: {total_prompt:,} prompt + {total_completion:,} completion = "
        f"{total_prompt+total_completion:,} total")

    token_usage = {
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
    }

    if output_file:
        with open(output_file, "w") as f:
            json.dump({
                "split": split,
                "model": model_name,
                "provider": provider,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "num_attempts": num_attempts,
                "total_pairs": total_pairs,
                "pair_correct": pair_correct,
                "file_correct": correct_files,
                "errors": errors,
                "token_usage": token_usage,
                "results": all_results,
            }, f, indent=2)
        log(f"Results saved to {output_file}")

    db = EvalClient()
    run_id = db.record_baseline_run("arc_base", split, all_results, token_usage)
    db.close()
    if run_id is not None:
        log(f"Recorded arc_base run {run_id} in DB")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ARC Prize 'Base LLM' harness (faithful port)")
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task", type=str, default=None, help="Single task ID")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--provider", type=str, default="openrouter",
                        choices=["deepseek", "openai", "openrouter"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--num-attempts", type=int, default=DEFAULT_NUM_ATTEMPTS,
                        help="Attempts per test pair (pass@N). Default 2 per ARC convention.")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional JSON dump of all results")
    args = parser.parse_args()

    task_ids = [args.task] if args.task else None
    evaluate(
        split=args.split,
        limit=args.limit,
        task_ids=task_ids,
        workers=args.workers,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_attempts=args.num_attempts,
        output_file=args.output,
    )
