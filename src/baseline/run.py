"""No-CoT few-shot baseline runner for ARC tasks using DeepSeek."""

import sys
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from database.client import EvalClient
from shared.types import Grid, Task
from shared.loader import load_task, get_task_ids
from shared.llm import create_client, get_default_model, get_extra_body, get_response_format, with_retry_sync
from baseline.prompt import BASELINE_SYSTEM_PROMPT, build_user_message
from baseline.extract_response import extract_response

load_dotenv()

_print_lock = threading.Lock()

MAX_TOKENS = 8192


def log(msg: str = ""):
    """Thread-safe print with flush."""
    with _print_lock:
        print(msg, flush=True)


def solve_task(client: OpenAI, task: Task, test_index: int = 0, stream: bool = False, provider: str = "openrouter") -> tuple[Grid, dict]:
    """Solve a single ARC task test case with few-shot only (no CoT).

    Returns (predicted_grid, usage_dict).
    """
    user_msg = build_user_message(task, test_index)
    usage = {"prompt_tokens": 0, "completion_tokens": 0}

    if stream:
        resp_stream = client.chat.completions.create(
            model=get_default_model(provider),
            messages=[
                {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            response_format=get_response_format(provider),
            extra_body=get_extra_body(provider),
            stream=True,
            stream_options={"include_usage": True},
        )
        raw = ""
        with _print_lock:
            sys.stdout.write(f"  [{task.task_id}] Streaming: ")
            sys.stdout.flush()
        for chunk in resp_stream:
            if chunk.usage:
                usage["prompt_tokens"] = chunk.usage.prompt_tokens
                usage["completion_tokens"] = chunk.usage.completion_tokens
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                raw += token
                with _print_lock:
                    sys.stdout.write(token)
                    sys.stdout.flush()
        with _print_lock:
            sys.stdout.write("\n")
            sys.stdout.flush()
    else:
        resp = with_retry_sync(
            lambda: client.chat.completions.create(
                model=get_default_model(provider),
                messages=[
                    {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=MAX_TOKENS,
                response_format=get_response_format(provider),
                extra_body=get_extra_body(provider),
            ),
            label=f"baseline.{provider}",
        )
        raw = resp.choices[0].message.content
        usage["prompt_tokens"] = resp.usage.prompt_tokens
        usage["completion_tokens"] = resp.usage.completion_tokens

    return extract_response(raw), usage


def _run_single(client: OpenAI, task_id: str, split: str, provider: str = "openrouter") -> list[dict]:
    """Run a single task and return result dicts. Used by parallel evaluate."""
    task = load_task(task_id, split)
    results = []
    for test_idx in range(task.num_test):
        try:
            predicted, usage = solve_task(client, task, test_idx, provider=provider)
            expected = task.test[test_idx].output
            match = predicted == expected
            results.append({
                "task_id": task_id,
                "test_index": test_idx,
                "correct": match,
                "error": None,
                "predicted": predicted,
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
            })
        except Exception as e:
            results.append({
                "task_id": task_id,
                "test_index": test_idx,
                "correct": False,
                "error": str(e),
                "predicted": None,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            })
    return results


def evaluate(
    split: str = "training",
    limit: Optional[int] = None,
    task_ids: Optional[list[str]] = None,
    workers: int = 1,
    stream: bool = False,
    output_file: Optional[str] = None,
    provider: str = "openrouter",
):
    """Evaluate no-CoT baseline on a set of tasks."""
    log(f"Initializing client (provider={provider})...")
    client = create_client(provider)
    log("Client ready.\n")

    if task_ids:
        ids = task_ids
    else:
        ids = get_task_ids(split)
        if limit:
            ids = ids[:limit]

    log(f"Evaluating {len(ids)} tasks from '{split}' split (no-CoT, workers={workers})")
    log(f"{'='*60}")

    all_results = []

    if workers <= 1:
        correct = 0
        total = 0
        errors = 0
        for task_id in ids:
            task = load_task(task_id, split)
            for test_idx in range(task.num_test):
                total += 1
                log(f"\n[{total}] Task {task_id} (test {test_idx})")
                try:
                    predicted, usage = solve_task(client, task, test_idx, stream=stream, provider=provider)
                    expected = task.test[test_idx].output
                    match = predicted == expected
                    if match:
                        correct += 1
                    status = "CORRECT" if match else "WRONG"
                    log(f"  Result: {status} | {usage['completion_tokens']}c tokens | Running: {correct}/{total} ({correct/total*100:.1f}%)")
                    all_results.append({"task_id": task_id, "test_index": test_idx, "correct": match, "error": None,
                                        "predicted": predicted,
                                        "prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"]})
                except Exception as e:
                    errors += 1
                    log(f"  ERROR: {e}")
                    all_results.append({"task_id": task_id, "test_index": test_idx, "correct": False, "error": str(e),
                                        "predicted": None,
                                        "prompt_tokens": 0, "completion_tokens": 0})
    else:
        tasks_done = 0
        tests_done = 0
        correct = 0
        errors = 0
        total_tasks = len(ids)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_single, client, task_id, split, provider): task_id
                for task_id in ids
            }
            for future in as_completed(futures):
                task_id = futures[future]
                tasks_done += 1
                try:
                    task_results = future.result()
                    all_results.extend(task_results)
                    for r in task_results:
                        tests_done += 1
                        if r["error"]:
                            errors += 1
                        elif r["correct"]:
                            correct += 1
                    first = task_results[0]
                    label = "CORRECT" if first["correct"] else ("ERROR" if first["error"] else "WRONG")
                    err_info = f" ({first['error'][:60]})" if first["error"] else ""
                    log(f"  [{tasks_done}/{total_tasks} tasks, {tests_done} tests] {task_id}: {label}{err_info} "
                        f"| Running: {correct}/{tests_done} ({correct/tests_done*100:.1f}%)")
                except Exception as e:
                    tests_done += 1
                    errors += 1
                    log(f"  [{tasks_done}/{total_tasks} tasks] {task_id}: EXCEPTION {e}")
                    all_results.append({"task_id": task_id, "test_index": 0, "correct": False, "error": str(e),
                                        "predicted": None,
                                        "prompt_tokens": 0, "completion_tokens": 0})

    total_tests = len(all_results)
    correct = sum(1 for r in all_results if r["correct"])
    errors = sum(1 for r in all_results if r["error"])
    total_prompt = sum(r["prompt_tokens"] for r in all_results)
    total_completion = sum(r["completion_tokens"] for r in all_results)

    log(f"\n{'='*60}")
    log(f"Final Results: {correct}/{total_tests} correct ({correct/total_tests*100:.1f}%), {errors} errors")
    log(f"Token usage: {total_prompt:,} prompt + {total_completion:,} completion = {total_prompt+total_completion:,} total")

    token_usage = {
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
    }

    if output_file:
        output = {
            "split": split,
            "model": "deepseek-chat",
            "baseline": "no-cot",
            "total": total_tests,
            "correct": correct,
            "errors": errors,
            "accuracy": correct / total_tests if total_tests else 0,
            "token_usage": token_usage,
            "results": all_results,
        }
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        log(f"Results saved to {output_file}")

    db = EvalClient()
    run_id = db.record_baseline_run("baseline", split, all_results, token_usage)
    db.close()
    if run_id is not None:
        log(f"Recorded baseline run {run_id} in DB")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="No-CoT few-shot baseline for ARC")
    parser.add_argument("--split", default="training")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task", type=str, default=None, help="Single task ID to run")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default: 1 = sequential)")
    parser.add_argument("--stream", action="store_true", help="Stream responses (only in sequential mode)")
    parser.add_argument("--output", type=str, default=None, help="Save results JSON to file")
    parser.add_argument("--provider", type=str, default="openrouter",
                        choices=["deepseek", "openai", "openrouter", "openrouter-friendli",
                                 "openrouter-qwen3", "openrouter-llama-3.3", "openrouter-llama-4-maverick", "openrouter-kimi-k2"])
    args = parser.parse_args()

    task_ids = [args.task] if args.task else None
    evaluate(
        split=args.split,
        limit=args.limit,
        task_ids=task_ids,
        workers=args.workers,
        stream=args.stream,
        output_file=args.output,
        provider=args.provider,
    )
