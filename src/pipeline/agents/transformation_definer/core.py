"""TransformationDefiner agent — synthesize explorer findings into executable code.

The agent runs in two phases:

  **Phase 1** (initial define + execution repair): up to 10 step LLM loop. The
  model uses `think` and `define_transformation` tools. On a clean execution,
  records an `Attempt` and exits the phase. On execution error, increments
  `repair_attempts` and continues (up to 3 repairs).

  **Phase 2** (train-feedback refinement, B5): runs only if Phase 1 produced a
  clean Attempt with `train_num_correct < train_num_total`. Up to 2 refinement
  iterations; each shows the model its failing train pairs and gives it up to
  5 substeps to refine. Same execution-repair logic as Phase 1.

Both phases call a single primitive (`_run_define_loop`) so the step/repair/
warning/execute logic lives in one place.
"""

import json
from dataclasses import dataclass
from typing import Callable, Optional

from openai import AsyncOpenAI

from shared.llm import with_retry
from shared.types import Task
from shared.code_exec import execute_transformation
from pipeline.agents.pattern_explorer.types import ExplorationResult

from .feedback import render_train_feedback
from .types import (
    Attempt, TestPairResult, TrainPairResult, TraceEntry, TransformationResult,
)
from .context.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_B5
from .context.rendering import build_definer_messages
from .tools import TOOL_DEFINITIONS

_noop = lambda msg: None

MAX_REFINEMENTS = 2  # Phase 2 iterations; total defines per definer ≤ 3
MAX_REFINEMENT_SUBSTEPS = 5


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _remaining_steps_message(remaining: int, total: int) -> str:
    """Graduated urgency message. `remaining` = steps left INCLUDING the current one."""
    step_num = total - remaining + 1
    if remaining <= 1:
        return (
            f"This is step {step_num} of {total} — your final step in this loop. "
            f"You must call `define_transformation` in this response."
        )
    return (
        f"This is step {step_num} of {total}. You have {remaining} steps total "
        f"(this one and {remaining - 1} more) to call `define_transformation`. "
        f"Plan to define your transformation soon."
    )


def _parse_tool_calls(
    response,
    trace: list[TraceEntry],
    log_fn: Callable[[str], None] = _noop,
) -> dict | None:
    """Extract tool calls; append to trace; return define_transformation args if any."""
    message = response.choices[0].message
    if not message.tool_calls:
        return None

    result = None
    for tc in message.tool_calls:
        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            continue

        if tc.function.name == "think":
            thought = args.get("thought", "")
            trace.append(TraceEntry(kind="think", content=thought))
            log_fn(f"    💭 think: {thought[:120]}")

        elif tc.function.name == "define_transformation":
            trace.append(TraceEntry(
                kind="define_transformation",
                content=args.get("transformation_summary", ""),
            ))
            log_fn(f"    🎯 define_transformation: {args.get('transformation_summary', '')[:120]}")
            result = args

    return result


def _accumulate_usage(usage: dict, response):
    if response.usage:
        usage["prompt_tokens"] += response.usage.prompt_tokens
        usage["completion_tokens"] += response.usage.completion_tokens


def _exec_on_test_pairs(code: str, task: Task) -> tuple[list[TestPairResult], str | None]:
    """Run the transform on all test pairs. Returns (results, first_error_if_any).

    Preserves B4 semantics: any test-pair execution error → triggers repair branch.
    """
    results: list[TestPairResult] = []
    first_error: str | None = None
    for i in range(task.num_test):
        predicted, error = execute_transformation(code, task.test[i].input)
        if error is not None:
            first_error = first_error or error
            results.append(TestPairResult(test_index=i, error=error))
        else:
            results.append(TestPairResult(
                test_index=i,
                predicted_output=predicted,
                expected_output=task.test[i].output,
                correct=(predicted == task.test[i].output),
            ))
    return results, first_error


def _exec_on_train_pairs(code: str, task: Task) -> list[TrainPairResult]:
    """Run the transform on all train pairs; capture results including errors.

    Unlike test execution, train errors do NOT trigger the repair branch — they
    surface as failing pairs in the B5 refinement feedback instead. Train
    failures during Phase 1 are silent (matching B4 behavior).
    """
    results: list[TrainPairResult] = []
    for i, ex in enumerate(task.train):
        predicted, error = execute_transformation(code, ex.input)
        if error is not None:
            results.append(TrainPairResult(
                pair_index=i,
                input_grid=ex.input,
                expected_output=ex.output,
                error=error,
                correct=False,
            ))
        else:
            results.append(TrainPairResult(
                pair_index=i,
                input_grid=ex.input,
                expected_output=ex.output,
                predicted_output=predicted,
                correct=(predicted == ex.output),
            ))
    return results


def _failing_train_pairs_payload(attempt: Attempt) -> list[dict]:
    """Convert the last attempt's failing train results into the dict shape
    that `render_train_feedback` expects."""
    payload = []
    for tr in attempt.train_results:
        if tr.correct:
            continue
        payload.append({
            "pair_index": tr.pair_index,
            "input": tr.input_grid,
            "expected": tr.expected_output,
            "predicted": tr.predicted_output,
            "error": tr.error,
        })
    return payload


# ────────────────────────────────────────────────────────────────────
# Unified define loop primitive
# ────────────────────────────────────────────────────────────────────

@dataclass
class _DefineLoopResult:
    attempt: Attempt | None
    repair_attempts: int
    exit_reason: str  # "clean_define" | "max_repairs_exhausted" | "max_steps_reached"
    last_error: str | None = None


async def _run_define_loop(
    *,
    task: Task,
    exploration_result: ExplorationResult,
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,             # SYSTEM_PROMPT (B4) or SYSTEM_PROMPT_B5
    max_steps: int,
    max_repairs: int,
    max_tokens: int,
    temperature: float,
    extra_body: dict | None,
    initial_warnings: list[str],   # sticky for whole loop (e.g. train_feedback)
    iter_label: int,                # 0 for Phase 1; 1..MAX_REFINEMENTS for refinement
    phase_label: str,               # "phase1" or "refinement"
    trace: list[TraceEntry],        # SHARED mutable
    usage: dict,                    # SHARED mutable
    log_fn: Callable[[str], None],
) -> _DefineLoopResult:
    """Step the model up to `max_steps` times; on a clean define, return the
    Attempt. On execution error, repair (up to `max_repairs`). On no tool call
    in a step, continue silently.

    Warnings are passed to `build_definer_messages` as an ordered list:
    `[pending_exec_error?, *initial_warnings, urgency_if_low?]`. The renderer
    joins them into one trailing user message.
    """
    pending_error: str | None = None
    repair_attempts = 0
    last_error: str | None = None

    for step in range(max_steps):
        log_fn(f"  [{phase_label} iter={iter_label}] Step {step + 1}/{max_steps}")

        warnings: list[str] = []
        if pending_error:
            warnings.append(pending_error)
        warnings.extend(initial_warnings)
        remaining = max_steps - step
        if remaining <= 2:
            warnings.append(_remaining_steps_message(remaining, max_steps))

        pending_error = None  # consumed for this step

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(build_definer_messages(
            task, exploration_result, trace, warnings=warnings,
        ))

        response = await with_retry(
            lambda: client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
            ),
            attempts=3, base_delay=1.5, label=f"definer.{phase_label}",
        )

        _accumulate_usage(usage, response)
        definition = _parse_tool_calls(response, trace, log_fn=log_fn)

        if definition is None:
            # No tool call this step — silently continue.
            continue

        code = definition.get("code", "") or ""
        log_fn(
            f"  🔧 Executing transform() against {task.num_test} test + "
            f"{task.num_train} train pair(s)..."
        )
        test_results, first_test_error = _exec_on_test_pairs(code, task)

        if first_test_error is not None:
            repair_attempts += 1
            last_error = first_test_error
            log_fn(
                f"  ❌ Execution error (repair {repair_attempts}/{max_repairs}): "
                f"{first_test_error[:120]}"
            )
            if repair_attempts >= max_repairs:
                return _DefineLoopResult(
                    attempt=None,
                    repair_attempts=repair_attempts,
                    exit_reason="max_repairs_exhausted",
                    last_error=first_test_error,
                )
            pending_error = (
                f"Your transform() raised an error:\n```\n{first_test_error}\n```\n"
                "Fix the code and call `define_transformation` again."
            )
            continue

        # Clean test execution — also run on train pairs for the Attempt record.
        train_results = _exec_on_train_pairs(code, task)
        train_num_correct = sum(1 for tr in train_results if tr.correct)
        num_test_correct = sum(1 for tr in test_results if tr.correct)
        log_fn(
            f"  ✅ Code executed cleanly — {num_test_correct}/{len(test_results)} test correct, "
            f"{train_num_correct}/{len(train_results)} train correct"
        )

        attempt = Attempt(
            iter=iter_label,
            phase=phase_label,
            code=code,
            transformation_summary=definition.get("transformation_summary", ""),
            reasoning=definition.get("reasoning", ""),
            test_results=test_results,
            train_results=train_results,
            train_num_correct=train_num_correct,
            train_num_total=len(train_results),
            final_error=None,
        )
        return _DefineLoopResult(
            attempt=attempt,
            repair_attempts=repair_attempts,
            exit_reason="clean_define",
            last_error=None,
        )

    return _DefineLoopResult(
        attempt=None,
        repair_attempts=repair_attempts,
        exit_reason="max_steps_reached",
        last_error=last_error,
    )


# ────────────────────────────────────────────────────────────────────
# Public driver
# ────────────────────────────────────────────────────────────────────

async def define_transformation(
    task: Task,
    exploration_result: ExplorationResult,
    client: AsyncOpenAI,
    model: str,
    max_steps: int = 10,
    max_repairs: int = 3,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    log_fn: Optional[Callable[[str], None]] = None,
    extra_body: Optional[dict] = None,
    enable_refinement: bool = True,
) -> TransformationResult:
    """Run the transformation definer.

    Phase 1 produces an initial define + executes (with B4 repair behavior).
    If `enable_refinement` is True and Phase 1 produced an attempt with
    train < 100%, Phase 2 runs up to MAX_REFINEMENTS refinement iterations.

    Top-level `TransformationResult` fields (`code`, `test_results`, etc.) are
    populated from the **best-by-train-score** attempt across the trajectory
    for backward compat with B4-era readers. The full attempt history lives
    on `.attempts` and is what the M-selection layer should consult.
    """
    _log = log_fn or _noop
    trace: list[TraceEntry] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    attempts: list[Attempt] = []
    total_repair_attempts = 0

    # Pick the prompt version: B5 (refinement-aware) if enable_refinement is on;
    # B4 (no refinement contract) otherwise. Picked once so both phases use the
    # same prompt — Phase 1 still benefits from the model knowing about the
    # downstream refinement opportunity (e.g. it can be less paranoid about
    # corner cases in its first attempt).
    system_prompt = SYSTEM_PROMPT_B5 if enable_refinement else SYSTEM_PROMPT

    # ── Phase 1 ──
    p1 = await _run_define_loop(
        task=task, exploration_result=exploration_result,
        client=client, model=model, system_prompt=system_prompt,
        max_steps=max_steps, max_repairs=max_repairs,
        max_tokens=max_tokens, temperature=temperature, extra_body=extra_body,
        initial_warnings=[],
        iter_label=0, phase_label="phase1",
        trace=trace, usage=usage, log_fn=_log,
    )
    total_repair_attempts += p1.repair_attempts

    if p1.attempt is not None:
        attempts.append(p1.attempt)

    # Determine if we run Phase 2.
    run_phase2 = (
        enable_refinement
        and p1.attempt is not None
        and p1.attempt.train_num_total > 0
        and p1.attempt.train_num_correct < p1.attempt.train_num_total
    )

    # ── Phase 2 ──
    if run_phase2:
        for refine_iter in range(1, MAX_REFINEMENTS + 1):
            last_attempt = attempts[-1]
            if last_attempt.train_score >= 1.0:
                break

            failing_pairs = _failing_train_pairs_payload(last_attempt)
            train_feedback = render_train_feedback(
                failing_pairs, iteration=refine_iter, max_iters=MAX_REFINEMENTS,
            )

            _log(f"🔁 Refinement iter {refine_iter}/{MAX_REFINEMENTS} (last train score "
                 f"{last_attempt.train_num_correct}/{last_attempt.train_num_total})")

            result = await _run_define_loop(
                task=task, exploration_result=exploration_result,
                client=client, model=model, system_prompt=system_prompt,
                max_steps=MAX_REFINEMENT_SUBSTEPS, max_repairs=max_repairs,
                max_tokens=max_tokens, temperature=temperature, extra_body=extra_body,
                initial_warnings=[train_feedback],
                iter_label=refine_iter, phase_label="refinement",
                trace=trace, usage=usage, log_fn=_log,
            )
            total_repair_attempts += result.repair_attempts

            if result.attempt is not None:
                attempts.append(result.attempt)
                continue

            # Graceful retry: if this iter produced no Attempt at all, nudge once
            # and rerun the substep loop. Catches the "model emitted text instead
            # of tools across all substeps" case.
            _log(f"  ⚠️  refinement iter {refine_iter} produced no clean define "
                 f"(reason={result.exit_reason}); retrying once with explicit nudge.")
            nudge = (
                "You must call either the `think` or `define_transformation` tool — "
                "plain text responses are not accepted. Submit your refined "
                "transformation via `define_transformation` now."
            )
            retry = await _run_define_loop(
                task=task, exploration_result=exploration_result,
                client=client, model=model, system_prompt=system_prompt,
                max_steps=MAX_REFINEMENT_SUBSTEPS, max_repairs=max_repairs,
                max_tokens=max_tokens, temperature=temperature, extra_body=extra_body,
                initial_warnings=[train_feedback, nudge],
                iter_label=refine_iter, phase_label="refinement",
                trace=trace, usage=usage, log_fn=_log,
            )
            total_repair_attempts += retry.repair_attempts
            if retry.attempt is not None:
                attempts.append(retry.attempt)
            else:
                _log(f"  🛑 retry also produced no clean define; exiting Phase 2.")
                break

    # ── Assemble final TransformationResult from best-by-train attempt ──
    if attempts:
        best = max(attempts, key=lambda a: (a.train_score, a.iter))
        return TransformationResult(
            task_id=task.task_id,
            transformation_summary=best.transformation_summary,
            reasoning=best.reasoning,
            code=best.code,
            trace=trace,
            repair_attempts=total_repair_attempts,
            max_repairs=max_repairs,
            final_error=None,
            test_results=best.test_results,
            train_num_correct=best.train_num_correct,
            train_num_total=best.train_num_total,
            attempts=attempts,
            usage=usage,
        )

    # Degenerate path: no clean define ever produced.
    _log(f"  ⏱️  No clean define produced (Phase 1 exit: {p1.exit_reason})")
    return TransformationResult(
        task_id=task.task_id,
        trace=trace,
        repair_attempts=total_repair_attempts,
        max_repairs=max_repairs,
        final_error=p1.last_error or "Phase 1 produced no clean define",
        attempts=[],
        usage=usage,
    )
