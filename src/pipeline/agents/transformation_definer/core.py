"""TransformationDefiner agent — synthesize explorer findings into executable code.

Two-phase architecture:

  **Phase 1** (initial define + execution repair): up to 10 step LLM loop using
  the B4 SYSTEM_PROMPT and `[think, define_transformation]` tools. On a clean
  execution, records an `Attempt` and exits. On execution error, increments
  `repair_attempts` and continues (up to 3 repairs). Bit-identical to B4 — no
  refinement contract in the prompt, no awareness of Phase 2.

  **Phase 2** (B5b train-feedback refinement): runs only if Phase 1 produced
  a clean Attempt with `train_score >= 0.5` AND `train_score < 1.0`. Strict
  2-substep budget, single iteration. The SYSTEM_PROMPT_B5B (B4 prompt + a
  trailing refinement contract section) is swapped in at the next API call.
  Tools list switches to `[think, submit_refined_transformation]` for substep
  1 and `[submit_refined_transformation]` only for substep 2 (forced).

  Threshold rationale (train_score >= 0.5): below this, the failure tends to
  indicate a wrong abstraction rather than a surgical bug; refinement-via-
  train-feedback is the wrong intervention. See
  `docs/qualitative_analysis.md` for the 0607ce86 (wrong abstraction) vs.
  1e97544e (surgical fix) examples that motivated this.

  Why the system prompt swap (rather than a single dual-purpose prompt): the
  B5-as-built attempt described refinement in the system prompt from the
  start, and the model engaged with feedback in `think` calls but never
  re-committed via `define_transformation` (zero clean refined attempts
  across 873 definers). Keeping refinement out of Phase 1's awareness — and
  introducing it only when failure context is present — is the B5b
  hypothesis.

The B5 path (prompt-aware-from-start, 2 iters × 5 substeps) is intentionally
NOT implemented here. Use git history if you need to revisit it.
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
from .context.prompts import (
    SYSTEM_PROMPT, SYSTEM_PROMPT_B5B,
    SYSTEM_PROMPT_ACT_ONLY, SYSTEM_PROMPT_B5B_ACT_ONLY,
)
from .context.rendering import build_definer_messages
from .tools import (
    PHASE1_TOOLS, PHASE2_TOOLS, PHASE2_FORCE_TOOLS,
    PHASE1_TOOLS_ACT_ONLY, PHASE2_TOOLS_ACT_ONLY,
)

_noop = lambda msg: None

# B5b thresholds (locked 2026-05-26)
REFINEMENT_TRIGGER_MIN_TRAIN = 0.5     # only refine if train_score >= this
PHASE2_MAX_ITERS = 1                    # single refinement iteration
PHASE2_MAX_STEPS = 2                    # think OR submit substeps; exec errors do NOT burn budget
PHASE2_MAX_REPAIRS = 3                  # safety cap on consecutive exec-error retries


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _remaining_steps_message(remaining: int, total: int) -> str:
    """Graduated urgency message for Phase 1 step budget. `remaining` = steps
    left INCLUDING the current one."""
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


@dataclass
class _ParsedToolCall:
    """What came back from one model response.

    `kind` ∈ {"think", "define_transformation", "submit_refined_transformation", None}.
    `args` is the JSON-decoded arguments dict for the submission tools (None
    for think and no-tool-call).
    """
    kind: str | None
    args: dict | None = None


def _parse_tool_calls(
    response,
    trace: list[TraceEntry],
    log_fn: Callable[[str], None] = _noop,
) -> _ParsedToolCall:
    """Extract the first submission-shaped tool call; append think and
    submission entries to the trace; return what was called.

    If both a think and a submission appear in the same response, the
    submission wins (we execute it). Think entries still land in the trace
    so the model sees its own reasoning replayed.

    Trace entries for submissions store the full `args` dict so subsequent
    rendering can replay the prior tool call faithfully (incl. the actual
    code). See `feedback_trace_fidelity_in_agentic_replay` memory.
    """
    message = response.choices[0].message
    if not message.tool_calls:
        return _ParsedToolCall(kind=None)

    submission: _ParsedToolCall | None = None
    for tc in message.tool_calls:
        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            continue

        name = tc.function.name
        if name == "think":
            thought = args.get("thought", "")
            trace.append(TraceEntry(kind="think", content=thought))
            log_fn(f"    💭 think: {thought[:120]}")

        elif name == "define_transformation":
            trace.append(TraceEntry(
                kind="define_transformation",
                content=args.get("transformation_summary", ""),
                args=dict(args),
            ))
            log_fn(f"    🎯 define_transformation: {args.get('transformation_summary', '')[:120]}")
            if submission is None:
                submission = _ParsedToolCall(kind="define_transformation", args=args)

        elif name == "submit_refined_transformation":
            trace.append(TraceEntry(
                kind="submit_refined_transformation",
                content=args.get("what_changed", ""),
                args=dict(args),
            ))
            log_fn(f"    🔁 submit_refined_transformation: {args.get('what_changed', '')[:120]}")
            if submission is None:
                submission = _ParsedToolCall(kind="submit_refined_transformation", args=args)

    return submission or _ParsedToolCall(kind="think" if message.tool_calls else None)


def _accumulate_usage(usage: dict, response):
    if response.usage:
        usage["prompt_tokens"] += response.usage.prompt_tokens
        usage["completion_tokens"] += response.usage.completion_tokens


def _exec_on_test_pairs(code: str, task: Task) -> tuple[list[TestPairResult], str | None]:
    """Run the transform on all test pairs. Returns (results, first_error_if_any)."""
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
    """Run the transform on all train pairs; capture results including errors."""
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


def _build_attempt_from_submission(
    *,
    iter_label: int,
    phase_label: str,
    args: dict,
    task: Task,
    submission_kind: str,
    log_fn: Callable[[str], None],
) -> tuple[Attempt | None, str | None]:
    """Execute the submitted code on test + train and build an Attempt. Returns
    (attempt, exec_error_if_any). If exec error occurred, attempt is None."""
    code = args.get("code", "") or ""
    log_fn(
        f"  🔧 Executing transform() against {task.num_test} test + "
        f"{task.num_train} train pair(s)..."
    )
    test_results, first_test_error = _exec_on_test_pairs(code, task)
    if first_test_error is not None:
        return None, first_test_error

    train_results = _exec_on_train_pairs(code, task)
    train_num_correct = sum(1 for tr in train_results if tr.correct)
    num_test_correct = sum(1 for tr in test_results if tr.correct)
    log_fn(
        f"  ✅ Code executed cleanly — {num_test_correct}/{len(test_results)} test correct, "
        f"{train_num_correct}/{len(train_results)} train correct"
    )

    # Field naming differs across tools: define_transformation supplies
    # transformation_summary + reasoning; submit_refined_transformation
    # supplies what_changed. Normalize for the Attempt record.
    if submission_kind == "submit_refined_transformation":
        what_changed = args.get("what_changed", "")
        summary = what_changed
        reasoning = f"Refinement delta: {what_changed}"
    else:
        summary = args.get("transformation_summary", "")
        reasoning = args.get("reasoning", "")

    attempt = Attempt(
        iter=iter_label,
        phase=phase_label,
        code=code,
        transformation_summary=summary,
        reasoning=reasoning,
        test_results=test_results,
        train_results=train_results,
        train_num_correct=train_num_correct,
        train_num_total=len(train_results),
        final_error=None,
    )
    return attempt, None


# ────────────────────────────────────────────────────────────────────
# Phase 1 loop (bit-identical to B4)
# ────────────────────────────────────────────────────────────────────

@dataclass
class _Phase1Result:
    attempt: Attempt | None
    repair_attempts: int
    exit_reason: str  # "clean_define" | "max_repairs_exhausted" | "max_steps_reached"
    last_error: str | None = None


async def _run_phase1(
    *,
    task: Task,
    exploration_result: ExplorationResult,
    client: AsyncOpenAI,
    model: str,
    max_steps: int,
    max_repairs: int,
    max_tokens: int,
    temperature: float,
    extra_body: dict | None,
    trace: list[TraceEntry],
    usage: dict,
    log_fn: Callable[[str], None],
    definer_variant: str = "react",
) -> _Phase1Result:
    """B4 step loop: think + define_transformation; repair on exec error.

    `definer_variant`:
      - "react"    : standard B4/B5b (think + define_transformation)
      - "act_only" : Ablation A — only define_transformation in the tool schema,
                     and the act-only system prompt directs the model to submit
                     immediately. Step budget kept identical to preserve a
                     single-axis comparison.
    """
    if definer_variant == "act_only":
        system_prompt = SYSTEM_PROMPT_ACT_ONLY
        tools = PHASE1_TOOLS_ACT_ONLY
    else:
        system_prompt = SYSTEM_PROMPT
        tools = PHASE1_TOOLS

    pending_error: str | None = None
    repair_attempts = 0
    last_error: str | None = None

    for step in range(max_steps):
        log_fn(f"  [phase1 iter=0] Step {step + 1}/{max_steps}")

        warnings: list[str] = []
        if pending_error:
            warnings.append(pending_error)
        remaining = max_steps - step
        if remaining <= 2:
            warnings.append(_remaining_steps_message(remaining, max_steps))

        pending_error = None

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(build_definer_messages(
            task, exploration_result, trace, warnings=warnings,
        ))

        response = await with_retry(
            lambda: client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
            ),
            attempts=3, base_delay=1.5, label="definer.phase1",
        )

        _accumulate_usage(usage, response)
        call = _parse_tool_calls(response, trace, log_fn=log_fn)

        if call.kind != "define_transformation":
            continue

        attempt, exec_error = _build_attempt_from_submission(
            iter_label=0, phase_label="phase1",
            args=call.args, task=task,
            submission_kind="define_transformation",
            log_fn=log_fn,
        )
        if exec_error is None:
            return _Phase1Result(
                attempt=attempt, repair_attempts=repair_attempts,
                exit_reason="clean_define",
            )

        # Execution error — start repair cycle.
        repair_attempts += 1
        last_error = exec_error
        log_fn(
            f"  ❌ Execution error (repair {repair_attempts}/{max_repairs}): "
            f"{exec_error[:120]}"
        )
        if repair_attempts >= max_repairs:
            return _Phase1Result(
                attempt=None, repair_attempts=repair_attempts,
                exit_reason="max_repairs_exhausted", last_error=exec_error,
            )
        pending_error = (
            f"Your transform() raised an error:\n```\n{exec_error}\n```\n"
            "Fix the code and call `define_transformation` again."
        )

    return _Phase1Result(
        attempt=None, repair_attempts=repair_attempts,
        exit_reason="max_steps_reached", last_error=last_error,
    )


# ────────────────────────────────────────────────────────────────────
# Phase 2 (B5b): 2-substep refinement
# ────────────────────────────────────────────────────────────────────

PHASE2_STEP0_INSTRUCTION = (
    "Refine your code based on the failure feedback above. You may use `think` "
    "once briefly to localize the bug(s) if needed, then call "
    "`submit_refined_transformation` with your refined code."
)

# Act-only variant: no think tool available; direct to submission.
PHASE2_STEP0_INSTRUCTION_ACT_ONLY = (
    "Refine your code based on the failure feedback above and call "
    "`submit_refined_transformation` with your refined code in this response."
)

PHASE2_FORCE_NUDGE = (
    "You must now call the `submit_refined_transformation` tool with your refined "
    "code in this response. Plain-text responses are not accepted at this step."
)


def _phase2_repair_message(error: str) -> str:
    """User message when the most recent submission's code errored at execution.

    Repairs do not consume the substep budget — see `_run_phase2_b5b`.
    """
    return (
        f"Your refined code raised an execution error:\n```\n{error}\n```\n"
        "Fix and re-emit `submit_refined_transformation`."
    )


@dataclass
class _Phase2Result:
    attempt: Attempt | None
    steps_used: int          # thought-or-submitted substeps (errors not counted)
    repairs_used: int        # exec-error retries
    exit_reason: str         # "submit_clean" | "max_repairs_exhausted" | "step_budget_exhausted"


async def _run_phase2_b5b(
    *,
    task: Task,
    exploration_result: ExplorationResult,
    client: AsyncOpenAI,
    model: str,
    failure_feedback: str,
    max_tokens: int,
    temperature: float,
    extra_body: dict | None,
    trace: list[TraceEntry],
    usage: dict,
    log_fn: Callable[[str], None],
    definer_variant: str = "react",
) -> _Phase2Result:
    """B5b refinement phase. Loop with two counters:

      - `step` advances when the model thinks OR emits no tool call. Bounded
        by PHASE2_MAX_STEPS (= 2: one think + one forced submit).
      - `repairs` advances when the model submits code that errors at exec.
        Bounded by PHASE2_MAX_REPAIRS (safety cap on exec-error retries).
        Exec errors do NOT advance `step` — Phase 1 has already proven the
        code path can execute cleanly, so Phase 2 exec errors should be rare
        and we don't want to spend the substep budget on them.

    Tool availability is determined by `step`:
      - step == 0: PHASE2_TOOLS (think + submit_refined_transformation).
      - step >= 1: PHASE2_FORCE_TOOLS (submit_refined_transformation only).

    The failure feedback is inserted **once into the trace as a user_message
    entry before the loop begins**, so it stays anchored at a fixed position
    in the replayed conversation. Any new Phase 2 think/submit entries follow
    it naturally in order. Per-iteration `warnings` carry only the transient
    messages tied to the current turn (repair message or force nudge), which
    *should* sit at the end of the message list since they refer to the most
    recent user-turn context.

    When the model submits clean code, we return immediately with the Attempt.
    """
    # Anchor the failure feedback at a fixed trace position. Subsequent think
    # and submit entries appended to `trace` during this phase land after it
    # in the conversation history when `_build_trace_messages` replays them.
    trace.append(TraceEntry(kind="user_message", content=failure_feedback))

    # Variant-specific prompt + tool selection. Act-only collapses Phase 2 to a
    # single tool (no think), but the step-budget structure is preserved so the
    # comparison stays single-axis. `act_only_tools` is used at every step in
    # the act_only variant; standard variant swaps think out at step 1.
    if definer_variant == "act_only":
        system_prompt_p2 = SYSTEM_PROMPT_B5B_ACT_ONLY
        step0_instruction = PHASE2_STEP0_INSTRUCTION_ACT_ONLY
    else:
        system_prompt_p2 = SYSTEM_PROMPT_B5B
        step0_instruction = PHASE2_STEP0_INSTRUCTION

    step = 0
    repairs = 0
    pending_error: str | None = None

    while step < PHASE2_MAX_STEPS:
        if definer_variant == "act_only":
            tools = PHASE2_TOOLS_ACT_ONLY
            force_label = ""  # only one tool at every step; nothing to "force"
        else:
            tools = PHASE2_TOOLS if step == 0 else PHASE2_FORCE_TOOLS
            force_label = "" if step == 0 else " (forced — think dropped)"
        log_fn(
            f"  [phase2 step={step}/{PHASE2_MAX_STEPS - 1}, repairs={repairs}/{PHASE2_MAX_REPAIRS}]"
            f"{force_label}"
        )

        # Transient warnings only — repair-message OR step-0 instruction OR
        # force-nudge, never combined. The failure feedback itself lives in the
        # trace (above), not here.
        warnings: list[str] = []
        if pending_error is not None:
            warnings.append(_phase2_repair_message(pending_error))
            pending_error = None  # consumed for this step
        elif step == 0:
            warnings.append(step0_instruction)
        else:  # step > 0
            warnings.append(PHASE2_FORCE_NUDGE)

        messages = [{"role": "system", "content": system_prompt_p2}]
        messages.extend(build_definer_messages(
            task, exploration_result, trace, warnings=warnings,
        ))

        response = await with_retry(
            lambda: client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",  # AtlasCloud FP8 rejects "required"; rely on tool-list + nudge
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
            ),
            attempts=3, base_delay=1.5, label=f"definer.phase2.step{step}",
        )
        _accumulate_usage(usage, response)
        call = _parse_tool_calls(response, trace, log_fn=log_fn)

        if call.kind == "submit_refined_transformation":
            attempt, exec_error = _build_attempt_from_submission(
                iter_label=1, phase_label="refinement",
                args=call.args, task=task,
                submission_kind="submit_refined_transformation",
                log_fn=log_fn,
            )
            if exec_error is None:
                return _Phase2Result(
                    attempt=attempt, steps_used=step + 1, repairs_used=repairs,
                    exit_reason="submit_clean",
                )
            # Exec error: retry without burning step budget, up to PHASE2_MAX_REPAIRS.
            repairs += 1
            log_fn(f"  ❌ Submit errored (repair {repairs}/{PHASE2_MAX_REPAIRS}): {exec_error[:120]}")
            if repairs >= PHASE2_MAX_REPAIRS:
                return _Phase2Result(
                    attempt=None, steps_used=step + 1, repairs_used=repairs,
                    exit_reason="max_repairs_exhausted",
                )
            pending_error = exec_error
            continue  # do not advance `step`

        # think OR no tool call → burn the substep.
        step += 1

    log_fn(f"  🛑 Phase 2 step budget exhausted with no clean submission.")
    return _Phase2Result(
        attempt=None, steps_used=step, repairs_used=repairs,
        exit_reason="step_budget_exhausted",
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
    enable_refinement: bool = False,
    definer_variant: str = "react",
) -> TransformationResult:
    """Run the transformation definer.

    Phase 1 produces an initial define + executes (B4 behavior, bit-identical).
    If `enable_refinement` is True AND Phase 1 produced an attempt with
    train_score in [REFINEMENT_TRIGGER_MIN_TRAIN, 1.0), Phase 2 (B5b) runs:
    1 iteration, 2-substep budget, new tool, swapped system prompt.

    Top-level `TransformationResult` fields are populated from the
    **best-by-train-score** attempt across the trajectory (ties broken by later
    iter) — `attempts` carries the full history for the M-selection layer.
    """
    _log = log_fn or _noop
    trace: list[TraceEntry] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    attempts: list[Attempt] = []

    # ── Phase 1 ──
    p1 = await _run_phase1(
        task=task, exploration_result=exploration_result,
        client=client, model=model,
        max_steps=max_steps, max_repairs=max_repairs,
        max_tokens=max_tokens, temperature=temperature, extra_body=extra_body,
        trace=trace, usage=usage, log_fn=_log,
        definer_variant=definer_variant,
    )
    if p1.attempt is not None:
        attempts.append(p1.attempt)

    # ── Decide whether to run Phase 2 (B5b refinement) ──
    run_phase2 = (
        enable_refinement
        and p1.attempt is not None
        and p1.attempt.train_num_total > 0
        and p1.attempt.train_score >= REFINEMENT_TRIGGER_MIN_TRAIN
        and p1.attempt.train_score < 1.0
    )

    if enable_refinement and p1.attempt is not None and not run_phase2:
        reason = (
            "train_score=1.0 (perfect)"
            if p1.attempt.train_score >= 1.0
            else f"train_score={p1.attempt.train_score:.2f} < {REFINEMENT_TRIGGER_MIN_TRAIN} threshold"
        )
        _log(f"  ⏸️  Skipping Phase 2 — {reason}")

    # ── Phase 2 ──
    if run_phase2:
        failure_feedback = render_train_feedback(
            _failing_train_pairs_payload(p1.attempt),
        )
        _log(
            f"🔁 Phase 2 (B5b refinement) — Phase 1 train="
            f"{p1.attempt.train_num_correct}/{p1.attempt.train_num_total}"
        )
        p2 = await _run_phase2_b5b(
            task=task, exploration_result=exploration_result,
            client=client, model=model,
            failure_feedback=failure_feedback,
            max_tokens=max_tokens, temperature=temperature, extra_body=extra_body,
            trace=trace, usage=usage, log_fn=_log,
            definer_variant=definer_variant,
        )
        if p2.attempt is not None:
            attempts.append(p2.attempt)

    # ── Assemble final TransformationResult from best-by-train attempt ──
    if attempts:
        best = max(attempts, key=lambda a: (a.train_score, a.iter))
        return TransformationResult(
            task_id=task.task_id,
            transformation_summary=best.transformation_summary,
            reasoning=best.reasoning,
            code=best.code,
            trace=trace,
            repair_attempts=p1.repair_attempts,
            max_repairs=max_repairs,
            final_error=None,
            test_results=best.test_results,
            train_num_correct=best.train_num_correct,
            train_num_total=best.train_num_total,
            attempts=attempts,
            usage=usage,
        )

    # Degenerate: no clean define ever produced.
    _log(f"  ⏱️  No clean define produced (Phase 1 exit: {p1.exit_reason})")
    return TransformationResult(
        task_id=task.task_id,
        trace=trace,
        repair_attempts=p1.repair_attempts,
        max_repairs=max_repairs,
        final_error=p1.last_error or "Phase 1 produced no clean define",
        attempts=[],
        usage=usage,
    )
