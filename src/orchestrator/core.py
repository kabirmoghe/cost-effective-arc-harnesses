"""B8 — Reflective Orchestrator agentic loop.

Single agentic loop with all 5 tools available throughout. The agent decides
every action; we enforce only cost-containment fuses and the structural
train==1.0 short-circuit. Reuses pipeline primitives (code execution, attempt
construction, train-feedback rendering, with_retry) via library imports.

Exit paths in priority order:
  1. Submission with train_score == 1.0 → "perfect_train"
  2. Agent calls done(reason) → "done"
  3. Iteration / exec-error fuses → "max_iterations" / "consecutive_exec_errors"
  4. Two consecutive no-tool-call responses → "no_tool_call"
  5. Budget exhausted with no clean attempt → "no_clean_define"
"""

import json
from dataclasses import dataclass
from typing import Callable, Optional

from openai import AsyncOpenAI

from shared.llm import with_retry
from shared.types import Task

from pipeline.agents.transformation_definer.core import (
    _build_attempt_from_submission,
    _accumulate_usage,
    _failing_train_pairs_payload,
)
from pipeline.agents.transformation_definer.feedback import render_train_feedback
from pipeline.agents.transformation_definer.types import (
    Attempt, TraceEntry,
)
from pipeline.agents.pattern_explorer.types import ExplorationResult

from .types import OrchestratorResult
from .tools import ORCHESTRATOR_TOOLS
from .rendering import build_orchestrator_messages
from .prompts import SYSTEM_PROMPT_ORCHESTRATOR
from .spawn import spawn_focused_explorers, MAX_SPAWN_CALLS


_noop = lambda msg: None

# Cost-containment fuses (see orchestrator_spec.md). All are fuses, not behavioral
# knobs — the model should rarely bind them.
MAX_ITERATIONS = 15                  # think/define/submit/spawn each = 1; exec repairs free
MAX_CONSECUTIVE_EXEC_ERRORS = 5      # 5 consecutive broken submissions → fuse


# ────────────────────────────────────────────────────────────────────
# Tool-call parsing (orchestrator-specific; handles all 5 tool kinds)
# ────────────────────────────────────────────────────────────────────

@dataclass
class _ParsedToolCall:
    """`kind` ∈ {"think", "define_transformation",
                  "submit_refined_transformation",
                  "explore_new_patterns", "done", None}.
    `args` is the JSON-decoded arguments dict for action tools (None for think
    and no-tool-call)."""
    kind: str | None
    args: dict | None = None


def _parse_tool_calls(
    response,
    trace: list[TraceEntry],
    log_fn: Callable[[str], None] = _noop,
) -> _ParsedToolCall:
    """Extract the first action-shaped tool call; append every tool call
    (including think) to the trace; return what action was called.

    Trace entries store the full `args` dict so subsequent rendering can replay
    the prior tool call faithfully (incl. the actual code). See
    `feedback_trace_fidelity_in_agentic_replay` memory.
    """
    message = response.choices[0].message
    if not message.tool_calls:
        return _ParsedToolCall(kind=None)

    action: _ParsedToolCall | None = None
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
            if action is None:
                action = _ParsedToolCall(kind="define_transformation", args=args)

        elif name == "submit_refined_transformation":
            trace.append(TraceEntry(
                kind="submit_refined_transformation",
                content=args.get("what_changed", ""),
                args=dict(args),
            ))
            log_fn(f"    🔁 submit_refined_transformation: {args.get('what_changed', '')[:120]}")
            if action is None:
                action = _ParsedToolCall(kind="submit_refined_transformation", args=args)

        elif name == "explore_new_patterns":
            guidance = args.get("guidance", "")
            trace.append(TraceEntry(
                kind="explore_new_patterns",
                content=guidance,
                args=dict(args),
            ))
            log_fn(f"    🔭 explore_new_patterns: {guidance[:120]}")
            if action is None:
                action = _ParsedToolCall(kind="explore_new_patterns", args=args)

        elif name == "done":
            reason = args.get("reason", "")
            trace.append(TraceEntry(
                kind="done",
                content=reason,
                args=dict(args),
            ))
            log_fn(f"    🏁 done: {reason[:120]}")
            if action is None:
                action = _ParsedToolCall(kind="done", args=args)

    return action or _ParsedToolCall(kind="think" if message.tool_calls else None)


# ────────────────────────────────────────────────────────────────────
# Orchestrator-side helpers (transient user messages)
# ────────────────────────────────────────────────────────────────────

def _orchestrator_repair_message(error: str) -> str:
    """User message after a submission errored at exec, framed agent-neutrally
    (does not force the model back to the same tool)."""
    return (
        f"Your code raised an execution error:\n```\n{error}\n```\n"
        "Address this — you may refine your transformation code, define a fresh implementation, "
        "or spawn explorers if the error suggests your abstraction is wrong. "
        "Your choice depending on the nature of the error."
    )


def _orchestrator_train_feedback_message(attempt: Attempt) -> str:
    """User message surfacing per-train-pair results after a clean submission
    that did not solve all training pairs. Reuses the pipeline feedback renderer.
    """
    failing = _failing_train_pairs_payload(attempt)
    body = render_train_feedback(failing)
    header = (
        f"Your code executed cleanly and solved "
        f"{attempt.train_num_correct}/{attempt.train_num_total} training pairs "
        f"(train_score = {attempt.train_score:.2f}).\n\n"
    )
    return header + body


# ────────────────────────────────────────────────────────────────────
# Public driver
# ────────────────────────────────────────────────────────────────────

async def run_orchestrator(
    task: Task,
    exploration_result: ExplorationResult,
    client: AsyncOpenAI,
    model: str,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    log_fn: Optional[Callable[[str], None]] = None,
    extra_body: Optional[dict] = None,
) -> OrchestratorResult:
    """B8 — single agentic loop, all 5 tools available throughout."""
    _log = log_fn or _noop
    trace: list[TraceEntry] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    attempts: list[Attempt] = []
    spawn_calls: list[dict] = []

    iterations_used = 0
    consecutive_exec_errors = 0
    consecutive_no_tool_call = 0
    pending_warnings: list[str] = []
    exit_reason: str | None = None
    done_reason: str | None = None

    while iterations_used < MAX_ITERATIONS:
        _log(
            f"  [orchestrator iter={iterations_used + 1}/{MAX_ITERATIONS}, "
            f"attempts={len(attempts)}, spawns={len(spawn_calls)}/{MAX_SPAWN_CALLS}, "
            f"exec_errs={consecutive_exec_errors}/{MAX_CONSECUTIVE_EXEC_ERRORS}]"
        )

        messages = [{"role": "system", "content": SYSTEM_PROMPT_ORCHESTRATOR}]
        messages.extend(build_orchestrator_messages(
            task, exploration_result, trace, warnings=pending_warnings,
        ))
        pending_warnings = []

        response = await with_retry(
            lambda: client.chat.completions.create(
                model=model,
                messages=messages,
                tools=ORCHESTRATOR_TOOLS,
                tool_choice="auto",
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
            ),
            attempts=3, base_delay=1.5, label="orchestrator",
        )
        _accumulate_usage(usage, response)
        call = _parse_tool_calls(response, trace, log_fn=_log)

        # ── No tool call: warn once; second consecutive triggers fuse exit ──
        if call.kind is None:
            consecutive_no_tool_call += 1
            iterations_used += 1
            if consecutive_no_tool_call >= 2:
                _log(f"  🛑 fuse: two consecutive no-tool-call responses")
                exit_reason = "no_tool_call"
                break
            pending_warnings.append(
                "You did not call any tool in your last response. Plain-text "
                "responses are not accepted. Call a tool now — `think`, "
                "`define_transformation`, `submit_refined_transformation`, "
                "`explore_new_patterns`, or `done`."
            )
            continue
        consecutive_no_tool_call = 0

        # ── think: pure scratchpad; advances iteration counter ──
        if call.kind == "think":
            iterations_used += 1
            continue

        # ── done: agent-mediated exit ──
        if call.kind == "done":
            iterations_used += 1
            done_reason = (call.args or {}).get("reason", "")
            exit_reason = "done"
            break

        # ── explore_new_patterns: spawn with cap fuse ──
        if call.kind == "explore_new_patterns":
            iterations_used += 1
            guidance = (call.args or {}).get("guidance", "")
            if len(spawn_calls) >= MAX_SPAWN_CALLS:
                _log(f"  🚫 spawn budget exhausted ({MAX_SPAWN_CALLS}/{MAX_SPAWN_CALLS})")
                pending_warnings.append(
                    f"Spawn budget exhausted ({MAX_SPAWN_CALLS} calls used) — "
                    "You cannot not call `explore_new_patterns` again."
                    "proceed with current findings. You may still refine, "
                    "define fresh, or call `done`."
                )
                continue
            _log(f"  🔭 spawning focused explorers: {guidance[:120]}")
            try:
                new_docs = await spawn_focused_explorers(
                    task, guidance, client, model,
                    extra_body=extra_body, log_fn=_log,
                )
            except BaseException as e:
                _log(f"  ❌ spawn failed: {type(e).__name__}: {str(e)[:120]}")
                new_docs = []
            spawn_calls.append({
                "iteration": iterations_used,
                "guidance": guidance,
                "num_returned": len(new_docs),
            })
            for doc in new_docs:
                exploration_result.documents.append(doc)
                if doc.usage:
                    usage["prompt_tokens"] += doc.usage.get("prompt_tokens", 0)
                    usage["completion_tokens"] += doc.usage.get("completion_tokens", 0)
            # Anchor a user_message summarizing the spawn result.
            summary = (
                f"Spawned {len(new_docs)} focused explorer(s) with guidance: "
                f"\"{guidance[:200]}{'…' if len(guidance) > 200 else ''}\". "
                f"Their findings have been appended above (see PatternExplorer {len(exploration_result.documents) - len(new_docs) + 1} onward)."
            )
            trace.append(TraceEntry(kind="user_message", content=summary))
            continue

        # ── submission tools ──
        if call.kind in ("define_transformation", "submit_refined_transformation"):
            attempt, exec_error = _build_attempt_from_submission(
                iter_label=len(attempts),
                phase_label="orchestrator",
                args=call.args or {},
                task=task,
                submission_kind=call.kind,
                log_fn=_log,
            )
            if exec_error is not None:
                # Exec error does NOT consume iteration budget.
                consecutive_exec_errors += 1
                _log(
                    f"  ❌ exec error (consecutive {consecutive_exec_errors}/"
                    f"{MAX_CONSECUTIVE_EXEC_ERRORS}): {exec_error[:120]}"
                )
                if consecutive_exec_errors >= MAX_CONSECUTIVE_EXEC_ERRORS:
                    exit_reason = "consecutive_exec_errors"
                    break
                pending_warnings.append(_orchestrator_repair_message(exec_error))
                continue

            # Clean execution.
            consecutive_exec_errors = 0
            attempts.append(attempt)
            iterations_used += 1

            # Structural exit on perfect train.
            if attempt.train_num_total > 0 and attempt.train_score >= 1.0:
                _log(f"  ✅ perfect train ({attempt.train_num_correct}/{attempt.train_num_total}) — submitting")
                exit_reason = "perfect_train"
                break

            feedback = _orchestrator_train_feedback_message(attempt)
            trace.append(TraceEntry(kind="user_message", content=feedback))
            continue

        # Unreachable for well-formed responses; treat as think to keep loop moving.
        _log(f"  ⚠ unexpected tool kind: {call.kind!r}; treating as think")
        iterations_used += 1

    if exit_reason is None:
        exit_reason = "max_iterations"

    # ── Assemble result ──
    if attempts:
        best = max(attempts, key=lambda a: (a.train_score, a.iter))
        return OrchestratorResult(
            task_id=task.task_id,
            transformation_summary=best.transformation_summary,
            reasoning=best.reasoning,
            code=best.code,
            trace=trace,
            repair_attempts=0,
            max_repairs=MAX_CONSECUTIVE_EXEC_ERRORS,
            final_error=None,
            test_results=best.test_results,
            train_num_correct=best.train_num_correct,
            train_num_total=best.train_num_total,
            attempts=attempts,
            usage=usage,
            exit_reason=exit_reason,
            done_reason=done_reason,
            iterations_used=iterations_used,
            spawn_calls=spawn_calls,
        )

    _log(f"  ⏱️  No clean define produced (orchestrator exit: {exit_reason})")
    return OrchestratorResult(
        task_id=task.task_id,
        trace=trace,
        repair_attempts=0,
        max_repairs=MAX_CONSECUTIVE_EXEC_ERRORS,
        final_error=f"Orchestrator produced no clean define (exit: {exit_reason})",
        attempts=[],
        usage=usage,
        exit_reason="no_clean_define" if exit_reason == "max_iterations" else exit_reason,
        done_reason=done_reason,
        iterations_used=iterations_used,
        spawn_calls=spawn_calls,
    )
