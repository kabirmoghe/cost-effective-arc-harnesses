"""TransformationDefiner agent — synthesize explorer findings into executable code."""

import json
from typing import Callable, Optional
from openai import AsyncOpenAI

from shared.types import Task
from shared.code_exec import execute_transformation
from pipeline.agents.pattern_explorer.types import ExplorationResult
from .types import TraceEntry, TestPairResult, TransformationResult
from .context.prompts import SYSTEM_PROMPT
from .context.rendering import build_definer_messages
from .tools import TOOL_DEFINITIONS

_noop = lambda msg: None


_SAVE_CONTEXT = False

def _save_context(messages, path="definer_context.txt"):
    if not _SAVE_CONTEXT:
        return
    with open(path, "w") as f:
        for msg in messages:
            role = msg["role"]
            content = msg.get("content") or ""
            if msg.get("tool_calls"):
                parts = []
                for tc in msg["tool_calls"]:
                    name = tc["function"]["name"]
                    args = tc["function"]["arguments"]
                    parts.append(f"[{name}] {args}")
                content = "\n".join(parts)
            f.write(f"<{role}>\n{content}\n</{role}>\n\n")


def _parse_tool_calls(
    response,
    trace: list[TraceEntry],
    log_fn: Callable[[str], None] = _noop,
) -> dict | None:
    """Extract tool calls from the response. Returns define_transformation args if found."""
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
) -> TransformationResult:
    """Run the transformation definer loop.

    The agent thinks via the think tool, then emits define_transformation.
    On code execution failure, it gets the error and tries to repair.
    """
    _log = log_fn or _noop
    trace: list[TraceEntry] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    repair_attempts = 0
    pending_warning = None

    for step in range(max_steps):
        _log(f"  Step {step + 1}/{max_steps}")

        warning = pending_warning
        pending_warning = None

        remaining = max_steps - step
        if remaining <= 2:
            warning = f"You have {remaining} step(s) remaining. Submit your transformation now."

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(build_definer_messages(task, exploration_result, trace, warning=warning))
        _save_context(messages)

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="required",
            temperature=temperature,
            max_tokens=max_tokens,
        )

        _accumulate_usage(usage, response)
        definition = _parse_tool_calls(response, trace, log_fn=_log)

        if definition is None:
            continue

        code = definition.get("code", "")
        _log(f"  🔧 Executing transform() against {task.num_test} test input(s)...")

        test_results = []
        first_error = None
        for i in range(task.num_test):
            predicted, error = execute_transformation(code, task.test[i].input)
            if error is not None:
                first_error = first_error or error
                test_results.append(TestPairResult(
                    test_index=i, error=error,
                ))
            else:
                test_results.append(TestPairResult(
                    test_index=i,
                    predicted_output=predicted,
                    expected_output=task.test[i].output,
                    correct=predicted == task.test[i].output,
                ))

        if first_error is None:
            num_correct = sum(1 for r in test_results if r.correct)
            _log(f"  ✅ Code executed successfully — {num_correct}/{len(test_results)} correct")
            return TransformationResult(
                task_id=task.task_id,
                transformation_summary=definition.get("transformation_summary", ""),
                reasoning=definition.get("reasoning", ""),
                code=code,
                trace=trace,
                repair_attempts=repair_attempts,
                final_error=None,
                test_results=test_results,
                usage=usage,
            )

        repair_attempts += 1
        _log(f"  ❌ Execution error (repair {repair_attempts}/{max_repairs}): {first_error[:120]}")

        if repair_attempts >= max_repairs:
            _log(f"  🛑 Max repairs reached, returning with error")
            return TransformationResult(
                task_id=task.task_id,
                transformation_summary=definition.get("transformation_summary", ""),
                reasoning=definition.get("reasoning", ""),
                code=code,
                trace=trace,
                repair_attempts=repair_attempts,
                max_repairs=max_repairs,
                final_error=first_error,
                test_results=test_results,
                usage=usage,
            )

        pending_warning = (
            f"Your transform() raised an error:\n```\n{first_error}\n```\n"
            "Fix the code and call define_transformation again."
        )

    _log(f"  ⏱️ Max steps reached without successful transformation")
    return TransformationResult(
        task_id=task.task_id,
        trace=trace,
        repair_attempts=repair_attempts,
        max_repairs=max_repairs,
        final_error="Max steps reached without emitting define_transformation",
        usage=usage,
    )
