"""PatternExplorer agent — iterative pattern discovery loop (async)."""

import json
from typing import Callable, Optional
from openai import AsyncOpenAI

from shared.llm import with_retry
from shared.types import Task
from .types import Pattern, TraceEntry, PatternDocument, ExplorationResult
from .context.prompts import (
    SYSTEM_PROMPT,
    SYNTHESIS_PROMPT,
)
from .tools import TOOL_DEFINITIONS
from .context.rendering import build_explorer_messages

# Default no-op logger
_noop = lambda msg: None


def _parse_tool_calls(
    response,
    trace: list[TraceEntry],
    patterns: list[Pattern] = None,
    log_fn: Callable[[str], None] = _noop,
):
    """Extract tool calls from the response and update patterns/trace in place."""
    message = response.choices[0].message
    if not message.tool_calls:
        return

    for tc in message.tool_calls:
        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            continue

        if tc.function.name == "think":
            thought = args.get("thought", "")
            trace.append(TraceEntry(kind="think", content=thought))
            log_fn(f"    💭 think: {thought[:100]}...")

        elif tc.function.name == "note_pattern":
            pattern_text = args.get("pattern", "")
            pattern_id = len(patterns) + 1
            pattern = Pattern(id=pattern_id, text=pattern_text)
            patterns.append(pattern)
            trace.append(TraceEntry(
                kind="note_pattern",
                content=pattern_text,
                pattern_id=pattern_id,
            ))
            log_fn(f"    📌 pattern #{pattern_id}: {pattern_text[:100]}...")


_SAVE_CONTEXT = False

def _save_context(messages, path="agent_context.txt"):
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

def _accumulate_usage(usage: dict, response):
    """Add response token usage to running totals."""
    if response.usage:
        usage["prompt_tokens"] += response.usage.prompt_tokens
        usage["completion_tokens"] += response.usage.completion_tokens


async def explore_patterns(
    task: Task,
    client: AsyncOpenAI,
    model: str,
    max_steps: int = 10,
    min_exploration_steps: int = 3,
    pattern_threshold: int = 3,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    log_fn: Optional[Callable[[str], None]] = None,
    extra_body: Optional[dict] = None,
) -> PatternDocument:
    """Run the pattern exploration loop on a task.

    Returns:
        PatternDocument with discovered patterns, trace, and synthesis.
    """
    _log = log_fn or _noop
    trace: list[TraceEntry] = []
    patterns: list[Pattern] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0}

    for step in range(max_steps):
        if (
            len(patterns) >= pattern_threshold
            and step >= min_exploration_steps
        ):
            _log(f"  ⏩ Early exit: {len(patterns)} patterns found after {step} steps")
            break

        warning = None
        remaining = max_steps - step
        if remaining <= 2:
            warning = f"You have {remaining} exploration step(s) remaining. Start converging on your findings."

        _log(f"  Step {step + 1}/{max_steps} ({len(patterns)} patterns so far)")

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(build_explorer_messages(task, patterns, trace, warning=warning))
        _save_context(messages)

        response = await with_retry(
            lambda: client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",  # "required" is unsupported by AtlasCloud FP8; "auto" works, system prompt + tools reliably steer to a call, and _parse_tool_calls handles a no-tool-call step gracefully.
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
            ),
            attempts=3, base_delay=1.5, label=f"explorer.step",
        )

        _accumulate_usage(usage, response)
        _parse_tool_calls(response, trace, patterns, log_fn=_log)

    # Synthesis phase
    _log(f"  🔬 Synthesizing ({len(patterns)} patterns, {len(trace)} trace entries)...")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(build_explorer_messages(task, patterns, trace))
    messages.append({"role": "user", "content": SYNTHESIS_PROMPT})
    _save_context(messages)
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )

    _accumulate_usage(usage, response)
    synthesis = response.choices[0].message.content or ""
    _log(f"  ✅ Done — {len(patterns)} patterns, {usage['prompt_tokens']+usage['completion_tokens']} total tokens")

    return PatternDocument(
        task_id=task.task_id,
        patterns=patterns,
        trace=trace,
        synthesis=synthesis,
        usage=usage,
    )


async def run_parallel_explorers(
    task: Task,
    client: AsyncOpenAI,
    model: str,
    num_explorers: int = 3,
    **kwargs,
) -> ExplorationResult:
    """Run N explorers in parallel on one task, return combined result."""
    import asyncio

    # Each explorer gets a prefixed logger
    base_log = kwargs.pop("log_fn", None) or _noop

    async def _run_one(idx: int) -> PatternDocument:
        def prefixed_log(msg: str):
            base_log(f"  [explorer {idx}]{msg}")
        return await explore_patterns(
            task, client, model, log_fn=prefixed_log, **kwargs
        )

    raw = await asyncio.gather(
        *[_run_one(i) for i in range(num_explorers)],
        return_exceptions=True,
    )
    docs = []
    for i, r in enumerate(raw):
        if isinstance(r, BaseException):
            base_log(f"  [explorer {i}] ❌ failed after retries: {type(r).__name__}: {str(r)[:120]}")
            continue
        docs.append(r)
    if not docs:
        # All explorers failed; surface the first exception so the task is
        # flagged 💥 rather than silently producing an empty exploration.
        raise next(r for r in raw if isinstance(r, BaseException))

    return ExplorationResult(
        task_id=task.task_id,
        documents=docs,
    )
