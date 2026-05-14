"""Context rendering helpers for the PatternExplorer agent."""

import json

from shared.types import Task
from shared.formatting import format_task_examples
from pipeline.agents.pattern_explorer.types import Pattern, TraceEntry


def _format_patterns_section(patterns: list[Pattern]) -> str:
    if not patterns:
        return "Patterns you've noted so far: (none yet)"
    lines = ["Patterns you've noted so far:"]
    for p in patterns:
        lines.append(f"  {p.id}. {p.text}")
    return "\n".join(lines)


def _build_trace_messages(trace: list[TraceEntry]) -> list[dict]:
    """Convert trace entries into assistant tool_call / tool response pairs."""
    messages = []
    for i, entry in enumerate(trace):
        call_id = f"trace_{i}"

        if entry.kind == "think":
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "think",
                        "arguments": json.dumps({"thought": entry.content}),
                    },
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": "Noted.",
            })

        elif entry.kind == "note_pattern":
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "note_pattern",
                        "arguments": json.dumps({"pattern": entry.content}),
                    },
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": f"Pattern #{entry.pattern_id} recorded.",
            })

    return messages


def build_explorer_messages(
    task: Task,
    patterns: list[Pattern],
    trace: list[TraceEntry],
    warning: str | None = None,
) -> list[dict]:
    """Build the full message list (excluding system prompt) for the explorer loop."""
    user_content = "\n\n".join([
        "Analyze the following task training examples and discover the transformation pattern.",
        format_task_examples(task),
        _format_patterns_section(patterns),
    ])

    messages = [{"role": "user", "content": user_content}]
    messages.extend(_build_trace_messages(trace))

    if warning:
        messages.append({"role": "user", "content": f"⚠️ {warning}"})

    return messages
