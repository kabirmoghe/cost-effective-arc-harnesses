"""Context rendering helpers for the TransformationDefiner agent."""

import json

from shared.types import Task
from shared.formatting import format_task_examples
from pipeline.agents.transformation_definer.types import TraceEntry
from pipeline.agents.pattern_explorer.types import ExplorationResult


def _format_exploration_findings(exploration_result: ExplorationResult) -> str:
    sections = ["Here are the findings from the PatternExplorer Agents:\n"]
    for i, doc in enumerate(exploration_result.documents):
        sections.append(f"## PatternExplorer {i + 1}")
        sections.append("")
        if doc.patterns:
            sections.append("### Patterns")
            for p in doc.patterns:
                sections.append(f"{p.id}. {p.text}")
            sections.append("")
        if doc.synthesis:
            sections.append("### Transformation Rule Analysis")
            sections.append(doc.synthesis)
            sections.append("")
    return "\n".join(sections)


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

        elif entry.kind == "define_transformation":
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "define_transformation",
                        "arguments": json.dumps({
                            "transformation_summary": entry.content,
                            "reasoning": "",
                            "code": "",
                        }),
                    },
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": "Transformation submitted for execution.",
            })

    return messages


def build_definer_messages(
    task: Task,
    exploration_result: ExplorationResult,
    trace: list[TraceEntry],
    warning: str | None = None,
) -> list[dict]:
    """Build the full message list (excluding system prompt) for the definer loop."""
    examples_content = "\n\n".join([
        "Here are the task's training examples:",
        format_task_examples(task),
    ])

    messages = [
        {"role": "user", "content": examples_content},
        {"role": "user", "content": _format_exploration_findings(exploration_result)},
    ]
    messages.extend(_build_trace_messages(trace))

    if warning:
        messages.append({"role": "user", "content": warning})

    return messages
