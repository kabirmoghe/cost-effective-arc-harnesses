"""B8 — Orchestrator message construction.

Mirrors `pipeline.agents.transformation_definer.context.rendering.build_definer_messages`
but adds trace-replay branches for the two orchestrator-only TraceEntry kinds:
`explore_new_patterns` and `done`. The body otherwise duplicates pipeline's
rendering logic — this is conscious code duplication chosen over a pipeline
extension hook (see D4 in the refactor plan).
"""

import json

from shared.types import Task
from shared.formatting import format_task_examples
from pipeline.agents.transformation_definer.types import TraceEntry
from pipeline.agents.pattern_explorer.types import ExplorationResult


def _format_exploration_findings(exploration_result: ExplorationResult) -> str:
    sections = ["Here are the findings from the initial PatternExplorer Agent fleet:\n"]
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
    """Convert trace entries into assistant tool_call / tool response pairs.
    Handles the 4 pipeline-shared kinds (think, define_transformation,
    submit_refined_transformation, user_message) PLUS the 2 orchestrator-only
    kinds (explore_new_patterns, done).
    """
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
            args = entry.args or {
                "transformation_summary": entry.content,
                "reasoning": "",
                "code": "",
            }
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "define_transformation",
                        "arguments": json.dumps(args),
                    },
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": "Transformation submitted for execution.",
            })

        elif entry.kind == "submit_refined_transformation":
            args = entry.args or {
                "what_changed": entry.content,
                "code": "",
            }
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "submit_refined_transformation",
                        "arguments": json.dumps(args),
                    },
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": "Refined transformation submitted for execution.",
            })

        elif entry.kind == "explore_new_patterns":
            args = entry.args or {"guidance": entry.content}
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "explore_new_patterns",
                        "arguments": json.dumps(args),
                    },
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": "Focused explorers dispatched; findings appended to context.",
            })

        elif entry.kind == "done":
            args = entry.args or {"reason": entry.content}
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "done",
                        "arguments": json.dumps(args),
                    },
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": "Acknowledged.",
            })

        elif entry.kind == "user_message":
            messages.append({"role": "user", "content": entry.content})

    return messages


def build_orchestrator_messages(
    task: Task,
    exploration_result: ExplorationResult,
    trace: list[TraceEntry],
    warnings: list[str] | None = None,
) -> list[dict]:
    """Build the full message list (excluding system prompt) for one iteration
    of the orchestrator loop.

    `warnings` is an ordered list of transient user-directed notices, appended
    after the trace as a single user message (joined with blank lines). Mirrors
    pipeline's contract.
    """
    examples_content = "\n\n".join([
        "Here are the task's training examples:",
        format_task_examples(task),
    ])

    messages = [
        {"role": "user", "content": examples_content},
        {"role": "user", "content": _format_exploration_findings(exploration_result)},
    ]
    messages.extend(_build_trace_messages(trace))

    if warnings:
        joined = "\n\n".join(w for w in warnings if w)
        if joined:
            messages.append({"role": "user", "content": joined})

    return messages
