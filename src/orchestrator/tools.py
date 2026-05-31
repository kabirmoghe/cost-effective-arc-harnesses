"""B8 — Orchestrator toolset.

Reuses the 3 shared tools from `pipeline.agents.transformation_definer.tools`
(`_THINK`, `_DEFINE_TRANSFORMATION`, `_SUBMIT_REFINED_TRANSFORMATION`) and adds
two orchestrator-specific tools (`_EXPLORE_NEW_PATTERNS`, `_DONE`). Exports
`ORCHESTRATOR_TOOLS` — a single list passed unmodified throughout the loop.
"""

from pipeline.agents.transformation_definer.tools import (
    _THINK,
    _DEFINE_TRANSFORMATION,
    _SUBMIT_REFINED_TRANSFORMATION,
)


_EXPLORE_NEW_PATTERNS = {
    "type": "function",
    "function": {
        "name": "explore_new_patterns",
        "description": (
            "Spawn focused PatternExplorers mid-loop to investigate a specific question about the task. "
            "Use ONLY when your own reasoning has been exhausted and a fresh set of eyes would help. "
            "The most valuable `guidance` is NEGATIVE: anti-patterns you've ruled out, abstractions you've tried that failed, "
            "constraints you've confirmed. Positive hypotheses are also helpful when you have them — but if you had a confident "
            "new hypothesis, you'd usually just call `define_transformation` with it. Spawning is expensive; use sparingly."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "guidance": {
                    "type": "string",
                    "description": (
                        "Specific guidance for the focused explorers. Lead with what NOT to explore (dead ends from prior attempts "
                        "and confirmed constraints), then optionally describe new directions worth examining."
                    ),
                },
            },
            "required": ["guidance"],
        },
    },
}

_DONE = {
    "type": "function",
    "function": {
        "name": "done",
        "description": (
            "Deliberately exit the loop. Call when further iteration would not improve the result — either you've converged on a "
            "best attempt, you're stumped without a clear next move, or further exploration is unlikely to help. The best-by-train "
            "attempt across your trajectory will be selected."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "One sentence explaining why you're stopping — what you've concluded or what's blocking further progress.",
                },
            },
            "required": ["reason"],
        },
    },
}


ORCHESTRATOR_TOOLS = [
    _THINK,
    _DEFINE_TRANSFORMATION,
    _SUBMIT_REFINED_TRANSFORMATION,
    _EXPLORE_NEW_PATTERNS,
    _DONE,
]
