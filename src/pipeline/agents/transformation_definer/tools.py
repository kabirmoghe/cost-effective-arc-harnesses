"""Tool definitions for the TransformationDefiner.

Three tools and three named subsets:

  - `think`: record reasoning (used in both Phase 1 and Phase 2).
  - `define_transformation`: initial submission (Phase 1 only).
  - `submit_refined_transformation`: refinement submission (B5b Phase 2 only).
    Takes the same `code` field plus a required `what_changed` field that
    forces the model to articulate the delta from its previous submission —
    keeps refinement framing surgical rather than re-derivative.

Subsets (immutable per phase/substep, swapped via the `tools=` API parameter):

  - PHASE1_TOOLS: Phase 1 step loop. `[think, define_transformation]`.
  - PHASE2_TOOLS: B5b refinement substep 1. `[think, submit_refined_transformation]`.
    `define_transformation` is intentionally absent so the model commits to the
    refinement tool.
  - PHASE2_FORCE_TOOLS: B5b refinement substep 2 (forced commit). The think
    tool is dropped so the model has only one path forward.

`TOOL_DEFINITIONS` is kept for backward compatibility with B4/B5 callers that
load the full set unconditionally; it equals PHASE1_TOOLS.
"""

_THINK = {
    "type": "function",
    "function": {
        "name": "think",
        "description": "Record a reasoning step. Use this to analyze the explorers' findings, compare candidate transformations, identify consensus, and plan your implementation before writing code.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "Your reasoning or analysis.",
                }
            },
            "required": ["thought"],
        },
    },
}

_DEFINE_TRANSFORMATION = {
    "type": "function",
    "function": {
        "name": "define_transformation",
        "description": "Once maximum confidence has been reached, define a transformation that describes the current task's examples.",
        "parameters": {
            "type": "object",
            "properties": {
                "transformation_summary": {
                    "type": "string",
                    "description": "A brief description of the transformation.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Concise reasoning for why this was the transformation that was defined, including references from PatternExplorer agents, patterns, etc.",
                },
                "code": {
                    "type": "string",
                    "description": "Self-contained Python code defining `def transform(grid: list[list[int]]) -> list[list[int]]`. No external imports beyond the standard library. The grid is a 2D list of ints 0-9."
                },
            },
            "required": ["transformation_summary", "reasoning", "code"],
        },
    },
}

_SUBMIT_REFINED_TRANSFORMATION = {
    "type": "function",
    "function": {
        "name": "submit_refined_transformation",
        "description": "Submit a refined version of your previous transformation. Use this in refinement mode only — make surgical changes to your previous code based on the failure feedback, do not re-derive from scratch.",
        "parameters": {
            "type": "object",
            "properties": {
                "what_changed": {
                    "type": "string",
                    "description": "One or two sentences describing the specific delta from your previous code — name the bug you are fixing. Forces you to articulate the surgical change rather than rewriting.",
                },
                "code": {
                    "type": "string",
                    "description": "Self-contained Python code defining `def transform(grid: list[list[int]]) -> list[list[int]]`. No external imports beyond the standard library. The grid is a 2D list of ints 0-9.",
                },
            },
            "required": ["what_changed", "code"],
        },
    },
}


PHASE1_TOOLS = [_THINK, _DEFINE_TRANSFORMATION]
PHASE2_TOOLS = [_THINK, _SUBMIT_REFINED_TRANSFORMATION]
PHASE2_FORCE_TOOLS = [_SUBMIT_REFINED_TRANSFORMATION]

# Ablation A (act-only / ReAct-ablated): `think` removed from both phases. Only
# the action tool is exposed at each step, so the model has no scratchpad
# capability and must reason inside the action tool's structured fields (or not
# at all). Phase 2 collapses to a single tool, matching what PHASE2_FORCE_TOOLS
# is in the standard variant.
PHASE1_TOOLS_ACT_ONLY = [_DEFINE_TRANSFORMATION]
PHASE2_TOOLS_ACT_ONLY = [_SUBMIT_REFINED_TRANSFORMATION]

# Backward compat: B4/B5 callers expect TOOL_DEFINITIONS to be the full set.
# We point it at PHASE1_TOOLS so existing call sites continue to work.
TOOL_DEFINITIONS = PHASE1_TOOLS
