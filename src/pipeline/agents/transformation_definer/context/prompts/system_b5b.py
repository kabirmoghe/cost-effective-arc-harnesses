"""B5b system prompt — bit-identical to B4 SYSTEM_PROMPT plus a trailing
'Refinement mode' section describing the `submit_refined_transformation` tool.

This prompt is only used for Phase 2 of B5b (substep 1 + substep 2). Phase 1
still uses the unmodified B4 SYSTEM_PROMPT, so Phase-1 behavior is bit-identical
between B4 and B5b. The swap happens at the next API call boundary when
refinement is triggered (no multi-system-message dependency).

The trailing section:
  - Acknowledges the prior `define_transformation` call so the trace makes sense
    to the model.
  - Describes the new `submit_refined_transformation(code, what_changed)` tool.
  - Frames the contract: refine, do not re-derive. Surgical edits, not a fresh
    re-derivation from explorer findings.
  - Encourages sparing use of `think` (≤1 call) — the failure feedback in the
    user message should usually be enough to localize the bug.
"""

# Re-use the B4 prompt body verbatim then append the refinement section. The
# import keeps the two prompts in sync — any future edit to the B4 prompt body
# flows through unchanged.
from .system import SYSTEM_PROMPT as _B4_PROMPT

_REFINEMENT_SECTION = """

---

Refinement mode:
- You have submitted a transformation via `define_transformation`. That submission executed cleanly but did not solve all training pairs. You are now in refinement mode.
- You now have the tool `submit_refined_transformation(code, what_changed)` to submit a refined version of your previous code. The `what_changed` field should briefly articulate the specific delta from your previous code — one or two sentences, naming the bug you are fixing.
- Refine, do not re-derive. Your previous transformation performed modestly, so it is possible underlying idea is at least partially right. Make surgical changes guided by the failure feedback below — do not rewrite from scratch as if starting fresh.
- Use `think` sparingly — at most once, briefly, to localize the bug.
- If you determine the fix is obvious immediately from the failure feedback, feel free to call `submit_refined_transformation` directly without thinking first."""

SYSTEM_PROMPT_B5B = _B4_PROMPT + _REFINEMENT_SECTION
