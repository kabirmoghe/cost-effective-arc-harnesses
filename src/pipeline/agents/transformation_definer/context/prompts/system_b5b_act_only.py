"""B5b act-only system prompt — B5b refinement contract with `think` removed.

Used only for Phase 2 of the act-only ablation. Mirrors `system_b5b.py` but
imports the act-only Phase 1 prompt as the base body (so the trailing
refinement section sits on top of a think-stripped prompt) and rewrites the
refinement contract to remove the "use `think` sparingly" guidance.
"""

from .system_act_only import SYSTEM_PROMPT_ACT_ONLY as _B4_PROMPT_ACT_ONLY

_REFINEMENT_SECTION = """

---

Refinement mode:
- You have submitted a transformation via `define_transformation`. That submission executed cleanly but did not solve all training pairs. You are now in refinement mode.
- You now have the tool `submit_refined_transformation(code, what_changed)` to submit a refined version of your previous code. The `what_changed` field should briefly articulate the specific delta from your previous code — one or two sentences, naming the bug you are fixing.
- Refine, do not re-derive. Your previous transformation performed modestly, so it is possible the underlying idea is at least partially right. Make surgical changes guided by the failure feedback below — do not rewrite from scratch as if starting fresh.
- Implement changes directly by calling `submit_refined_transformation`."""

SYSTEM_PROMPT_B5B_ACT_ONLY = _B4_PROMPT_ACT_ONLY + _REFINEMENT_SECTION
