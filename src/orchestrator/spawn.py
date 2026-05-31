"""B8 — spawn focused PatternExplorers mid-loop from the Reflective Orchestrator.

The orchestrator's `explore_new_patterns(guidance)` tool calls this module to
spin up `SPAWN_K` parallel explorers with:
  - The FOCUSED system prompt (acknowledges mid-pipeline context, treats the
    upstream guidance as the anchor).
  - The orchestrator's guidance string injected as a fixed user message.
  - A tighter step budget (`SPAWN_MAX_STEPS = 5`, vs. default 10 for initial
    explorers) — focused scope, faster convergence, lower per-spawn cost.

Returns a list of `PatternDocument`s. The orchestrator appends these to its
active `ExplorationResult.documents` so subsequent rendering calls automatically
include the new findings in the orchestrator's context.

Hard cap on spawn-call frequency (`MAX_SPAWN_CALLS = 3`) is enforced by the
orchestrator loop in `core.py`, NOT here.
"""

import asyncio
from typing import Callable, Optional

from openai import AsyncOpenAI

from shared.types import Task
from pipeline.agents.pattern_explorer.core import explore_patterns
from pipeline.agents.pattern_explorer.types import PatternDocument
from orchestrator.prompts import SYSTEM_PROMPT_FOCUSED

# Constants (also referenced by the orchestrator loop for telemetry/fuse logic).
SPAWN_K = 2                  # explorers per spawn call
SPAWN_MAX_STEPS = 5          # tighter than the default 10 for initial explorers
MAX_SPAWN_CALLS = 3          # per definer; fuse enforced by orchestrator

_noop = lambda msg: None


async def spawn_focused_explorers(
    task: Task,
    guidance: str,
    client: AsyncOpenAI,
    model: str,
    *,
    k: int = SPAWN_K,
    max_steps: int = SPAWN_MAX_STEPS,
    temperature: float = 0.5,
    extra_body: Optional[dict] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> list[PatternDocument]:
    """Run K focused explorers in parallel with the given guidance string.

    Returns a list of `PatternDocument`s (one per surviving explorer). If all
    explorers fail, returns an empty list — the orchestrator can decide to
    `done` or continue with current findings.
    """
    log = log_fn or _noop

    async def _run_one(idx: int) -> PatternDocument:
        def prefixed_log(msg: str):
            log(f"  [focused-explorer {idx}]{msg}")
        return await explore_patterns(
            task, client, model,
            max_steps=max_steps,
            temperature=temperature,
            log_fn=prefixed_log,
            extra_body=extra_body,
            system_prompt=SYSTEM_PROMPT_FOCUSED,
            guidance=guidance,
        )

    raw = await asyncio.gather(
        *[_run_one(i) for i in range(k)],
        return_exceptions=True,
    )
    docs: list[PatternDocument] = []
    for i, r in enumerate(raw):
        if isinstance(r, BaseException):
            log(f"  [focused-explorer {i}] ❌ failed: {type(r).__name__}: {str(r)[:120]}")
            continue
        docs.append(r)
    log(f"  🔭 spawn complete: {len(docs)}/{k} focused explorers returned")
    return docs
