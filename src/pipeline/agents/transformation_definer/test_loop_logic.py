"""Smoketests for the definer Phase 1 + Phase 2 (B5b) loop logic.

The OpenAI client is mocked so these tests run instantly with no API cost.
Each test feeds canned chat-completions responses (think calls, define calls,
submit_refined_transformation calls, text-only responses) and asserts behavior
of `_run_phase1`, `_run_phase2_b5b`, and the public `define_transformation`
driver.

Run with:
    uv run pytest src/pipeline/agents/transformation_definer/test_loop_logic.py -v
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import pytest

from pipeline.agents.pattern_explorer.types import ExplorationResult, PatternDocument
from pipeline.agents.transformation_definer.core import (
    define_transformation,
    REFINEMENT_TRIGGER_MIN_TRAIN, PHASE2_MAX_ITERS,
    PHASE2_MAX_STEPS, PHASE2_MAX_REPAIRS,
)
from pipeline.agents.transformation_definer.context.prompts import (
    SYSTEM_PROMPT, SYSTEM_PROMPT_B5B,
)
from shared.types import Example, Task


# ────────────────────────────────────────────────────────────────────
# Mock OpenAI async client
# ────────────────────────────────────────────────────────────────────

@dataclass
class _MockFunc:
    name: str
    arguments: str


@dataclass
class _MockToolCall:
    function: _MockFunc
    id: str = "mock_call"
    type: str = "function"


@dataclass
class _MockMessage:
    tool_calls: list[_MockToolCall] | None = None
    content: str | None = None


@dataclass
class _MockChoice:
    message: _MockMessage


@dataclass
class _MockUsage:
    prompt_tokens: int = 100
    completion_tokens: int = 20


@dataclass
class _MockResponse:
    choices: list[_MockChoice]
    usage: _MockUsage | None = None


def _think_resp(thought: str = "thinking...") -> _MockResponse:
    tc = _MockToolCall(_MockFunc("think", json.dumps({"thought": thought})))
    return _MockResponse(
        choices=[_MockChoice(_MockMessage(tool_calls=[tc]))],
        usage=_MockUsage(),
    )


def _define_resp(code: str, summary: str = "test", reasoning: str = "") -> _MockResponse:
    tc = _MockToolCall(_MockFunc("define_transformation", json.dumps({
        "code": code, "transformation_summary": summary, "reasoning": reasoning,
    })))
    return _MockResponse(
        choices=[_MockChoice(_MockMessage(tool_calls=[tc]))],
        usage=_MockUsage(prompt_tokens=200, completion_tokens=50),
    )


def _refine_resp(code: str, what_changed: str = "fixed bug") -> _MockResponse:
    """A submit_refined_transformation call."""
    tc = _MockToolCall(_MockFunc("submit_refined_transformation", json.dumps({
        "code": code, "what_changed": what_changed,
    })))
    return _MockResponse(
        choices=[_MockChoice(_MockMessage(tool_calls=[tc]))],
        usage=_MockUsage(prompt_tokens=200, completion_tokens=50),
    )


def _text_resp(text: str = "I'm just thinking out loud") -> _MockResponse:
    """No tool call — simulates the model emitting plain text."""
    return _MockResponse(
        choices=[_MockChoice(_MockMessage(tool_calls=None, content=text))],
        usage=_MockUsage(prompt_tokens=50, completion_tokens=10),
    )


class _MockChatCompletions:
    def __init__(self, responses: list[_MockResponse]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise RuntimeError(
                f"Mock client ran out of canned responses (call #{len(self.calls)})"
            )
        return self.responses.pop(0)


class _MockChat:
    def __init__(self, completions):
        self.completions = completions


class MockAsyncClient:
    """Stand-in for openai.AsyncOpenAI with a finite queue of canned responses."""

    def __init__(self, responses: list[_MockResponse]):
        self.completions = _MockChatCompletions(responses)
        self.chat = _MockChat(self.completions)

    @property
    def call_count(self) -> int:
        return len(self.completions.calls)

    def last_messages(self) -> list[dict]:
        return self.completions.calls[-1]["messages"]

    def call_tools_names(self, call_idx: int) -> list[str]:
        """Tool names available on the i-th API call (0-indexed)."""
        tools = self.completions.calls[call_idx]["tools"]
        return [t["function"]["name"] for t in tools]

    def call_system(self, call_idx: int) -> str:
        return self.completions.calls[call_idx]["messages"][0]["content"]


# ────────────────────────────────────────────────────────────────────
# Fixtures: a tiny task with 2 train pairs + 1 test pair
# ────────────────────────────────────────────────────────────────────

def _make_task(task_id: str = "test_task") -> Task:
    return Task(
        task_id=task_id,
        train=[
            Example(input=[[0, 0], [1, 1]], output=[[0, 0], [1, 1]]),
            Example(input=[[2, 2], [3, 3]], output=[[2, 2], [3, 3]]),
        ],
        test=[
            Example(input=[[4, 4], [5, 5]], output=[[4, 4], [5, 5]]),
        ],
    )


def _make_exploration() -> ExplorationResult:
    return ExplorationResult(
        task_id="test_task",
        documents=[
            PatternDocument(
                task_id="test_task", agent_idx=0,
                synthesis="The output equals the input (identity).",
            ),
        ],
    )


# Code samples
IDENTITY_CODE = "def transform(grid):\n    return grid"
BROKEN_CODE = "def transform(grid):\n    raise ValueError('oops')"
PARTIAL_CODE = "def transform(grid):\n    return [[0, 0], [1, 1]]"  # matches train[0] only → 1/2 train
ALL_WRONG_CODE = "def transform(grid):\n    return [[9, 9], [9, 9]]"  # matches neither train → 0/2


def _run(coro):
    return asyncio.run(coro)


# ────────────────────────────────────────────────────────────────────
# Phase 1 — bit-identical to B4
# ────────────────────────────────────────────────────────────────────

def test_solves_immediately_skips_phase2():
    """Phase 1 step 1: define identity (2/2 train) → Phase 2 skipped (train_score=1.0)."""
    client = MockAsyncClient([_define_resp(IDENTITY_CODE, summary="identity")])
    result = _run(define_transformation(
        _make_task(), _make_exploration(), client, model="mock",
        enable_refinement=True,
    ))
    assert len(result.attempts) == 1
    assert result.attempts[0].iter == 0 and result.attempts[0].phase == "phase1"
    assert result.attempts[0].train_num_correct == 2
    assert result.code == IDENTITY_CODE
    assert client.call_count == 1, "should not have entered Phase 2"


def test_phase1_repairs_on_exec_error():
    """Phase 1: broken define → repair warning fires → next step defines clean."""
    client = MockAsyncClient([
        _define_resp(BROKEN_CODE),   # step 1: errors at exec
        _define_resp(IDENTITY_CODE), # step 2: clean
    ])
    result = _run(define_transformation(
        _make_task(), _make_exploration(), client, model="mock",
        enable_refinement=False,
    ))
    assert result.repair_attempts == 1
    assert len(result.attempts) == 1, "only the clean define is logged as an Attempt"
    assert result.attempts[0].code == IDENTITY_CODE


def test_phase1_late_first_define():
    """Phase 1 step 9 errors → step 10 gets the repair chance → if also errors,
    bounded exit (B4 semantics for late define)."""
    client = MockAsyncClient(
        [_think_resp() for _ in range(8)]            # steps 1-8: think
        + [_define_resp(BROKEN_CODE)]                # step 9: define + error
        + [_define_resp(BROKEN_CODE)]                # step 10: repair attempt also errors
    )
    result = _run(define_transformation(
        _make_task(), _make_exploration(), client, model="mock",
        enable_refinement=False,
    ))
    assert result.repair_attempts == 2
    assert result.attempts == []
    assert result.final_error is not None
    assert client.call_count == 10, "loop should consume exactly max_steps=10 calls"


def test_phase1_uses_b4_system_prompt():
    """Phase 1 calls must use the unmodified B4 system prompt (bit-identical to B4)."""
    client = MockAsyncClient([_define_resp(IDENTITY_CODE)])
    _run(define_transformation(
        _make_task(), _make_exploration(), client, model="mock",
        enable_refinement=True,
    ))
    assert client.call_system(0) == SYSTEM_PROMPT


def test_phase1_uses_phase1_tools_only():
    """Phase 1 tools list must be [think, define_transformation] — no submit_refined."""
    client = MockAsyncClient([_define_resp(IDENTITY_CODE)])
    _run(define_transformation(
        _make_task(), _make_exploration(), client, model="mock",
        enable_refinement=True,
    ))
    assert client.call_tools_names(0) == ["think", "define_transformation"]


# ────────────────────────────────────────────────────────────────────
# Phase 2 — B5b refinement
# ────────────────────────────────────────────────────────────────────

def test_threshold_skips_below_50pct_train():
    """Phase 1 gets 0/2 train → train_score=0 < 0.5 → Phase 2 must not fire."""
    client = MockAsyncClient([_define_resp(ALL_WRONG_CODE)])
    result = _run(define_transformation(
        _make_task(), _make_exploration(), client, model="mock",
        enable_refinement=True,
    ))
    assert len(result.attempts) == 1
    assert result.attempts[0].iter == 0
    assert client.call_count == 1, "Phase 2 should not have fired below threshold"


def test_threshold_skips_perfect_train():
    """Phase 1 gets 2/2 train → train_score=1.0 → Phase 2 must not fire."""
    client = MockAsyncClient([_define_resp(IDENTITY_CODE)])
    result = _run(define_transformation(
        _make_task(), _make_exploration(), client, model="mock",
        enable_refinement=True,
    ))
    assert client.call_count == 1, "Phase 2 should not fire when train is already 100%"


def test_threshold_fires_at_exactly_50pct_train():
    """Phase 1 gets 1/2 train (=0.5) → at threshold → Phase 2 must fire."""
    client = MockAsyncClient([
        _define_resp(PARTIAL_CODE),                  # Phase 1: 1/2 train
        _refine_resp(IDENTITY_CODE, "fixed branch"), # Phase 2 substep 1: clean, 2/2 train
    ])
    result = _run(define_transformation(
        _make_task(), _make_exploration(), client, model="mock",
        enable_refinement=True,
    ))
    assert len(result.attempts) == 2
    assert result.attempts[1].phase == "refinement" and result.attempts[1].iter == 1
    assert result.attempts[1].train_num_correct == 2


def test_phase2_substep1_clean_submit_no_substep2():
    """Substep 1 submits clean refined code → substep 2 never runs."""
    client = MockAsyncClient([
        _define_resp(PARTIAL_CODE),
        _refine_resp(IDENTITY_CODE, "fixed"),
    ])
    result = _run(define_transformation(
        _make_task(), _make_exploration(), client, model="mock",
        enable_refinement=True,
    ))
    assert client.call_count == 2  # Phase 1 + Phase 2 substep 1 only
    assert result.code == IDENTITY_CODE


def test_phase2_substep1_uses_b5b_system_and_phase2_tools():
    """Substep 1 must swap to SYSTEM_PROMPT_B5B and use PHASE2_TOOLS."""
    client = MockAsyncClient([
        _define_resp(PARTIAL_CODE),
        _refine_resp(IDENTITY_CODE),
    ])
    _run(define_transformation(
        _make_task(), _make_exploration(), client, model="mock",
        enable_refinement=True,
    ))
    # Call 0 = Phase 1; Call 1 = Phase 2 substep 1
    assert client.call_system(1) == SYSTEM_PROMPT_B5B
    assert client.call_tools_names(1) == ["think", "submit_refined_transformation"]


def test_phase2_substep1_think_then_substep2_submit():
    """Substep 1 thinks (no submit) → substep 2 forces submit with think dropped."""
    client = MockAsyncClient([
        _define_resp(PARTIAL_CODE),                   # Phase 1: 1/2
        _think_resp("let me localize the bug"),       # Substep 1: think only
        _refine_resp(IDENTITY_CODE, "fixed it"),      # Substep 2: forced submit
    ])
    result = _run(define_transformation(
        _make_task(), _make_exploration(), client, model="mock",
        enable_refinement=True,
    ))
    assert client.call_count == 3
    # Substep 2 (call idx 2) must have think dropped — only submit_refined available.
    assert client.call_tools_names(2) == ["submit_refined_transformation"]
    assert len(result.attempts) == 2
    assert result.attempts[1].train_num_correct == 2


def test_phase2_substep1_no_tool_call_then_substep2_force():
    """Substep 1 emits plain text → substep 2 forces submit with no-submit nudge."""
    client = MockAsyncClient([
        _define_resp(PARTIAL_CODE),
        _text_resp("hmm let me think out loud"),
        _refine_resp(IDENTITY_CODE, "fixed"),
    ])
    result = _run(define_transformation(
        _make_task(), _make_exploration(), client, model="mock",
        enable_refinement=True,
    ))
    assert client.call_count == 3
    # Substep 2 must contain the no-submit nudge in the trailing user message.
    substep2_msgs = client.completions.calls[2]["messages"]
    trailing_user = [m["content"] for m in substep2_msgs if m["role"] == "user"][-1]
    assert "you must now call" in trailing_user.lower() and "plain-text" in trailing_user.lower(), (
        f"forced nudge missing from trailing user message: {trailing_user[:200]!r}"
    )
    assert len(result.attempts) == 2


def test_phase2_submit_errors_repair_without_burning_step_budget():
    """Step 0 submits broken code → repair (NOT a step) → step 0 submits clean.
    Verifies exec errors don't increment the step counter."""
    client = MockAsyncClient([
        _define_resp(PARTIAL_CODE),                   # Phase 1: 1/2
        _refine_resp(BROKEN_CODE, "tried a fix"),     # Step 0: submits, errors
        _refine_resp(IDENTITY_CODE, "real fix"),      # Step 0 again (repair): clean
    ])
    result = _run(define_transformation(
        _make_task(), _make_exploration(), client, model="mock",
        enable_refinement=True,
    ))
    assert client.call_count == 3
    # Repair call (call idx 2) must STILL be on step 0 → tools include think.
    assert client.call_tools_names(2) == ["think", "submit_refined_transformation"]
    # Repair user message must include the exec-error context.
    trailing_user = [m["content"] for m in client.completions.calls[2]["messages"]
                     if m["role"] == "user"][-1]
    assert "raised an execution error" in trailing_user.lower()
    assert len(result.attempts) == 2
    assert result.attempts[1].train_num_correct == 2


def test_phase2_max_repairs_exhausted():
    """Step 0 submits broken code PHASE2_MAX_REPAIRS times in a row → exit
    with no Phase 2 attempt, Phase 1 attempt preserved."""
    client = MockAsyncClient([
        _define_resp(PARTIAL_CODE),                   # Phase 1
        *[_refine_resp(BROKEN_CODE, f"attempt {i}") for i in range(PHASE2_MAX_REPAIRS)],
    ])
    result = _run(define_transformation(
        _make_task(), _make_exploration(), client, model="mock",
        enable_refinement=True,
    ))
    assert client.call_count == 1 + PHASE2_MAX_REPAIRS
    assert len(result.attempts) == 1  # Phase 1 only
    assert result.code == PARTIAL_CODE


def test_phase2_step_budget_exhausted_no_submission():
    """Step 0 thinks → step 1 thinks/no-submit → out of step budget → exit."""
    client = MockAsyncClient([
        _define_resp(PARTIAL_CODE),                   # Phase 1: 1/2 (preserved)
        _text_resp("..."),                            # Step 0: text only → step++
        _text_resp("..."),                            # Step 1 (forced): text only → step++ → exit
    ])
    result = _run(define_transformation(
        _make_task(), _make_exploration(), client, model="mock",
        enable_refinement=True,
    ))
    assert client.call_count == 1 + PHASE2_MAX_STEPS
    assert len(result.attempts) == 1
    assert result.attempts[0].phase == "phase1"


def test_phase2_failure_feedback_anchored_before_step0_think():
    """Regression test: the failure feedback must appear at a FIXED position
    in the message list — after the Phase 1 trace, BEFORE any Phase 2 think
    entries — not floating at the end via the transient `warnings` channel.

    Setup: Phase 1 defines partial → step 0 thinks → step 1 (forced) submits clean.
    Inspect step 1's message list and verify the failure-feedback user message
    appears before the step-0 think tool_call.
    """
    client = MockAsyncClient([
        _define_resp(PARTIAL_CODE),                # Phase 1
        _think_resp("step 0 thinking about bug"),  # Step 0: think → step=1
        _refine_resp(IDENTITY_CODE, "fixed"),      # Step 1: forced submit, clean
    ])
    _run(define_transformation(
        _make_task(), _make_exploration(), client, model="mock",
        enable_refinement=True,
    ))

    # Step 1 is call idx 2 (Phase 1 + step 0 + step 1).
    step1_messages = client.completions.calls[2]["messages"]

    # Find the index of the user message containing the failure feedback grids.
    # The renderer puts "Your transform did not solve all training pairs" near the top.
    feedback_idx = next(
        (i for i, m in enumerate(step1_messages)
         if m["role"] == "user" and "did not solve all training pairs" in m.get("content", "")),
        None,
    )
    assert feedback_idx is not None, "failure feedback user message missing from step 1"

    # Find the index of the assistant tool_call carrying the step-0 think.
    step0_think_idx = next(
        (i for i, m in enumerate(step1_messages)
         if m["role"] == "assistant" and m.get("tool_calls")
         and any(tc["function"]["name"] == "think"
                 and "step 0 thinking about bug" in tc["function"]["arguments"]
                 for tc in m["tool_calls"])),
        None,
    )
    assert step0_think_idx is not None, "step-0 think missing from step 1's replayed trace"

    # The anchoring contract: failure feedback comes BEFORE the step-0 think.
    assert feedback_idx < step0_think_idx, (
        f"failure feedback at idx {feedback_idx} but step-0 think at idx {step0_think_idx} "
        f"— ordering wrong: failure feedback must precede Phase 2 think entries"
    )

    # And there must NOT be a *second* copy of the failure-feedback grids in the
    # trailing warning (which would happen if we kept it in the warnings list).
    trailing_user = next(
        (m["content"] for m in reversed(step1_messages) if m["role"] == "user"),
        "",
    )
    # The trailing message should be ONLY the force nudge, not the grids.
    assert "Your transform did not solve" not in trailing_user, (
        "failure-feedback grids leaked into the trailing warning — should only be in the trace"
    )
    assert "you must now call" in trailing_user.lower() and "plain-text" in trailing_user.lower(), (
        "trailing user message should carry the force nudge"
    )


def test_phase2_step1_force_submit_errors_then_repair():
    """Step 0 thinks → step 1 (forced) submits broken → repair (still step 1,
    force tools) → submits clean."""
    client = MockAsyncClient([
        _define_resp(PARTIAL_CODE),                   # Phase 1
        _think_resp("localize bug"),                  # Step 0: think → step=1
        _refine_resp(BROKEN_CODE, "fix v1"),          # Step 1 forced: errors
        _refine_resp(IDENTITY_CODE, "fix v2"),        # Step 1 repair (still forced): clean
    ])
    result = _run(define_transformation(
        _make_task(), _make_exploration(), client, model="mock",
        enable_refinement=True,
    ))
    assert client.call_count == 4
    # The repair call (idx 3) must be on step 1 → force tools (no think).
    assert client.call_tools_names(3) == ["submit_refined_transformation"]
    assert len(result.attempts) == 2
    assert result.attempts[1].train_num_correct == 2


def test_refinement_disabled_skips_phase2():
    """enable_refinement=False → Phase 2 never runs even when train < 1.0."""
    client = MockAsyncClient([_define_resp(PARTIAL_CODE)])
    result = _run(define_transformation(
        _make_task(), _make_exploration(), client, model="mock",
        enable_refinement=False,
    ))
    assert client.call_count == 1
    assert len(result.attempts) == 1


# ────────────────────────────────────────────────────────────────────
# Result-shape & backward-compat sanity
# ────────────────────────────────────────────────────────────────────

def test_attempts_field_serializes_round_trip():
    """to_dict / from_dict preserve the attempts list."""
    from pipeline.agents.transformation_definer.types import (
        Attempt, TransformationResult,
    )
    r = TransformationResult(
        task_id="t",
        attempts=[
            Attempt(iter=0, phase="phase1", code="x",
                    train_num_correct=2, train_num_total=2),
            Attempt(iter=1, phase="refinement", code="y",
                    train_num_correct=3, train_num_total=3),
        ],
    )
    d = r.to_dict()
    r2 = TransformationResult.from_dict(d)
    assert len(r2.attempts) == 2
    assert r2.attempts[0].iter == 0 and r2.attempts[1].iter == 1
    assert r2.best_attempt().train_score == 1.0


def test_from_dict_handles_legacy_b4_records():
    """B4 records (no attempts field) must deserialize cleanly."""
    from pipeline.agents.transformation_definer.types import TransformationResult
    legacy = {
        "task_id": "t", "code": "x",
        "train_num_correct": 1, "train_num_total": 2,
        "test_results": [], "trace": [],
    }
    r = TransformationResult.from_dict(legacy)
    assert r.attempts == []
    assert r.best_attempt() is None
    assert r.train_num_correct == 1


def test_b5b_constants_locked():
    """Sanity that the B5b knobs are at their locked values."""
    assert REFINEMENT_TRIGGER_MIN_TRAIN == 0.5
    assert PHASE2_MAX_ITERS == 1
    assert PHASE2_MAX_STEPS == 2
    assert PHASE2_MAX_REPAIRS == 3
