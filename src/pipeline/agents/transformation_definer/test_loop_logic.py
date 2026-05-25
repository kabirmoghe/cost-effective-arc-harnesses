"""Smoketests for the definer Phase 1 + Phase 2 loop logic.

The OpenAI client is mocked so these tests run instantly with no API cost.
Each test feeds canned chat-completions responses (think calls, define calls,
text-only responses) and asserts behavior of the unified `_run_define_loop`
primitive + the public `define_transformation` driver.

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
    MAX_REFINEMENTS, MAX_REFINEMENT_SUBSTEPS, define_transformation,
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
PARTIAL_CODE = "def transform(grid):\n    return [[0, 0], [1, 1]]"  # matches train[0] only


def _run(coro):
    return asyncio.run(coro)


# ────────────────────────────────────────────────────────────────────
# Phase 1 only — these run with enable_refinement=False to isolate Phase 1 behavior
# ────────────────────────────────────────────────────────────────────

def test_solves_immediately():
    """Phase 1 step 1: define identity — train 100% → 1 attempt, no Phase 2."""
    client = MockAsyncClient([_define_resp(IDENTITY_CODE, summary="identity")])
    result = _run(define_transformation(
        _make_task(), _make_exploration(), client, model="mock",
        enable_refinement=True,
    ))
    assert len(result.attempts) == 1
    assert result.attempts[0].iter == 0 and result.attempts[0].phase == "phase1"
    assert result.attempts[0].train_num_correct == 2
    assert result.attempts[0].train_num_total == 2
    assert result.code == IDENTITY_CODE
    assert client.call_count == 1, "should not have entered Phase 2"


def test_repairs_on_exec_error():
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


def test_late_first_define():
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


def test_warning_concat_no_clobber():
    """Phase 1 step 8 define errors; step 9 should receive BOTH the exec-error
    warning AND the "remaining steps" urgency. Regression test for the bug
    where urgency overwrote the error warning."""
    client = MockAsyncClient([
        *[_think_resp() for _ in range(7)],   # steps 1-7
        _define_resp(BROKEN_CODE),            # step 8: define + error
        _define_resp(BROKEN_CODE),            # step 9: should see both warnings
        _define_resp(IDENTITY_CODE),          # step 10: clean (gives loop room to land)
    ])
    _ = _run(define_transformation(
        _make_task(), _make_exploration(), client, model="mock",
        enable_refinement=False,
    ))

    # Step 9 is the 9th LLM call. Inspect the user message appended to its messages.
    step9_messages = client.completions.calls[8]["messages"]
    user_blocks = [m["content"] for m in step9_messages if m["role"] == "user"]
    trailing = user_blocks[-1]
    assert "raised an error" in trailing, "exec-error warning missing from step 9 messages"
    # Urgency message at remaining=2 includes "step 9 of 10" + "Plan to define".
    assert "step 9 of 10" in trailing or "final step" in trailing.lower(), (
        "urgency warning missing from step 9 messages — concat is dropping it"
    )
    # And exec-error should appear BEFORE the urgency in the joined text.
    assert trailing.index("raised an error") < trailing.index("step 9 of 10"), (
        "ordering wrong — exec-error should appear first"
    )


# ────────────────────────────────────────────────────────────────────
# Phase 2 — refinement loop
# ────────────────────────────────────────────────────────────────────

def test_refines_once_to_solution():
    """Phase 1 defines train 1/2 (partial); Phase 2 iter 1 defines train 2/2 → exit."""
    client = MockAsyncClient([
        _define_resp(PARTIAL_CODE),   # Phase 1: partial (1/2 train)
        _define_resp(IDENTITY_CODE),  # Phase 2 iter 1 substep 1: clean, 2/2 train
    ])
    result = _run(define_transformation(
        _make_task(), _make_exploration(), client, model="mock",
        enable_refinement=True,
    ))
    assert len(result.attempts) == 2
    assert result.attempts[0].iter == 0 and result.attempts[0].train_num_correct == 1
    assert result.attempts[1].iter == 1 and result.attempts[1].train_num_correct == 2
    # Top-level fields mirror best-by-train (the second attempt).
    assert result.code == IDENTITY_CODE
    assert result.train_num_correct == 2
    # Phase 2 exits early after a 100% train iter, so iter 2 should NOT run.
    assert client.call_count == 2


def test_refines_twice_fails():
    """Phase 1 + 2 refinements all defines, all stay at 1/2 train → 3 attempts logged,
    best-by-train selected (ties to later iter)."""
    client = MockAsyncClient([
        _define_resp(PARTIAL_CODE),   # Phase 1: 1/2
        _define_resp(PARTIAL_CODE),   # Refinement iter 1: 1/2
        _define_resp(PARTIAL_CODE),   # Refinement iter 2: 1/2
    ])
    result = _run(define_transformation(
        _make_task(), _make_exploration(), client, model="mock",
        enable_refinement=True,
    ))
    assert len(result.attempts) == 3
    iters = [a.iter for a in result.attempts]
    assert iters == [0, 1, 2]
    # All tied at 1/2 train; best_attempt picks the later one (tie-break by iter).
    best = result.best_attempt()
    assert best is not None and best.iter == 2


def test_refinement_repairs_exec_error():
    """Phase 2 iter 1: first substep emits broken code (repair); next substep
    clean. Verifies the unified primitive applies repair logic inside refinement."""
    client = MockAsyncClient([
        _define_resp(PARTIAL_CODE),   # Phase 1: 1/2 (triggers Phase 2)
        _define_resp(BROKEN_CODE),    # Refinement iter 1 substep 1: errors
        _define_resp(IDENTITY_CODE),  # Refinement iter 1 substep 2: clean, 2/2
    ])
    result = _run(define_transformation(
        _make_task(), _make_exploration(), client, model="mock",
        enable_refinement=True,
    ))
    assert len(result.attempts) == 2
    assert result.attempts[1].train_num_correct == 2
    assert result.repair_attempts == 1, "the broken refinement define counts as a repair"


def test_phase2_no_tool_call_graceful():
    """Phase 2 iter 1: model emits text-only across all 5 substeps. Driver should
    nudge and retry once (another 5 substep budget). If still nothing, exit Phase 2.

    Total calls: 1 (Phase 1) + 5 (iter 1) + 5 (iter 1 retry with nudge) = 11.
    We give the retry budget defines on the last substep so this iteration
    eventually lands an attempt.
    """
    text_block = [_text_resp() for _ in range(MAX_REFINEMENT_SUBSTEPS)]
    retry_block = (
        [_text_resp() for _ in range(MAX_REFINEMENT_SUBSTEPS - 1)]
        + [_define_resp(IDENTITY_CODE)]
    )
    client = MockAsyncClient([
        _define_resp(PARTIAL_CODE),  # Phase 1
        *text_block,                 # Refinement iter 1: all text
        *retry_block,                # Refinement iter 1 retry: text then define
    ])
    result = _run(define_transformation(
        _make_task(), _make_exploration(), client, model="mock",
        enable_refinement=True,
    ))
    # The retry landed a clean define — that's the iter=1 attempt.
    assert len(result.attempts) == 2
    assert result.attempts[1].iter == 1
    # Iter 2 still runs (since iter 1 ultimately yielded an attempt). Mock will
    # raise if any more LLM calls happen — so iter 2 must have nothing to do.
    # Actually iter 2 WILL be entered because iter 1 didn't hit train 100%
    # in the FIRST loop... but the retry did, so we exit Phase 2 early via the
    # `train_score >= 1.0` check at the top of the loop.
    # Total calls = 1 + 5 + 5 = 11; no more.
    assert client.call_count == 1 + MAX_REFINEMENT_SUBSTEPS * 2


# ────────────────────────────────────────────────────────────────────
# Result-shape & backward-compat sanity
# ────────────────────────────────────────────────────────────────────

def test_attempts_field_serializes_round_trip():
    """to_dict / from_dict preserve the attempts list (B5 schema additive)."""
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
