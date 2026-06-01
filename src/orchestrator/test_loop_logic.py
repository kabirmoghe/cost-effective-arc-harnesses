"""Smoketests for the B8 Reflective Orchestrator loop.

OpenAI client mocked with canned chat-completions responses; spawn primitive
monkey-patched. Each test feeds a sequence of tool-call responses and asserts
behavior of `run_orchestrator` against the 5 tools, 3 exit paths, and the fuses.

Run with:
    uv run pytest src/orchestrator/test_loop_logic.py -v
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import pytest

from pipeline.agents.pattern_explorer.types import (
    ExplorationResult, PatternDocument,
)
from pipeline.agents.transformation_definer.types import Attempt
from shared.types import Example, Task

from orchestrator import core as orch_core
from orchestrator.core import (
    run_orchestrator, _parse_tool_calls,
    MAX_ITERATIONS, MAX_CONSECUTIVE_EXEC_ERRORS,
)
from orchestrator.tools import ORCHESTRATOR_TOOLS
from orchestrator.spawn import MAX_SPAWN_CALLS


# ────────────────────────────────────────────────────────────────────
# Mock OpenAI async client (reused pattern from B7 test_loop_logic)
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
    tc = _MockToolCall(_MockFunc("submit_refined_transformation", json.dumps({
        "code": code, "what_changed": what_changed,
    })))
    return _MockResponse(
        choices=[_MockChoice(_MockMessage(tool_calls=[tc]))],
        usage=_MockUsage(prompt_tokens=200, completion_tokens=50),
    )


def _spawn_resp(guidance: str = "what about rotations?") -> _MockResponse:
    tc = _MockToolCall(_MockFunc(
        "explore_new_patterns", json.dumps({"guidance": guidance})
    ))
    return _MockResponse(
        choices=[_MockChoice(_MockMessage(tool_calls=[tc]))],
        usage=_MockUsage(),
    )


def _done_resp(reason: str = "stuck") -> _MockResponse:
    tc = _MockToolCall(_MockFunc("done", json.dumps({"reason": reason})))
    return _MockResponse(
        choices=[_MockChoice(_MockMessage(tool_calls=[tc]))],
        usage=_MockUsage(),
    )


def _text_resp(text: str = "thinking out loud") -> _MockResponse:
    return _MockResponse(
        choices=[_MockChoice(_MockMessage(tool_calls=None, content=text))],
        usage=_MockUsage(),
    )


def _unknown_resp(name: str = "magic_solve") -> _MockResponse:
    tc = _MockToolCall(_MockFunc(name, json.dumps({"foo": "bar"})))
    return _MockResponse(
        choices=[_MockChoice(_MockMessage(tool_calls=[tc]))],
        usage=_MockUsage(),
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
    def __init__(self, responses: list[_MockResponse]):
        self.completions = _MockChatCompletions(responses)
        self.chat = _MockChat(self.completions)

    @property
    def call_count(self) -> int:
        return len(self.completions.calls)

    def call_tools_names(self, call_idx: int) -> list[str]:
        tools = self.completions.calls[call_idx]["tools"]
        return [t["function"]["name"] for t in tools]

    def last_user_message(self, call_idx: int) -> str:
        msgs = self.completions.calls[call_idx]["messages"]
        for m in reversed(msgs):
            if m["role"] == "user":
                return m["content"]
        return ""


# ────────────────────────────────────────────────────────────────────
# Fixtures
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


IDENTITY_CODE = "def transform(grid):\n    return grid"
BROKEN_CODE = "def transform(grid):\n    raise ValueError('oops')"
PARTIAL_CODE = "def transform(grid):\n    return [[0, 0], [1, 1]]"  # 1/2 train
ALL_WRONG_CODE = "def transform(grid):\n    return [[9, 9], [9, 9]]"  # 0/2 train


def _run(coro):
    return asyncio.run(coro)


# ────────────────────────────────────────────────────────────────────
# 1. Structural perfect-train exit
# ────────────────────────────────────────────────────────────────────

def test_perfect_train_exits_immediately():
    """define_transformation that hits 2/2 train → exit_reason='perfect_train'."""
    client = MockAsyncClient([_define_resp(IDENTITY_CODE, summary="identity")])
    result = _run(run_orchestrator(
        _make_task(), _make_exploration(), client, model="mock",
    ))
    assert result.exit_reason == "perfect_train"
    assert len(result.attempts) == 1
    assert result.attempts[0].train_num_correct == 2
    assert result.code == IDENTITY_CODE
    assert client.call_count == 1


def test_refine_then_perfect_train_exits():
    """Partial define (1/2) → refine to identity (2/2) → exit perfect_train."""
    client = MockAsyncClient([
        _define_resp(PARTIAL_CODE),
        _refine_resp(IDENTITY_CODE, "fixed second branch"),
    ])
    result = _run(run_orchestrator(
        _make_task(), _make_exploration(), client, model="mock",
    ))
    assert result.exit_reason == "perfect_train"
    assert len(result.attempts) == 2
    assert result.attempts[1].train_num_correct == 2
    assert client.call_count == 2


# ────────────────────────────────────────────────────────────────────
# 2. done(reason) exit path
# ────────────────────────────────────────────────────────────────────

def test_done_exits_with_best_by_train():
    """define partial (1/2) → done → submit best-by-train attempt with done_reason."""
    client = MockAsyncClient([
        _define_resp(PARTIAL_CODE, summary="partial"),
        _done_resp("can't find the bug"),
    ])
    result = _run(run_orchestrator(
        _make_task(), _make_exploration(), client, model="mock",
    ))
    assert result.exit_reason == "done"
    assert result.done_reason == "can't find the bug"
    assert result.code == PARTIAL_CODE
    assert result.train_num_correct == 1


def test_done_without_any_clean_attempt():
    """done before any define → exit_reason='done' preserved, no attempts,
    final_error reflects the missing submission. The agent CHOSE to exit, so
    the exit reason stays 'done' — final_error carries the no-clean-define info."""
    client = MockAsyncClient([_done_resp("giving up")])
    result = _run(run_orchestrator(
        _make_task(), _make_exploration(), client, model="mock",
    ))
    assert result.exit_reason == "done"
    assert result.done_reason == "giving up"
    assert result.attempts == []
    assert result.final_error is not None
    assert "no clean define" in result.final_error.lower()


def test_max_iterations_without_any_clean_attempt_becomes_no_clean_define():
    """All-think to MAX_ITERATIONS with no submission → exit_reason converted
    from 'max_iterations' to 'no_clean_define' to flag the empty trajectory."""
    client = MockAsyncClient([_think_resp(f"loop {i}") for i in range(MAX_ITERATIONS)])
    result = _run(run_orchestrator(
        _make_task(), _make_exploration(), client, model="mock",
    ))
    assert result.exit_reason == "no_clean_define"
    assert result.attempts == []
    assert result.iterations_used == MAX_ITERATIONS


# ────────────────────────────────────────────────────────────────────
# 3. Max-iterations fuse
# ────────────────────────────────────────────────────────────────────

def test_max_iterations_fuse_with_best_attempt():
    """One partial submit + (MAX_ITERATIONS-1) think loops → max_iterations fuse,
    best-by-train (the partial) survives."""
    responses = [_define_resp(PARTIAL_CODE)] + [
        _think_resp(f"loop {i}") for i in range(MAX_ITERATIONS - 1)
    ]
    client = MockAsyncClient(responses)
    result = _run(run_orchestrator(
        _make_task(), _make_exploration(), client, model="mock",
    ))
    assert result.exit_reason == "max_iterations"
    assert result.iterations_used == MAX_ITERATIONS
    assert result.code == PARTIAL_CODE


# ────────────────────────────────────────────────────────────────────
# 4. Consecutive exec-error fuse
# ────────────────────────────────────────────────────────────────────

def test_consecutive_exec_errors_fuse():
    """N=MAX_CONSECUTIVE_EXEC_ERRORS broken defines in a row → fuse exit."""
    responses = [_define_resp(BROKEN_CODE, summary=f"v{i}")
                 for i in range(MAX_CONSECUTIVE_EXEC_ERRORS)]
    client = MockAsyncClient(responses)
    result = _run(run_orchestrator(
        _make_task(), _make_exploration(), client, model="mock",
    ))
    assert result.exit_reason == "consecutive_exec_errors"
    assert result.attempts == []
    # Exec errors must NOT burn iteration budget.
    assert result.iterations_used == 0
    assert client.call_count == MAX_CONSECUTIVE_EXEC_ERRORS


def test_exec_error_counter_resets_on_clean_submit():
    """Broken → clean (counter resets) → broken×(MAX-1) → does NOT fuse."""
    # MAX = 5: errors at 1,2 (broken×2) then clean (reset) then errors 1..4 (not 5) then perfect.
    n_errors_before_reset = MAX_CONSECUTIVE_EXEC_ERRORS - 3
    responses = (
        [_define_resp(BROKEN_CODE)] * n_errors_before_reset
        + [_define_resp(PARTIAL_CODE)]
        + [_refine_resp(BROKEN_CODE)] * (MAX_CONSECUTIVE_EXEC_ERRORS - 1)
        + [_refine_resp(IDENTITY_CODE, "real fix")]
    )
    client = MockAsyncClient(responses)
    result = _run(run_orchestrator(
        _make_task(), _make_exploration(), client, model="mock",
    ))
    assert result.exit_reason == "perfect_train"
    assert result.train_num_correct == 2


# ────────────────────────────────────────────────────────────────────
# 5. Spawn invocation appends docs (with mocked spawn)
# ────────────────────────────────────────────────────────────────────

def _install_spawn_mock(monkeypatch, return_docs: list[PatternDocument]):
    """Patch the symbol that core.py resolved at import time."""
    async def _fake(task, guidance, client, model, **kwargs):
        return list(return_docs)
    monkeypatch.setattr(orch_core, "spawn_focused_explorers", _fake)


def test_spawn_appends_docs_and_increments_iterations(monkeypatch):
    """explore_new_patterns → spawn returns docs → appended to ExplorationResult
    → iter incremented, spawn_calls recorded."""
    new_doc = PatternDocument(
        task_id="test_task", agent_idx=99,
        synthesis="Spawned finding: output is rotated copy of input.",
    )
    _install_spawn_mock(monkeypatch, [new_doc, new_doc])

    client = MockAsyncClient([
        _spawn_resp("rule out reflection; test rotation"),
        _define_resp(IDENTITY_CODE),
    ])
    exploration = _make_exploration()
    initial_n = len(exploration.documents)
    result = _run(run_orchestrator(
        _make_task(), exploration, client, model="mock",
    ))
    assert result.exit_reason == "perfect_train"
    assert len(result.spawn_calls) == 1
    assert result.spawn_calls[0]["guidance"].startswith("rule out reflection")
    assert result.spawn_calls[0]["num_returned"] == 2
    assert len(exploration.documents) == initial_n + 2
    # The define call (call idx 1) must see the appended PatternExplorer docs in
    # its system/user content.
    msgs = client.completions.calls[1]["messages"]
    findings = next(m["content"] for m in msgs
                    if m["role"] == "user" and "PatternExplorer" in m["content"])
    assert "PatternExplorer 3" in findings  # initial=1 + 2 spawned


def test_spawn_budget_exhaustion_warns_then_continues(monkeypatch):
    """After MAX_SPAWN_CALLS spawns, the (MAX+1)th spawn call is rejected with
    a warning; the loop continues (doesn't fuse)."""
    _install_spawn_mock(monkeypatch, [PatternDocument(
        task_id="test_task", agent_idx=99, synthesis="spawned",
    )])
    # MAX_SPAWN_CALLS spawns (all succeed) + 1 over-budget spawn + final done.
    responses = (
        [_spawn_resp(f"guidance {i}") for i in range(MAX_SPAWN_CALLS)]
        + [_spawn_resp("over budget")]
        + [_done_resp("giving up after exhausting spawn budget")]
    )
    client = MockAsyncClient(responses)
    result = _run(run_orchestrator(
        _make_task(), _make_exploration(), client, model="mock",
    ))
    assert result.exit_reason == "done"
    assert len(result.spawn_calls) == MAX_SPAWN_CALLS
    # The call AFTER the over-budget spawn must contain the budget-exhausted warning.
    after_overrun = client.completions.calls[MAX_SPAWN_CALLS + 1]
    trailing_user = next(
        (m["content"] for m in reversed(after_overrun["messages"]) if m["role"] == "user"),
        "",
    )
    assert "spawn budget exhausted" in trailing_user.lower()


# ────────────────────────────────────────────────────────────────────
# 6. Train-feedback delivered after non-perfect submission
# ────────────────────────────────────────────────────────────────────

def test_train_feedback_anchored_in_trace_after_partial_submit():
    """Partial submit → next iter's messages contain the per-pair feedback
    (rendered by render_train_feedback) as a trace user_message."""
    client = MockAsyncClient([
        _define_resp(PARTIAL_CODE),
        _done_resp("ok"),
    ])
    result = _run(run_orchestrator(
        _make_task(), _make_exploration(), client, model="mock",
    ))
    # On the second call, the trace already contains the feedback user_message.
    second_call_msgs = client.completions.calls[1]["messages"]
    feedback_msg = next(
        (m["content"] for m in second_call_msgs
         if m["role"] == "user" and "1/2 training pairs" in m.get("content", "")),
        None,
    )
    assert feedback_msg is not None
    assert "train_score = 0.50" in feedback_msg
    assert result.exit_reason == "done"


# ────────────────────────────────────────────────────────────────────
# 7. No-tool-call handling — single warn, second consecutive fuses
# ────────────────────────────────────────────────────────────────────

def test_single_no_tool_call_warns_then_continues():
    """One plain-text response → warning fired → next iter the model submits clean."""
    client = MockAsyncClient([
        _text_resp("hmm let me think"),
        _define_resp(IDENTITY_CODE),
    ])
    result = _run(run_orchestrator(
        _make_task(), _make_exploration(), client, model="mock",
    ))
    assert result.exit_reason == "perfect_train"
    # Call 1 (after the no-tool-call) must carry the no-tool warning.
    warning = client.last_user_message(1)
    assert "did not call any tool" in warning.lower()


def test_two_consecutive_no_tool_calls_fuse():
    """Plain text twice in a row → no_tool_call fuse."""
    client = MockAsyncClient([_text_resp("..."), _text_resp("...")])
    result = _run(run_orchestrator(
        _make_task(), _make_exploration(), client, model="mock",
    ))
    assert result.exit_reason == "no_tool_call"


# ────────────────────────────────────────────────────────────────────
# 8. Unknown tool name → warning surfaces the name
# ────────────────────────────────────────────────────────────────────

def test_unknown_tool_name_surfaces_in_warning():
    """Model calls a tool not in the orchestrator's surface → next iter's
    warning names the offender; loop doesn't silently advance."""
    client = MockAsyncClient([
        _unknown_resp("magic_solve"),
        _define_resp(IDENTITY_CODE),
    ])
    result = _run(run_orchestrator(
        _make_task(), _make_exploration(), client, model="mock",
    ))
    assert result.exit_reason == "perfect_train"
    warning = client.last_user_message(1)
    assert "magic_solve" in warning
    assert "not a recognized tool" in warning.lower()


# ────────────────────────────────────────────────────────────────────
# 9. Tiebreaker — earlier iter wins on equal train_score
# ────────────────────────────────────────────────────────────────────

def test_tiebreaker_earlier_iter_wins():
    """Two partial attempts with the same train_score → best = earlier iter
    (mirrors pipeline.selection's lower-agent_idx-wins convention)."""
    client = MockAsyncClient([
        _define_resp(PARTIAL_CODE, summary="first partial"),
        _refine_resp(PARTIAL_CODE, "tried a tweak but same train_score"),
        _done_resp("no improvement"),
    ])
    result = _run(run_orchestrator(
        _make_task(), _make_exploration(), client, model="mock",
    ))
    assert result.exit_reason == "done"
    assert len(result.attempts) == 2
    # Both attempts are 1/2 train; earlier iter (iter=0) must win.
    assert result.attempts[0].iter == 0
    assert result.attempts[1].iter == 1
    assert result.transformation_summary == "first partial", (
        "tiebreaker should favor the earlier iter (iter=0 'first partial'), "
        f"but selected {result.transformation_summary!r}"
    )


# ────────────────────────────────────────────────────────────────────
# 10. Tool surface + system prompt locked
# ────────────────────────────────────────────────────────────────────

def test_tool_surface_locked_to_five_tools():
    """Every call must offer exactly the 5 orchestrator tools, in order."""
    client = MockAsyncClient([_define_resp(IDENTITY_CODE)])
    _run(run_orchestrator(
        _make_task(), _make_exploration(), client, model="mock",
    ))
    expected = ["think", "define_transformation", "submit_refined_transformation",
                "explore_new_patterns", "done"]
    assert client.call_tools_names(0) == expected
    # tool_choice must be auto throughout.
    assert client.completions.calls[0]["tool_choice"] == "auto"


def test_constants_locked():
    """Sanity that the fuses are at the locked-v1 values."""
    assert MAX_ITERATIONS == 15
    assert MAX_CONSECUTIVE_EXEC_ERRORS == 5
    assert MAX_SPAWN_CALLS == 3
    assert len(ORCHESTRATOR_TOOLS) == 5
