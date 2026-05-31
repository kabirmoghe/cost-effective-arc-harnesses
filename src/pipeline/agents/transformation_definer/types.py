"""Types for the pipeline Transformation Definer agent."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, List

from shared.formatting import grid_to_ascii

Grid = List[List[int]]


@dataclass
class TraceEntry:
    # Tool-call entries: "think", "define_transformation",
    #   "submit_refined_transformation" — render as assistant tool_call + tool result.
    # User-message entries: "user_message" — render as a plain user-role message
    #   in the conversation history. Used to anchor persistent context (e.g. the
    #   Phase 2 train-failure feedback grids) at a fixed position so subsequent
    #   think/submit entries naturally appear after it in the replay, instead of
    #   leaking to the end via the transient `warnings` channel.
    # Downstream consumers (e.g. orchestrator/) may use additional kind strings;
    # this layer is permissive about kind and only acts on the four enumerated above.
    kind: str
    content: str
    # Optional full tool args. Set by `_parse_tool_calls` for submission tools
    # so the trace replay can faithfully reconstruct the assistant's prior
    # tool call (including the actual code, what_changed, etc.). For `think`
    # and `user_message` this stays empty. Backward-compatible default:
    # disk-stored records without `args` deserialize fine; rendering falls
    # back to `content` when args is empty.
    args: dict = field(default_factory=dict)


@dataclass
class TestPairResult:
    """Result of applying the transform to one test pair."""
    test_index: int
    predicted_output: Grid | None = None
    expected_output: Grid | None = None
    correct: bool | None = None
    error: str | None = None


@dataclass
class TrainPairResult:
    """Result of applying the transform to one training pair. Used for the
    refinement feedback loop (B5) — captures the predicted output and any
    error so we can show the model what it produced vs. what was expected."""
    pair_index: int
    input_grid: Grid | None = None
    expected_output: Grid | None = None
    predicted_output: Grid | None = None
    correct: bool | None = None
    error: str | None = None


@dataclass
class Attempt:
    """One `define_transformation` call that executed cleanly and got a
    train-score. Failed defines (execution errors that were repaired) are
    NOT logged as Attempts — they're transient and tracked via `repair_attempts`.

    Attempts are logged in order; the M-selection layer picks
    `max(attempts, key=lambda a: (a.train_score, a.iter))` for downstream
    pass@k aggregation.
    """
    iter: int  # 0 = Phase 1 first clean define; 1, 2 = refinement iterations
    phase: str  # "phase1" or "refinement" (downstream consumers may use other values)
    code: str
    transformation_summary: str = ""
    reasoning: str = ""
    test_results: list[TestPairResult] = field(default_factory=list)
    train_results: list[TrainPairResult] = field(default_factory=list)
    train_num_correct: int = 0
    train_num_total: int = 0
    final_error: str | None = None

    @property
    def train_score(self) -> float:
        return self.train_num_correct / self.train_num_total if self.train_num_total else 0.0

    @property
    def is_clean(self) -> bool:
        return self.final_error is None and self.code != ""


@dataclass
class TransformationResult:
    """Output of the Transformation Definer agent."""
    # Identity
    task_id: str
    run_id: str = ""
    agent_idx: int = 0

    # Provenance
    model: str = ""
    provider: str = ""
    created_at: str = ""

    # Content
    transformation_summary: str = ""
    reasoning: str = ""
    code: str = ""
    trace: list[TraceEntry] = field(default_factory=list)

    # Execution
    repair_attempts: int = 0
    max_repairs: int = 3
    final_error: str | None = None

    # Evaluation (per test pair) — mirrors the best Attempt for backward compat
    test_results: list[TestPairResult] = field(default_factory=list)

    # Training-set score — ranking signal for pass@k selection across M definers.
    # Mirrors the best Attempt's score for backward compat with B4 consumers.
    train_num_correct: int = 0
    train_num_total: int = 0

    # B5: history of every clean-executing define from Phase 1 + Phase 2.
    # `code`/`test_results`/`train_num_*`/`final_error` etc above are populated
    # from `best_attempt()` at the end of `define_transformation()`; `attempts`
    # carries the full trajectory for the selection layer and post-hoc analysis.
    attempts: list[Attempt] = field(default_factory=list)

    # Metrics
    usage: dict = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0})

    @property
    def success(self) -> bool:
        return self.final_error is None and self.code != ""

    @property
    def train_score(self) -> float:
        return self.train_num_correct / self.train_num_total if self.train_num_total else 0.0

    @property
    def correct(self) -> bool | None:
        if not self.test_results:
            return None
        return all(r.correct for r in self.test_results)

    @property
    def num_correct(self) -> int:
        return sum(1 for r in self.test_results if r.correct)

    def best_attempt(self) -> Attempt | None:
        """Best-by-train-score across attempts, ties broken by later iter."""
        if not self.attempts:
            return None
        return max(self.attempts, key=lambda a: (a.train_score, a.iter))

    def to_markdown(self) -> str:
        title = f"# TransformationDefiner: {self.task_id}"
        if self.run_id:
            title += f" (run {self.run_id})"
        lines = [title, ""]

        meta_bits = []
        if self.provider or self.model:
            meta_bits.append(f"**Model:** {self.provider}/{self.model}".strip("/"))
        if self.created_at:
            meta_bits.append(f"**Created:** {self.created_at}")
        if self.usage:
            total = self.usage.get("prompt_tokens", 0) + self.usage.get("completion_tokens", 0)
            meta_bits.append(f"**Tokens:** {total:,}")
        if meta_bits:
            lines.append(" · ".join(meta_bits))
            lines.append("")

        if self.test_results:
            status = f"{self.num_correct}/{len(self.test_results)} correct"
        else:
            status = "No output"
        lines.append(f"**Result:** {status}")
        if self.train_num_total:
            lines.append(f"**Training score:** {self.train_num_correct}/{self.train_num_total}")
        if self.repair_attempts > 0:
            lines.append(f"**Repair attempts:** {self.repair_attempts}/{self.max_repairs}")
        if self.final_error:
            lines.append(f"**Error:** {self.final_error}")
        lines.append("")

        if self.transformation_summary:
            lines.append("## Transformation Summary")
            lines.append("")
            lines.append(self.transformation_summary)
            lines.append("")

        if self.reasoning:
            lines.append("## Reasoning")
            lines.append("")
            lines.append(self.reasoning)
            lines.append("")

        if self.code:
            lines.append("## Code")
            lines.append("")
            lines.append("```python")
            lines.append(self.code)
            lines.append("```")
            lines.append("")

        if self.test_results:
            lines.append("## Evaluation")
            lines.append("")
            for tr in self.test_results:
                tag = "CORRECT" if tr.correct else "INCORRECT" if tr.correct is False else "ERROR"
                lines.append(f"### Test {tr.test_index} — {tag}")
                if tr.error:
                    lines.append(f"Error: {tr.error}")
                if tr.predicted_output is not None:
                    lines.append("**Predicted:**")
                    lines.append("```")
                    lines.append(grid_to_ascii(tr.predicted_output))
                    lines.append("```")
                if tr.expected_output is not None:
                    lines.append("**Expected:**")
                    lines.append("```")
                    lines.append(grid_to_ascii(tr.expected_output))
                    lines.append("```")
                lines.append("")

        if self.trace:
            lines.append("## Trace")
            lines.append("")
            for entry in self.trace:
                if entry.kind == "think":
                    lines.append(f"- 💭 **Think:** {entry.content}")
                elif entry.kind == "define_transformation":
                    lines.append(f"- 🎯 **Define:** {entry.content}")
                elif entry.kind == "submit_refined_transformation":
                    lines.append(f"- 🔁 **Refine:** {entry.content}")
                elif entry.kind == "user_message":
                    snippet = entry.content[:240].replace("\n", " ")
                    lines.append(f"- 📨 **User:** {snippet}...")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["success"] = self.success
        d["correct"] = self.correct
        d["num_correct"] = self.num_correct
        d["train_score"] = self.train_score
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransformationResult":
        trace = [TraceEntry(**t) for t in data.get("trace", [])]
        test_results = [TestPairResult(**t) for t in data.get("test_results", [])]
        attempts = []
        for a in data.get("attempts", []):
            a_test = [TestPairResult(**t) for t in a.get("test_results", [])]
            a_train = [TrainPairResult(**t) for t in a.get("train_results", [])]
            attempts.append(Attempt(
                iter=a.get("iter", 0),
                phase=a.get("phase", "phase1"),
                code=a.get("code", ""),
                transformation_summary=a.get("transformation_summary", ""),
                reasoning=a.get("reasoning", ""),
                test_results=a_test,
                train_results=a_train,
                train_num_correct=a.get("train_num_correct", 0),
                train_num_total=a.get("train_num_total", 0),
                final_error=a.get("final_error"),
            ))
        data.pop("success", None)
        data.pop("correct", None)
        data.pop("num_correct", None)
        return cls(
            task_id=data["task_id"],
            run_id=data.get("run_id", ""),
            agent_idx=data.get("agent_idx", 0),
            model=data.get("model", ""),
            provider=data.get("provider", ""),
            created_at=data.get("created_at", ""),
            transformation_summary=data.get("transformation_summary", ""),
            reasoning=data.get("reasoning", ""),
            code=data.get("code", ""),
            trace=trace,
            repair_attempts=data.get("repair_attempts", 0),
            max_repairs=data.get("max_repairs", 3),
            final_error=data.get("final_error"),
            test_results=test_results,
            train_num_correct=data.get("train_num_correct", 0),
            train_num_total=data.get("train_num_total", 0),
            attempts=attempts,
            usage=data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0}),
        )
