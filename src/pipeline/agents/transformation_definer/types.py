"""Types for the pipeline Transformation Definer agent."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, List

from shared.formatting import grid_to_ascii

Grid = List[List[int]]


@dataclass
class TraceEntry:
    kind: str  # "think" or "define_transformation"
    content: str


@dataclass
class TestPairResult:
    """Result of applying the transform to one test pair."""
    test_index: int
    predicted_output: Grid | None = None
    expected_output: Grid | None = None
    correct: bool | None = None
    error: str | None = None


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

    # Evaluation (per test pair)
    test_results: list[TestPairResult] = field(default_factory=list)

    # Training-set score — ranking signal for pass@k selection across M definers
    train_num_correct: int = 0
    train_num_total: int = 0

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
            usage=data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0}),
        )
