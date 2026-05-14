"""Types for the pipeline pattern exploration agent."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Pattern:
    id: int
    text: str


@dataclass
class TraceEntry:
    kind: str  # "think" or "note_pattern"
    content: str
    pattern_id: int | None = None  # set for note_pattern entries


@dataclass
class PatternDocument:
    """Output of a single PatternExplorer agent run.

    Identity fields (task_id, run_id, agent_idx) uniquely identify this document
    within the pipeline. Provenance fields (model, provider, created_at) capture
    how it was produced. Content fields (patterns, trace, synthesis) are the
    agent's output.
    """
    # Identity
    task_id: str
    run_id: str = ""
    agent_idx: int = 0

    # Provenance
    model: str = ""
    provider: str = ""
    created_at: str = ""  # ISO 8601 UTC

    # Content
    patterns: list[Pattern] = field(default_factory=list)
    trace: list[TraceEntry] = field(default_factory=list)
    synthesis: str = ""

    # Metrics
    usage: dict = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PatternDocument":
        patterns = [Pattern(**p) for p in data.get("patterns", [])]
        trace = [TraceEntry(**t) for t in data.get("trace", [])]
        return cls(
            task_id=data["task_id"],
            run_id=data.get("run_id", ""),
            agent_idx=data.get("agent_idx", 0),
            model=data.get("model", ""),
            provider=data.get("provider", ""),
            created_at=data.get("created_at", ""),
            patterns=patterns,
            trace=trace,
            synthesis=data.get("synthesis", ""),
            usage=data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0}),
        )

    def to_markdown(self) -> str:
        """Render a human-readable markdown view of this document."""
        title = f"# PatternExplorer: {self.task_id}"
        if self.run_id:
            title += f" (run {self.run_id}, agent {self.agent_idx})"
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

        if self.synthesis:
            lines.append("## Transformation Rule (Synthesis)")
            lines.append("")
            lines.append(self.synthesis)
            lines.append("")

        if self.patterns:
            lines.append("## Noted Patterns")
            lines.append("")
            for p in self.patterns:
                lines.append(f"{p.id}. {p.text}")
            lines.append("")

        if self.trace:
            lines.append("## Exploration Trace")
            lines.append("")
            for entry in self.trace:
                if entry.kind == "think":
                    lines.append(f"- 💭 **Think:** {entry.content}")
                elif entry.kind == "note_pattern":
                    lines.append(f"- 📌 **Pattern #{entry.pattern_id}:** {entry.content}")
            lines.append("")

        return "\n".join(lines)


@dataclass
class ExplorationResult:
    """In-memory aggregation of N parallel explorers on one task."""
    task_id: str
    run_id: str = ""
    documents: list[PatternDocument] = field(default_factory=list)

    @property
    def total_usage(self) -> dict:
        return {
            "prompt_tokens": sum(d.usage["prompt_tokens"] for d in self.documents),
            "completion_tokens": sum(d.usage["completion_tokens"] for d in self.documents),
        }
