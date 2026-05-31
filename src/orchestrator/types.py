"""B8 — Orchestrator-specific result type.

Extends `pipeline.agents.transformation_definer.types.TransformationResult` with
4 fields the orchestrator populates: exit_reason, done_reason, iterations_used,
spawn_calls. Pipeline's TransformationResult stays pristine.

`OrchestratorResult IS A TransformationResult`, so the existing selection layer
and DB persistence layer accept it without modification (duck-typed access).
"""

from dataclasses import dataclass, field
from typing import Any

from pipeline.agents.transformation_definer.types import (
    TransformationResult,
    TraceEntry,
    TestPairResult,
    TrainPairResult,
    Attempt,
)


@dataclass
class OrchestratorResult(TransformationResult):
    """TransformationResult + orchestrator telemetry.

    `exit_reason` ∈ {"perfect_train", "done", "max_iterations",
      "consecutive_exec_errors", "no_tool_call", "no_clean_define", None}.
    `done_reason` populated only when exit_reason == "done".
    `iterations_used`: substep count (think / define / submit / spawn each =1;
      exec-error repairs free). Fuse fires at MAX_ITERATIONS.
    `spawn_calls`: list of {iteration: int, guidance: str, num_returned: int}.
    """
    exit_reason: str | None = None
    done_reason: str | None = None
    iterations_used: int = 0
    spawn_calls: list[dict] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrchestratorResult":
        # Reuse pipeline's TransformationResult deserialization for the
        # shared fields, then layer the orchestrator-specific fields on top.
        base = TransformationResult.from_dict(data)
        return cls(
            task_id=base.task_id,
            run_id=base.run_id,
            agent_idx=base.agent_idx,
            model=base.model,
            provider=base.provider,
            created_at=base.created_at,
            transformation_summary=base.transformation_summary,
            reasoning=base.reasoning,
            code=base.code,
            trace=base.trace,
            repair_attempts=base.repair_attempts,
            max_repairs=base.max_repairs,
            final_error=base.final_error,
            test_results=base.test_results,
            train_num_correct=base.train_num_correct,
            train_num_total=base.train_num_total,
            attempts=base.attempts,
            usage=base.usage,
            exit_reason=data.get("exit_reason"),
            done_reason=data.get("done_reason"),
            iterations_used=data.get("iterations_used", 0),
            spawn_calls=data.get("spawn_calls", []),
        )
