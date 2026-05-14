"""Metrics calculation for ARC evaluation."""

from typing import List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TaskResult:
    """Result for a single task evaluation."""

    task_id: str
    test_index: int
    success: Optional[bool]
    steps: int
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AccuracyMetrics:
    """Accuracy metrics for an evaluation run."""

    total_tasks: int
    correct: int
    incorrect: int
    errors: int
    accuracy: float
    error_rate: float

    def __str__(self) -> str:
        return (
            f"Accuracy: {self.accuracy:.1%} ({self.correct}/{self.total_tasks})\n"
            f"Incorrect: {self.incorrect}\n"
            f"Errors: {self.errors} ({self.error_rate:.1%})"
        )


def calculate_accuracy(results: List[TaskResult]) -> AccuracyMetrics:
    """Calculate accuracy metrics from task results.

    Args:
        results: List of TaskResult objects

    Returns:
        AccuracyMetrics with calculated values
    """
    total = len(results)
    if total == 0:
        return AccuracyMetrics(
            total_tasks=0,
            correct=0,
            incorrect=0,
            errors=0,
            accuracy=0.0,
            error_rate=0.0,
        )

    correct = sum(1 for r in results if r.success is True)
    incorrect = sum(1 for r in results if r.success is False)
    errors = sum(1 for r in results if r.error is not None)

    return AccuracyMetrics(
        total_tasks=total,
        correct=correct,
        incorrect=incorrect,
        errors=errors,
        accuracy=correct / total if total > 0 else 0.0,
        error_rate=errors / total if total > 0 else 0.0,
    )
