"""Evaluation harness for running the ARC agent on task sets."""

import json
from typing import List, Optional, Literal
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..config import AgentConfig, get_default_config
from ..data.loader import load_tasks, load_task
from ..data.types import Task
from ..agent.runner import AgentRunner, AgentResult
from .metrics import TaskResult, AccuracyMetrics, calculate_accuracy


@dataclass
class EvaluationResult:
    """Complete evaluation result."""

    split: str
    model: str
    timestamp: datetime
    task_results: List[TaskResult]
    metrics: AccuracyMetrics
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "split": self.split,
            "model": self.model,
            "timestamp": self.timestamp.isoformat(),
            "metrics": {
                "total_tasks": self.metrics.total_tasks,
                "correct": self.metrics.correct,
                "incorrect": self.metrics.incorrect,
                "errors": self.metrics.errors,
                "accuracy": self.metrics.accuracy,
                "error_rate": self.metrics.error_rate,
            },
            "task_results": [
                {
                    "task_id": r.task_id,
                    "test_index": r.test_index,
                    "success": r.success,
                    "steps": r.steps,
                    "error": r.error,
                }
                for r in self.task_results
            ],
            "config": self.config,
        }

    def save(self, path: Path) -> None:
        """Save evaluation result to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class EvaluationHarness:
    """Harness for evaluating the ARC agent."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        verbose: bool = True,
    ):
        """Initialize the evaluation harness.

        Args:
            config: Agent configuration
            verbose: Print progress during evaluation
        """
        self.config = config or get_default_config()
        self.verbose = verbose
        self.runner = AgentRunner(config=self.config)

    def evaluate_task(
        self,
        task: Task,
        test_index: int = 0,
    ) -> TaskResult:
        """Evaluate a single task.

        Args:
            task: Task to evaluate
            test_index: Which test example to evaluate

        Returns:
            TaskResult with success status
        """
        try:
            result = self.runner.run(task, test_index, verbose=self.verbose)
            return TaskResult(
                task_id=task.task_id,
                test_index=test_index,
                success=result.success,
                steps=result.steps,
            )
        except Exception as e:
            return TaskResult(
                task_id=task.task_id,
                test_index=test_index,
                success=None,
                steps=0,
                error=str(e),
            )

    def evaluate_split(
        self,
        split: Literal["training", "evaluation"],
        limit: Optional[int] = None,
        task_ids: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """Evaluate tasks from a split.

        Args:
            split: Data split to evaluate
            limit: Maximum number of tasks to evaluate
            task_ids: Specific task IDs to evaluate (overrides limit)

        Returns:
            EvaluationResult with all task results and metrics
        """
        # Load tasks
        if task_ids:
            tasks = [load_task(tid, split) for tid in task_ids]
        else:
            tasks = load_tasks(split, limit=limit)

        if self.verbose:
            print(f"Evaluating {len(tasks)} tasks from {split} split")
            print(f"Model: {self.config.model.model_name}")
            print("-" * 40)

        task_results = []

        for i, task in enumerate(tasks):
            if self.verbose:
                print(f"\n[{i+1}/{len(tasks)}] Task: {task.task_id}")

            # Evaluate each test case in the task
            for test_idx in range(task.num_test):
                result = self.evaluate_task(task, test_idx)
                task_results.append(result)

                if self.verbose:
                    status = "PASS" if result.success else ("ERROR" if result.error else "FAIL")
                    print(f"  Test {test_idx}: {status} ({result.steps} steps)")
                    if result.error:
                        print(f"    Error: {result.error}")

        metrics = calculate_accuracy(task_results)

        if self.verbose:
            print("\n" + "=" * 40)
            print("RESULTS")
            print("=" * 40)
            print(metrics)

        return EvaluationResult(
            split=split,
            model=self.config.model.model_name,
            timestamp=datetime.now(),
            task_results=task_results,
            metrics=metrics,
            config=self.config.model_dump(),
        )
