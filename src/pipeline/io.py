"""File I/O helpers for pipeline artifacts.

Layout (run-first, task-second):

    {output_dir}/{run_id}/{task_id}/pattern_explorer_{idx}.json
    {output_dir}/{run_id}/{task_id}/transformation_definer_{idx}.json
    {output_dir}/{run_id}/logs/{task_id}.log
    {output_dir}/{run_id}/summary.json

`run_id` is a UUID v7 string — the same value used as the primary key in the
`evals` DB table, so a folder on disk maps 1:1 to a DB row.
"""

from __future__ import annotations

import json
from pathlib import Path

from database.ids import uuid7
from pipeline.agents.pattern_explorer.types import PatternDocument
from pipeline.agents.transformation_definer.types import TransformationResult


def new_run_id() -> str:
    """Generate a fresh UUID v7 run id as a string."""
    return str(uuid7())


def run_dir(output_dir: Path, run_id: str) -> Path:
    return output_dir / run_id


def task_dir(output_dir: Path, run_id: str, task_id: str) -> Path:
    return run_dir(output_dir, run_id) / task_id


def pattern_explorer_filename(agent_idx: int) -> str:
    return f"pattern_explorer_{agent_idx}.json"


def pattern_explorer_path(output_dir: Path, doc: PatternDocument) -> Path:
    return task_dir(output_dir, doc.run_id, doc.task_id) / pattern_explorer_filename(doc.agent_idx)


def save_pattern_document(doc: PatternDocument, output_dir: Path) -> Path:
    path = pattern_explorer_path(output_dir, doc)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc.to_dict(), indent=2))
    return path


def load_pattern_document(json_path: Path | str) -> PatternDocument:
    data = json.loads(Path(json_path).read_text())
    return PatternDocument.from_dict(data)


def transformation_definer_filename(agent_idx: int) -> str:
    return f"transformation_definer_{agent_idx}.json"


def transformation_definer_path(output_dir: Path, result: TransformationResult) -> Path:
    return task_dir(output_dir, result.run_id, result.task_id) / transformation_definer_filename(result.agent_idx)


def save_transformation_result(result: TransformationResult, output_dir: Path) -> Path:
    path = transformation_definer_path(output_dir, result)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2))
    return path


def load_transformation_result(json_path: Path | str) -> TransformationResult:
    data = json.loads(Path(json_path).read_text())
    return TransformationResult.from_dict(data)
