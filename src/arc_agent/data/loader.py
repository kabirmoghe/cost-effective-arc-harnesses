"""Load ARC tasks from JSON files."""

import json
from pathlib import Path
from typing import List, Literal, Optional

from .types import Task, Example


# Default path to ARC data relative to project root
DEFAULT_DATA_ROOT = Path(__file__).parent.parent.parent.parent / "arc_official_repo" / "data"


def get_data_path(split: Literal["training", "evaluation"], data_root: Optional[Path] = None) -> Path:
    """Get the path to a data split directory."""
    root = data_root or DEFAULT_DATA_ROOT
    return root / split


def get_task_ids(split: Literal["training", "evaluation"], data_root: Optional[Path] = None) -> List[str]:
    """Get all task IDs for a given split."""
    data_path = get_data_path(split, data_root)
    return sorted([f.stem for f in data_path.glob("*.json")])


def load_task(
    task_id: str,
    split: Literal["training", "evaluation"],
    data_root: Optional[Path] = None
) -> Task:
    """Load a single task by ID."""
    data_path = get_data_path(split, data_root)
    task_file = data_path / f"{task_id}.json"

    if not task_file.exists():
        raise FileNotFoundError(f"Task file not found: {task_file}")

    with open(task_file) as f:
        data = json.load(f)

    return Task(
        task_id=task_id,
        train=[Example(**ex) for ex in data["train"]],
        test=[Example(**ex) for ex in data["test"]],
    )


def load_tasks(
    split: Literal["training", "evaluation"],
    limit: Optional[int] = None,
    data_root: Optional[Path] = None
) -> List[Task]:
    """Load all tasks from a split."""
    task_ids = get_task_ids(split, data_root)

    if limit is not None:
        task_ids = task_ids[:limit]

    return [load_task(tid, split, data_root) for tid in task_ids]
