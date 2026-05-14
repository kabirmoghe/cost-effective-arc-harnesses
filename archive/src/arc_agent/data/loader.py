"""Load ARC tasks from JSON files - re-exported from shared module."""

from shared.loader import (
    DEFAULT_DATA_ROOT,
    get_data_path,
    get_task_ids,
    load_task,
    load_tasks,
)

__all__ = ["DEFAULT_DATA_ROOT", "get_data_path", "get_task_ids", "load_task", "load_tasks"]
