# ARC-AGI-1 task data

Vendored from the official ARC-AGI repository (François Chollet),
<https://github.com/fchollet/ARC-AGI>, licensed under Apache 2.0 (see `LICENSE`).

- `training/` — 400 demonstration tasks
- `evaluation/` — 400 evaluation tasks

Each file is a single task: a JSON object with `train` and `test` arrays of
`{input, output}` grid pairs (2D arrays of integers 0–9). Loaded via
`src/shared/loader.py` (`load_task`, `get_task_ids`).
