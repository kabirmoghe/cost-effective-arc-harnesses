# Transformation Definer — Implementation Reference

Design notes for implementing `pipeline/agents/transformation_definer/`. Revisit
before picking this work back up.

## Role

Consumes compiled output from N parallel `pattern_explorer` runs for a task and
emits a single `Transformation` — a short summary, reasoning, and an executable
`transform(grid) -> grid` Python string. The agent thinks in a tool loop (same
`think` pattern as the explorer), then terminates by emitting a structured
result. Code is executed; on error, the agent is asked to repair; on success,
output is compared against ground truth.

## Decisions

### 1. Function skeleton

Mirror `pattern_explorer/core.py` for consistency, but:

- `_parse_tool_calls` currently carries a `note_pattern` branch copied from the
  explorer. The definer has no such tool — remove the branch and the `patterns`
  arg. Keep only `think`, plus the terminal `define_transformation` tool (see §3).
- `Transformation` dataclass in `types.py` is close. Add at minimum:
  `task_id: str`, `run_id: str`, `trace: list[TraceEntry]`, `usage: dict`,
  `repair_attempts: int`, `final_error: str | None`. This mirrors
  `PatternDocument`'s identity/provenance/content split and makes failures
  visible without re-running.

### 2. Pulling explorer results

Decide *where* explorer outputs live before writing
`_compile_explorer_outputs`:

- If outputs are persisted via `pipeline/io.py`, load by `run_id`.
- If they only live in memory during a run, pass the `ExplorationResult`
  directly. Skip `run_id` plumbing until multi-process runs actually exist.

Recommend: start with the in-memory `ExplorationResult` signature. Add
`run_id`-based loading only when needed.

Compiled context format (goes after system prompt, before the trace loop):

```
=== Explorer 1 ===
Patterns:
  1. ...
  2. ...
Transformation Rule (synthesis):
  ...

=== Explorer 2 ===
...
```

### 3. Final emission: tool call, not JSON mode

Use a `define_transformation` tool call whose parameters match the
`Transformation` dataclass.

Reasons:
- Reuses the existing tool-loop scaffolding — one more tool entry is cheaper
  than branching into a JSON-mode path.
- Tool schemas enforce the shape (required fields, types) more reliably than
  DeepSeek JSON mode.
- Clean terminal condition: "emitted `define_transformation` → exit loop",
  rather than detecting absence of tool calls.

Loop shape:

```
while step < max_steps:
    response = client.chat.completions.create(..., tool_choice="required")
    for tc in message.tool_calls:
        if tc.name == "think":         append to trace
        if tc.name == "define_transformation":
            candidate = coerce to Transformation
            result, err = execute_transformation(candidate.code, test_input)
            if err is None: return candidate (success)
            else: append repair message, continue loop
```

### 4. Code execution

No Docker/nsjail — overkill for research use. Use a subprocess with:

- Timeout (~5s).
- Restricted `__builtins__` dict.
- Captured stdout/stderr.
- Returns `(result_grid | None, error_str | None)`.

Put this in `shared/` (e.g. `shared/code_exec.py`) so the repair loop and final
evaluation use the same path:

```python
def execute_transformation(code: str, grid: Grid, timeout: float = 5.0)
    -> tuple[Grid | None, str | None]
```

### 5. Repair loop

Cap repair iterations (3–5). Without a cap, a bad candidate can spin forever on
an error the model can't fix.

On failure, feed error back as a user message:

```
Your transform() raised: <error>
Fix the code and call define_transformation again.
```

Store the last error in the returned `Transformation.final_error` so failures
remain visible downstream (metrics, traces).

## Open questions

- Should the definer see the test input grid? (Currently the explorer doesn't —
  only train pairs. The definer probably shouldn't either, to avoid leaking
  test-set signal into the code it writes. Confirm this matches the research
  framing.)
- How to weight disagreement across explorers? For now, present all of them
  verbatim and let the definer reason. Revisit if it systematically picks
  wrong candidates.
