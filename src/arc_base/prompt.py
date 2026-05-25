"""Verbatim port of the ARC Prize "Base LLM" prompt + grid format.

Sources (read 2026-05-14 from arcprize/arc-agi-benchmarking @ main):
- `src/arc_agi_benchmarking/prompts/system_prompt.txt` — the template below.
- `src/arc_agi_benchmarking/prompts/prompt_manager.py` — `json.dumps()` of grids,
  training examples laid out as `--Example {i}--` / `INPUT:` / `OUTPUT:` blocks.

Kept byte-faithful so any "did our run reproduce the upstream harness" question
is settled by reading this file rather than re-reading the upstream repo.
"""

import json

from shared.types import Grid, Example, Task


# Verbatim from upstream system_prompt.txt — do not edit without re-verifying.
ARC_BASE_PROMPT_TEMPLATE = """You are participating in a puzzle solving competition. You are an expert at solving puzzles.

Below is a list of input and output pairs with a pattern. Your goal is to identify the pattern or transformation in the training examples that maps the input to the output, then apply that pattern to the test input to give a final output.

Respond in the format of the training output examples

--Training Examples--
{training_examples}
--End of Training Examples--

--Test Input--
{test_input}
--End of Test Input--

Your response:"""


def format_grid(grid: Grid) -> str:
    """Upstream uses raw `json.dumps()` of the int array."""
    return json.dumps(grid)


def format_training_examples(examples: list[Example]) -> str:
    """`--Example {i}--` / `INPUT:` / `OUTPUT:` blocks, 0-indexed.

    Mirrors prompt_manager.py's layout in the upstream repo.
    """
    parts = []
    for i, ex in enumerate(examples):
        parts.append(f"--Example {i}--")
        parts.append("INPUT:")
        parts.append(format_grid(ex.input))
        parts.append("OUTPUT:")
        parts.append(format_grid(ex.output))
    return "\n".join(parts)


def build_prompt(task: Task, test_index: int = 0) -> str:
    """Format the full prompt for one test pair of a task."""
    return ARC_BASE_PROMPT_TEMPLATE.format(
        training_examples=format_training_examples(task.train),
        test_input=format_grid(task.test[test_index].input),
    )
