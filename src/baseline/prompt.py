"""Baseline (no-CoT) prompt.

Grid representation: ASCII with coordinate headers (matches the pipeline
explorer + definer agents) so the cross-architecture comparison holds grid
format constant. Only the output format (JSON `{"output": [[...]]}`) is
preserved from the original ARC Prize-style harness — so `extract_response`
is unchanged. See `shared/formatting.py::grid_to_ascii` for the format.
"""

from shared.formatting import grid_to_ascii, format_task_examples
from shared.types import Task


BASELINE_SYSTEM_PROMPT = """You are an expert at solving grid transformation puzzles.

Each grid is a 2D array of integers 0-9 representing colors:
0=black, 1=blue, 2=red, 3=green, 4=yellow, 5=gray, 6=magenta, 7=orange, 8=cyan, 9=maroon

You will be given training examples (input/output pairs) that demonstrate a transformation rule, then a test input. Apply the same transformation rule to produce the output.

Grids are rendered with row/column coordinate headers for easy reference.

# Output Format

**Do not** output any intermediate reasoning, explanation, markdown-fenced code.
Respond only with a single valid JSON object that contains exactly one key: "output", whose value is the transformed grid as a 2D array of integers.

Example of the ONLY acceptable response shape:
{"output": [[0,1,2],[3,4,5]]}"""


def build_user_message(task: Task, test_index: int = 0) -> str:
    few_shots = format_task_examples(task)
    test_input = grid_to_ascii(task.test[test_index].input)
    return f"""Here are the training examples:

{few_shots}

Test input:
{test_input}"""
