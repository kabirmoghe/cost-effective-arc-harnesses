import json
from shared.types import Grid, Example, Task
from typing import List


BASELINE_SYSTEM_PROMPT = """You are an expert at solving grid transformation puzzles.

Each grid is a 2D array of integers 0-9 representing colors:
0=black, 1=blue, 2=red, 3=green, 4=yellow, 5=gray, 6=magenta, 7=orange, 8=cyan, 9=maroon

You will be given training examples (input/output pairs) that demonstrate a transformation rule, then a test input. Apply the same transformation rule to produce the output.

# Output Format

**Do not** output any intermediate reasoning, explanation, markdown-fenced code. 
Respond only with a single valid JSON object that contains exactly one key: "output", whose value is the transformed grid as a 2D array of integers.

Example of the ONLY acceptable response shape:
{"output": [[0,1,2],[3,4,5]]}"""


def format_grid(grid: Grid) -> str:
    return json.dumps(grid)


def format_examples(examples: List[Example]) -> str:
    parts = []
    for i, ex in enumerate(examples, 1):
        parts.append(f"Example {i}:")
        parts.append(f"  Input:  {format_grid(ex.input)}")
        parts.append(f"  Output: {format_grid(ex.output)}")
    return "\n".join(parts)


def build_user_message(task: Task, test_index: int = 0) -> str:
    few_shots = format_examples(task.train)
    test_input = format_grid(task.test[test_index].input)
    return f"""Here are the training examples:

{few_shots}

Test input:
{test_input}"""
