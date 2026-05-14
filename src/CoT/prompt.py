import json
from shared.types import Grid, Example, Task
from typing import List


COT_SYSTEM_PROMPT = """You are an expert at solving grid transformation puzzles.

Each grid is a 2D array of integers 0-9 representing colors:
0=black, 1=blue, 2=red, 3=green, 4=yellow, 5=gray, 6=magenta, 7=orange, 8=cyan, 9=maroon

You will be given training examples (input/output pairs) that demonstrate a transformation rule, then a test input to apply the rule to.

Before producing your answer, reason step by step:
1. What patterns do you observe across the input/output examples?
2. What is the transformation rule?
3. Apply the rule to the test input.

Respond with valid JSON containing:
- "reasoning": your concise step-by-step analysis
- "output": the transformed grid as a 2D array of integers"""


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
