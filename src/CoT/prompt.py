"""CoT (structured) prompt.

Grid representation: ASCII with coordinate headers (matches the pipeline
explorer + definer agents) so the cross-architecture comparison holds grid
format constant. Output schema (`{"reasoning", "output"}`) is preserved from
the original — so `extract_response` is unchanged.
"""

from shared.formatting import grid_to_ascii, format_task_examples
from shared.types import Task


COT_SYSTEM_PROMPT = """You are an expert at solving grid transformation puzzles.

Each grid is a 2D array of integers 0-9 representing colors:
0=black, 1=blue, 2=red, 3=green, 4=yellow, 5=gray, 6=magenta, 7=orange, 8=cyan, 9=maroon

You will be given training examples (input/output pairs) that demonstrate a transformation rule, then a test input to apply the rule to.

Grids are rendered with row/column coordinate headers for easy reference.

Before producing your answer, reason step by step:
1. What patterns do you observe across the input/output examples?
2. What is the transformation rule?
3. Apply the rule to the test input.

Respond with valid JSON containing:
- "reasoning": your concise step-by-step analysis
- "output": the transformed grid as a 2D array of integers"""


def build_user_message(task: Task, test_index: int = 0) -> str:
    few_shots = format_task_examples(task)
    test_input = grid_to_ascii(task.test[test_index].input)
    return f"""Here are the training examples:

{few_shots}

Test input:
{test_input}"""
