"""System prompts and context formatting for the ARC agent."""

from typing import List, Optional

from ..data.types import Task, Example, Grid
from ..grid.serializer import grid_to_ascii, format_example_pair


SYSTEM_PROMPT = """You are an expert at solving ARC (Abstraction and Reasoning Corpus) puzzles.

ARC puzzles involve transforming input grids into output grids by discovering the underlying pattern from training examples. Each grid is a 2D array of colors (0-9):
- 0: black
- 1: blue
- 2: red
- 3: green
- 4: yellow
- 5: gray
- 6: magenta
- 7: orange
- 8: cyan
- 9: maroon

Your task is to:
1. Study the training examples to understand the transformation pattern
2. Apply that pattern to the test input to produce the correct output
3. Use the provided tools to manipulate the grid step by step
4. Submit your solution when complete

IMPORTANT GUIDELINES:
- Think step by step about what transformation is happening
- Consider: rotations, reflections, translations, color changes, pattern detection, object manipulation
- Test your hypothesis against ALL training examples before applying to the test input
- Use the tools provided to make changes to the grid
- When you're confident in your solution, use the submit tool

Available tools will be provided. Use them to manipulate the grid."""


def build_system_prompt(tools_description: Optional[str] = None) -> str:
    """Build the full system prompt with optional tools description."""
    prompt = SYSTEM_PROMPT
    if tools_description:
        prompt += f"\n\n{tools_description}"
    return prompt


def format_few_shots(examples: List[Example]) -> str:
    """Format training examples as few-shot demonstrations."""
    lines = ["Training Examples:", "=" * 40]

    for i, example in enumerate(examples, 1):
        lines.append(format_example_pair(example.input, example.output, i))
        lines.append("")

    return "\n".join(lines)


def format_test_input(grid: Grid) -> str:
    """Format the test input grid."""
    lines = ["Test Input:", "=" * 40]
    lines.append(grid_to_ascii(grid))
    lines.append("")
    lines.append("Transform this input to produce the correct output.")
    return "\n".join(lines)


def format_grid_state(grid: Grid, selected_cells=None, label: str = "Current Grid State") -> str:
    """Format the current grid state for the context."""
    lines = [f"<grid_state>", f"{label}:"]
    lines.append(grid_to_ascii(grid, selected_cells))
    if selected_cells:
        lines.append(f"Selected cells: {sorted(selected_cells)}")
    lines.append("</grid_state>")
    return "\n".join(lines)


def format_tool_call(tool_name: str, arguments: dict) -> str:
    """Format a tool call for the trace."""
    args_str = ", ".join(f"{k}={v}" for k, v in arguments.items())
    return f'<tool name="{tool_name}">{args_str}</tool>'


def format_tool_result(result_message: str) -> str:
    """Format a tool result for the trace."""
    return f"<result>{result_message}</result>"


def format_thought(thought: str) -> str:
    """Format a thought/reasoning step."""
    return f"<thought>{thought}</thought>"


def build_initial_context(task: Task, test_index: int = 0) -> str:
    """Build the initial context for solving a task."""
    lines = []

    # Training examples (few-shots)
    lines.append(format_few_shots(task.train))
    lines.append("")

    # Test input
    test_example = task.test[test_index]
    lines.append(format_test_input(test_example.input))
    lines.append("")

    # Initial grid state
    lines.append(format_grid_state(test_example.input, label="Starting Grid (your working copy)"))

    return "\n".join(lines)
