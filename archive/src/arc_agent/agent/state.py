"""Agent state definition for LangGraph."""

from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
import operator

from ..data.types import Task, Grid
from ..grid.state import GridState
from ..llm.parser import ToolCall


def merge_messages(left: List[Dict[str, Any]], right: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge message lists by appending right to left."""
    return left + right


class AgentState(TypedDict):
    """State for the ARC agent graph.

    This TypedDict defines all the state that flows through the LangGraph.
    """

    # Task context
    task: Task
    test_index: int

    # Grid state
    grid_state: GridState
    target_grid: Grid

    # Message history for LLM context
    messages: Annotated[List[Dict[str, Any]], merge_messages]

    # Current step tracking
    step_count: int
    max_steps: int

    # Tool execution results
    pending_tool_calls: List[ToolCall]
    last_tool_results: List[Dict[str, Any]]

    # Termination
    done: bool
    success: Optional[bool]
    final_message: Optional[str]


def create_initial_state(
    task: Task,
    test_index: int = 0,
    max_steps: int = 50,
) -> AgentState:
    """Create the initial agent state for a task.

    Args:
        task: The ARC task to solve
        test_index: Which test example to solve (usually 0)
        max_steps: Maximum number of reasoning steps

    Returns:
        Initial AgentState
    """
    test_example = task.test[test_index]

    return AgentState(
        task=task,
        test_index=test_index,
        grid_state=GridState.from_grid(test_example.input),
        target_grid=test_example.output,
        messages=[],
        step_count=0,
        max_steps=max_steps,
        pending_tool_calls=[],
        last_tool_results=[],
        done=False,
        success=None,
        final_message=None,
    )
