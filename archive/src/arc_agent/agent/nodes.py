"""Graph nodes for the ARC agent."""

from typing import Dict, Any, List

from ..llm.client import LLMClient
from ..llm.prompts import (
    build_system_prompt,
    format_few_shots,
    format_test_input,
    format_grid_state,
    format_tool_call,
    format_tool_result,
)
from ..llm.parser import parse_response, ToolCall
from ..tools.registry import ToolRegistry
from ..tools.submit import SubmitTool
from .state import AgentState


def build_messages(state: AgentState, tools_description: str) -> List[Dict[str, Any]]:
    """Build the message list for the LLM from current state."""
    messages = []

    # System message
    system_content = build_system_prompt(tools_description)
    messages.append({"role": "system", "content": system_content})

    # User message with task context
    task = state["task"]
    test_index = state["test_index"]
    test_example = task.test[test_index]

    user_content_parts = [
        format_few_shots(task.train),
        "",
        format_test_input(test_example.input),
        "",
        "Think step by step about the pattern, then use tools to transform the grid.",
        "When ready, use the submit tool to submit your solution.",
    ]
    messages.append({"role": "user", "content": "\n".join(user_content_parts)})

    # Add conversation history (tool calls and results)
    messages.extend(state["messages"])

    # Add current grid state
    grid_state = state["grid_state"]
    grid_state_content = format_grid_state(
        grid_state.grid,
        grid_state.selected_cells if grid_state.selected_cells else None,
        label="Current Grid State"
    )
    messages.append({"role": "user", "content": grid_state_content})

    return messages


def reason_node(
    state: AgentState,
    client: LLMClient,
    registry: ToolRegistry,
) -> Dict[str, Any]:
    """Reasoning node - calls LLM to get next action.

    Args:
        state: Current agent state
        client: LLM client
        registry: Tool registry

    Returns:
        State updates with tool calls
    """
    # Build messages
    tools_description = registry.get_tools_description()
    messages = build_messages(state, tools_description)

    # Get tool schemas for function calling
    tools = registry.get_openai_tools()

    # Call LLM
    response = client.chat(messages=messages, tools=tools)

    # Parse response
    parsed = parse_response(response)

    # Build message updates
    new_messages = []

    # Add assistant message with thought
    assistant_content = ""
    if parsed.thought:
        assistant_content = f"<thought>{parsed.thought}</thought>"

    if parsed.tool_calls:
        # Format tool calls for context
        tool_call_strs = []
        for tc in parsed.tool_calls:
            tool_call_strs.append(format_tool_call(tc.name, tc.arguments))
        assistant_content += "\n" + "\n".join(tool_call_strs)

    if assistant_content:
        new_messages.append({"role": "assistant", "content": assistant_content})

    # If no tool calls, add a prompt to encourage action
    if not parsed.tool_calls:
        new_messages.append({
            "role": "user",
            "content": "You must call a tool to proceed. Use edit() to modify cells, or submit() when done. Do not just think - take action now."
        })

    return {
        "messages": new_messages,
        "pending_tool_calls": parsed.tool_calls,
        "step_count": state["step_count"] + 1,
    }


def execute_tools_node(
    state: AgentState,
    registry: ToolRegistry,
) -> Dict[str, Any]:
    """Execute pending tool calls.

    Args:
        state: Current agent state
        registry: Tool registry

    Returns:
        State updates with tool results
    """
    tool_calls = state["pending_tool_calls"]
    grid_state = state["grid_state"]
    target_grid = state["target_grid"]

    tool_results = []
    new_messages = []
    done = False
    success = None
    final_message = None

    for tc in tool_calls:
        # Special handling for submit tool - set target
        tool = registry.get(tc.name)
        if isinstance(tool, SubmitTool):
            tool.set_target(target_grid)

        # Execute tool
        result = registry.execute(tc.name, grid_state, **tc.arguments)

        tool_results.append({
            "tool": tc.name,
            "arguments": tc.arguments,
            "success": result.success,
            "message": result.message,
            "data": result.data,
        })

        # Add result to messages
        new_messages.append({
            "role": "user",
            "content": format_tool_result(result.message),
        })

        # Check for terminal state
        if result.is_terminal:
            done = True
            final_message = result.message
            if result.data and "correct" in result.data:
                success = result.data["correct"]

    return {
        "messages": new_messages,
        "last_tool_results": tool_results,
        "pending_tool_calls": [],
        "done": done,
        "success": success,
        "final_message": final_message,
    }


def check_done_node(state: AgentState) -> Dict[str, Any]:
    """Check if the agent should stop.

    Args:
        state: Current agent state

    Returns:
        State updates if max steps reached
    """
    if state["done"]:
        return {}

    if state["step_count"] >= state["max_steps"]:
        return {
            "done": True,
            "success": False,
            "final_message": f"Max steps ({state['max_steps']}) reached without submitting.",
        }

    return {}


def should_continue(state: AgentState) -> str:
    """Determine next node based on state.

    Returns:
        "continue" to keep reasoning, "end" to stop
    """
    if state["done"]:
        return "end"
    return "continue"
