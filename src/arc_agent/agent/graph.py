"""LangGraph state graph definition for ARC agent."""

from functools import partial
from typing import Optional

from langgraph.graph import StateGraph, END

from ..llm.client import LLMClient
from ..tools.registry import ToolRegistry, get_default_registry
from .state import AgentState
from .nodes import reason_node, execute_tools_node, check_done_node, should_continue


def create_agent_graph(
    client: LLMClient,
    registry: Optional[ToolRegistry] = None,
) -> StateGraph:
    """Create the LangGraph state graph for the ARC agent.

    The graph has the following structure:
        [reason] --> [execute_tools] --> [check_done] --> [reason] (loop)
                                              |
                                              v
                                            [END]

    Args:
        client: LLM client for reasoning
        registry: Tool registry (uses default if not provided)

    Returns:
        Compiled StateGraph
    """
    if registry is None:
        registry = get_default_registry()

    # Create the graph
    graph = StateGraph(AgentState)

    # Add nodes with bound dependencies
    graph.add_node("reason", partial(reason_node, client=client, registry=registry))
    graph.add_node("execute_tools", partial(execute_tools_node, registry=registry))
    graph.add_node("check_done", check_done_node)

    # Add edges
    graph.add_edge("reason", "execute_tools")
    graph.add_edge("execute_tools", "check_done")

    # Add conditional edge from check_done
    graph.add_conditional_edges(
        "check_done",
        should_continue,
        {
            "continue": "reason",
            "end": END,
        }
    )

    # Set entry point
    graph.set_entry_point("reason")

    return graph.compile()
