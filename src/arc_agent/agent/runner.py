"""High-level interface for running the ARC agent."""

import re
from typing import Optional, Dict, Any
from dataclasses import dataclass

from ..config import AgentConfig, get_default_config
from ..data.types import Task
from ..grid.serializer import grid_to_ascii
from ..llm.client import create_client, LLMClient
from ..tools.registry import ToolRegistry, get_default_registry
from .state import AgentState, create_initial_state
from .graph import create_agent_graph


@dataclass
class AgentResult:
    """Result of running the agent on a task."""

    task_id: str
    test_index: int
    success: Optional[bool]
    steps: int
    final_grid: list
    target_grid: list
    final_message: str
    trace: list  # List of (thought, tool_calls, results) tuples


class AgentRunner:
    """High-level runner for the ARC agent."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        client: Optional[LLMClient] = None,
        registry: Optional[ToolRegistry] = None,
    ):
        """Initialize the agent runner.

        Args:
            config: Agent configuration (uses default if not provided)
            client: LLM client (created from config if not provided)
            registry: Tool registry (uses default if not provided)
        """
        self.config = config or get_default_config()
        self.client = client or create_client(self.config.model)
        self.registry = registry or get_default_registry()
        self.graph = create_agent_graph(self.client, self.registry)

    def run(
        self,
        task: Task,
        test_index: int = 0,
        max_steps: Optional[int] = None,
        verbose: bool = False,
    ) -> AgentResult:
        """Run the agent on a task.

        Args:
            task: The ARC task to solve
            test_index: Which test example to solve
            max_steps: Override max steps from config
            verbose: Print progress during execution

        Returns:
            AgentResult with success status and details
        """
        steps = max_steps or self.config.max_steps
        initial_state = create_initial_state(task, test_index, steps)

        if verbose:
            print(f"Running task {task.task_id} (test {test_index})")
            print(f"Grid size: {initial_state['grid_state'].rows}x{initial_state['grid_state'].cols}")
            print()

        # Run the graph with streaming for live output
        trace = []
        final_state = dict(initial_state)  # Start with initial state

        for event in self.graph.stream(initial_state, stream_mode="updates"):
            for node_name, updates in event.items():
                if updates is None:
                    continue

                # Merge updates into final_state
                final_state.update(updates)

                # Live output for reason node
                if node_name == "reason" and verbose:
                    step = updates.get("step_count", "?")
                    print(f"--- Step {step} ---")

                    # Check for thought in new messages
                    for msg in updates.get("messages", []):
                        if msg.get("role") == "assistant":
                            content = msg.get("content", "")
                            # Print the LLM's reasoning (content before tool calls)
                            if content:
                                # Remove XML tags for cleaner output
                                clean = re.sub(r"<[^>]+>", "", content).strip()
                                if clean:
                                    print(f"Thinking: {clean[:1500]}{'...' if len(clean) > 500 else ''}")

                    # Print tool calls
                    for tc in updates.get("pending_tool_calls", []):
                        args_str = ", ".join(f"{k}={v}" for k, v in tc.arguments.items())
                        print(f"  -> {tc.name}({args_str})")
                        trace.append({"type": "tool_call", "tool": tc.name, "arguments": tc.arguments})

                # Live output for tool execution
                if node_name == "execute_tools" and verbose:
                    for result in updates.get("last_tool_results", []):
                        status = "OK" if result["success"] else "FAIL"
                        print(f"  <- [{status}] {result['message']}")
                        trace.append({"type": "result", "success": result["success"], "message": result["message"]})

                    # Show current grid vs target after tool execution
                    current_grid = final_state["grid_state"].grid
                    target_grid = final_state["target_grid"]
                    print()
                    print("Current Grid:                Target Grid:")
                    current_lines = grid_to_ascii(current_grid, show_coordinates=False).split('\n')
                    target_lines = grid_to_ascii(target_grid, show_coordinates=False).split('\n')
                    # Pad to same length
                    max_len = max(len(current_lines), len(target_lines))
                    current_lines += [''] * (max_len - len(current_lines))
                    target_lines += [''] * (max_len - len(target_lines))
                    max_width = max(len(line) for line in current_lines) if current_lines else 0
                    for curr, targ in zip(current_lines, target_lines):
                        print(f"{curr:<{max_width}}    |    {targ}")
                    print()

        return AgentResult(
            task_id=task.task_id,
            test_index=test_index,
            success=final_state.get("success"),
            steps=final_state.get("step_count", 0),
            final_grid=final_state["grid_state"].grid,
            target_grid=final_state["target_grid"],
            final_message=final_state.get("final_message", ""),
            trace=trace,
        )


def run_task(
    task: Task,
    test_index: int = 0,
    config: Optional[AgentConfig] = None,
    verbose: bool = False,
) -> AgentResult:
    """Convenience function to run the agent on a task.

    Args:
        task: The ARC task to solve
        test_index: Which test example to solve
        config: Agent configuration
        verbose: Print progress

    Returns:
        AgentResult
    """
    runner = AgentRunner(config=config)
    return runner.run(task, test_index, verbose=verbose)
