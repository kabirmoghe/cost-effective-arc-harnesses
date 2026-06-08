# ARC Agent Usage Guide

## Installation

```bash
cd src
uv sync
```

## Configuration

Set your API key as an environment variable:

```bash
# DeepSeek (default)
export DEEPSEEK_API_KEY=your_key

# OpenAI (alternative)
export OPENAI_API_KEY=your_key
```

## CLI Commands

### List Tasks

```bash
# List training tasks
uv run arc-agent list

# Limit output
uv run arc-agent list --limit 10

# List evaluation tasks
uv run arc-agent list --split evaluation
```

### Inspect Task

```bash
# Basic info
uv run arc-agent inspect 007bbfb7

# Show grid contents
uv run arc-agent inspect 007bbfb7 --show-grids

# Hide test outputs (for blind testing)
uv run arc-agent inspect 007bbfb7 --show-grids --hide-test-output
```

### Run Agent on Single Task

```bash
# Run with DeepSeek (default) - shows thoughts, tool calls, results
uv run arc-agent run 007bbfb7

# Run with OpenAI
uv run arc-agent run 007bbfb7 --provider openai

# Show final grid comparison
uv run arc-agent run 007bbfb7 --show-grid

# Quiet mode (suppress step-by-step output)
uv run arc-agent run 007bbfb7 -q

# Custom step limit
uv run arc-agent run 007bbfb7 --max-steps 100

# Run specific test index
uv run arc-agent run 007bbfb7 --test-index 0
```

### Evaluate Multiple Tasks

```bash
# Evaluate first 10 training tasks
uv run arc-agent evaluate --limit 10

# Evaluate specific tasks
uv run arc-agent evaluate --task-ids 007bbfb7,00d62c1b

# Save results to file
uv run arc-agent evaluate --limit 10 -o results.json

# Quiet mode
uv run arc-agent evaluate --limit 10 -q

# Evaluate on evaluation split
uv run arc-agent evaluate --split evaluation --limit 5
```

## Programmatic Usage

```python
from arc_agent.data import load_task
from arc_agent.agent import AgentRunner
from arc_agent.config import get_default_config

# Load a task
task = load_task("007bbfb7", "training")

# Run agent
runner = AgentRunner()
result = runner.run(task, verbose=True)

print(f"Success: {result.success}")
print(f"Steps: {result.steps}")
```

## Tools Available to Agent

| Tool | Description |
|------|-------------|
| `select(cells)` | Select cells by [row, col] coordinates |
| `edit(row, col, color)` | Set single cell to color (0-9) |
| `flood_fill(row, col, color)` | Fill connected region |
| `change_color(color)` | Change all selected cells |
| `reset_grid()` | Reset to original input |
| `resize_grid(rows, cols)` | Resize grid to new dimensions (1-30) |
| `submit()` | Submit solution |

## Color Codes

| Value | Color |
|-------|-------|
| 0 | black |
| 1 | blue |
| 2 | red |
| 3 | green |
| 4 | yellow |
| 5 | gray |
| 6 | magenta |
| 7 | orange |
| 8 | cyan |
| 9 | maroon |
