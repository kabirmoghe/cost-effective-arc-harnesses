"""CLI entry point for the ARC agent."""

import argparse
import sys
from pathlib import Path

from .config import get_default_config, AgentConfig, ModelConfig
from .data.loader import load_task, get_task_ids
from .agent.runner import AgentRunner
from .eval.harness import EvaluationHarness


def cmd_run(args):
    """Run the agent on a single task."""
    # Load task
    try:
        task = load_task(args.task_id, args.split)
    except FileNotFoundError:
        print(f"Task not found: {args.task_id}")
        return 1

    # Create config
    config = get_default_config(args.provider)
    if args.max_steps:
        config.max_steps = args.max_steps

    # Run agent
    runner = AgentRunner(config=config)
    result = runner.run(task, test_index=args.test_index, verbose=True)#not args.quiet)

    # Print result
    print("\n" + "=" * 40)
    print("RESULT")
    print("=" * 40)
    print(f"Task: {result.task_id}")
    print(f"Success: {result.success}")
    print(f"Steps: {result.steps}")
    print(f"Message: {result.final_message}")

    if args.show_grid:
        print("\nFinal Grid:")
        for row in result.final_grid:
            print(" ".join(str(c) for c in row))
        print("\nTarget Grid:")
        for row in result.target_grid:
            print(" ".join(str(c) for c in row))

    return 0 if result.success else 1


def cmd_evaluate(args):
    """Run evaluation on a task split."""
    # Create config
    config = get_default_config(args.provider)
    if args.max_steps:
        config.max_steps = args.max_steps

    # Run evaluation
    harness = EvaluationHarness(config=config, verbose=not args.quiet)

    task_ids = args.task_ids.split(",") if args.task_ids else None
    result = harness.evaluate_split(
        split=args.split,
        limit=args.limit,
        task_ids=task_ids,
    )

    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        result.save(output_path)
        print(f"\nResults saved to: {output_path}")

    return 0


def cmd_list(args):
    """List available tasks."""
    task_ids = get_task_ids(args.split)

    if args.limit:
        task_ids = task_ids[:args.limit]

    print(f"Tasks in {args.split} split ({len(task_ids)} shown):")
    for tid in task_ids:
        print(f"  {tid}")

    return 0


def cmd_inspect(args):
    """Inspect a task's structure."""
    try:
        task = load_task(args.task_id, args.split)
    except FileNotFoundError:
        print(f"Task not found: {args.task_id}")
        return 1

    print(f"Task: {task.task_id}")
    print(f"Training examples: {task.num_train}")
    print(f"Test examples: {task.num_test}")

    print("\nTraining Examples:")
    for i, ex in enumerate(task.train):
        input_shape = f"{len(ex.input)}x{len(ex.input[0])}"
        output_shape = f"{len(ex.output)}x{len(ex.output[0])}"
        print(f"  {i}: Input {input_shape} -> Output {output_shape}")

    print("\nTest Examples:")
    for i, ex in enumerate(task.test):
        input_shape = f"{len(ex.input)}x{len(ex.input[0])}"
        output_shape = f"{len(ex.output)}x{len(ex.output[0])}"
        print(f"  {i}: Input {input_shape} -> Output {output_shape}")

    if args.show_grids:
        from .grid.serializer import grid_to_ascii

        print("\n" + "=" * 40)
        print("TRAINING EXAMPLES")
        print("=" * 40)
        for i, ex in enumerate(task.train):
            print(f"\nExample {i} Input:")
            print(grid_to_ascii(ex.input))
            print(f"\nExample {i} Output:")
            print(grid_to_ascii(ex.output))

        print("\n" + "=" * 40)
        print("TEST EXAMPLES")
        print("=" * 40)
        for i, ex in enumerate(task.test):
            print(f"\nTest {i} Input:")
            print(grid_to_ascii(ex.input))
            if not args.hide_test_output:
                print(f"\nTest {i} Output:")
                print(grid_to_ascii(ex.output))

    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="arc_agent",
        description="ARC AGI Baseline Agent - Solve ARC tasks with LLM tool use",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run agent on a single task")
    run_parser.add_argument("task_id", help="Task ID to run")
    run_parser.add_argument("--split", default="training", choices=["training", "evaluation"])
    run_parser.add_argument("--test-index", type=int, default=0, help="Which test example to solve")
    run_parser.add_argument("--provider", default="deepseek", choices=["deepseek", "deepseek-reasoner", "openai", "openai-o1"])
    run_parser.add_argument("--max-steps", type=int, help="Maximum reasoning steps")
    run_parser.add_argument("--show-grid", action="store_true", help="Show final and target grids")
    run_parser.add_argument("--quiet", "-q", action="store_true", help="Suppress step-by-step output")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation on task split")
    eval_parser.add_argument("--split", default="training", choices=["training", "evaluation"])
    eval_parser.add_argument("--limit", type=int, help="Limit number of tasks")
    eval_parser.add_argument("--task-ids", help="Comma-separated task IDs to evaluate")
    eval_parser.add_argument("--provider", default="deepseek", choices=["deepseek", "deepseek-reasoner", "openai", "openai-o1"])
    eval_parser.add_argument("--max-steps", type=int, help="Maximum reasoning steps")
    eval_parser.add_argument("--output", "-o", help="Output file for results (JSON)")
    eval_parser.add_argument("--quiet", "-q", action="store_true", help="Reduce output")

    # List command
    list_parser = subparsers.add_parser("list", help="List available tasks")
    list_parser.add_argument("--split", default="training", choices=["training", "evaluation"])
    list_parser.add_argument("--limit", type=int, help="Limit number shown")

    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a task")
    inspect_parser.add_argument("task_id", help="Task ID to inspect")
    inspect_parser.add_argument("--split", default="training", choices=["training", "evaluation"])
    inspect_parser.add_argument("--show-grids", action="store_true", help="Show grid contents")
    inspect_parser.add_argument("--hide-test-output", action="store_true", help="Hide test outputs")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "run":
        return cmd_run(args)
    elif args.command == "evaluate":
        return cmd_evaluate(args)
    elif args.command == "list":
        return cmd_list(args)
    elif args.command == "inspect":
        return cmd_inspect(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
