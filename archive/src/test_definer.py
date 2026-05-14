"""Mini end-to-end test for the TransformationDefiner using saved explorer outputs."""

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from shared.loader import load_task
from shared.llm import create_async_client, get_default_model
from pipeline.agents.pattern_explorer.types import PatternDocument, ExplorationResult
from pipeline.agents.transformation_definer.core import define_transformation
from pipeline.io import load_pattern_document

load_dotenv()

TASK_ID = "007bbfb7"
RUN_ID = "20260416-174410"
OUTPUT_DIR = Path("output/pipeline/007bbfb7")


def log(msg: str = ""):
    print(msg, flush=True)


async def main():
    doc_0 = load_pattern_document(OUTPUT_DIR / f"pattern_explorer_{TASK_ID}_{RUN_ID}_0.json")
    doc_1 = load_pattern_document(OUTPUT_DIR / f"pattern_explorer_{TASK_ID}_{RUN_ID}_1.json")
    log(f"Loaded 2 explorer docs: {len(doc_0.patterns)} and {len(doc_1.patterns)} patterns")

    exploration = ExplorationResult(
        task_id=TASK_ID,
        run_id=RUN_ID,
        documents=[doc_0, doc_1],
    )

    task = load_task(TASK_ID, "training")
    log(f"Task {TASK_ID}: {task.num_train} train pairs, {task.num_test} test pairs")

    client = create_async_client("deepseek")
    model = get_default_model("deepseek")
    log(f"Using model: {model}\n")

    result = await define_transformation(
        task=task,
        exploration_result=exploration,
        client=client,
        model=model,
        max_steps=10,
        max_repairs=3,
        log_fn=log,
    )

    log(f"\n{'=' * 60}")
    log(f"Summary: {result.transformation_summary}")
    log(f"Correct: {result.correct}")
    log(f"Repair attempts: {result.repair_attempts}")
    log(f"Final error: {result.final_error}")
    log(f"Tokens: {result.usage['prompt_tokens'] + result.usage['completion_tokens']}")

    if result.code:
        log(f"\nCode:\n{result.code}")

    out_path = OUTPUT_DIR / f"transformation_definer_{TASK_ID}_{RUN_ID}_test.json"
    out_path.write_text(json.dumps(result.to_dict(), indent=2))
    log(f"\nSaved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
