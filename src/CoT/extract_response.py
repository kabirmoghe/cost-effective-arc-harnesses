import json
import re
from shared.types import Grid


def extract_response(response: str) -> Grid:
    """Extract the output grid from a CoT JSON response.

    Handles both clean JSON and JSON embedded in markdown code blocks.
    """
    text = response.strip()

    # DeepSeek V4-flash leaks its thinking channel ahead of the JSON answer
    # even with `thinking: disabled`, when the prompt asks for any reasoning.
    # The separator is the chat-template token; if present, the real answer is
    # what comes after the LAST occurrence.
    if "<｜end▁of▁thinking｜>" in text:
        text = text.rsplit("<｜end▁of▁thinking｜>", 1)[1].strip()

    # Strip markdown code fences if present (V3.2 / V4-flash style)
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    elif not text.lstrip().startswith("{"):
        # Kimi K2 style: reasoning prose followed by a raw JSON object with
        # no markdown fences. Find the last balanced `{...}` block in the text.
        last_open = text.rfind("{")
        last_close = text.rfind("}")
        if 0 <= last_open < last_close:
            text = text[last_open:last_close + 1]

    # strict=False permits raw control characters inside JSON string values
    # (another V4-flash quirk — it emits literal `\n` inside `reasoning`).
    parsed = json.loads(text, strict=False)
    return parsed["output"]
