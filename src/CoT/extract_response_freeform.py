"""Parser for free-form CoT: reasoning prose + terminal `{"output": ...}` block.

Strategy: prefer the LAST balanced `{...}` block in the response (the answer);
fall back to fenced ```json``` blocks; tolerate models that ignore the
"no fences" instruction.
"""

import json
import re
from shared.types import Grid


def extract_response(response: str) -> Grid:
    text = response.strip()

    # Strip a chat-template thinking sentinel if any provider leaks it.
    if "<｜end▁of▁thinking｜>" in text:
        text = text.rsplit("<｜end▁of▁thinking｜>", 1)[1].strip()

    # Try fenced ```json``` first (some models still wrap despite the prompt).
    fence_matches = list(re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL))
    if fence_matches:
        candidate = fence_matches[-1].group(1)
    else:
        # Free-form: take the last balanced `{...}` block.
        last_close = text.rfind("}")
        if last_close < 0:
            raise ValueError("no JSON object found in response")
        # Walk backward from last_close to find the matching `{`.
        depth = 0
        start = -1
        for i in range(last_close, -1, -1):
            ch = text[i]
            if ch == "}":
                depth += 1
            elif ch == "{":
                depth -= 1
                if depth == 0:
                    start = i
                    break
        if start < 0:
            raise ValueError("unbalanced JSON braces in response")
        candidate = text[start:last_close + 1]

    parsed = json.loads(candidate, strict=False)
    return parsed["output"]
