import json
import re
from shared.types import Grid


def extract_response(response: str) -> Grid:
    """Extract the output grid from a CoT JSON response.

    Handles both clean JSON and JSON embedded in markdown code blocks.
    """
    text = response.strip()

    # Strip markdown code fences if present
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)

    parsed = json.loads(text)
    return parsed["output"]
