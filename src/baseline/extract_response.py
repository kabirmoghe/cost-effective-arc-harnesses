import json
import re
from shared.types import Grid


def extract_response(response: str) -> Grid:
    """Extract the output grid from a baseline JSON response."""
    text = response.strip()

    # Strip markdown code fences if present
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)

    parsed = json.loads(text)
    return parsed["output"]
