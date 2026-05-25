"""Extract a 2D int grid from an arc_base model response.

The upstream prompt instructs the model to "respond in the format of the
training output examples" — which are raw JSON 2D int arrays. The model may
emit a clean array, wrap in code fences, prefix with prose, or label as
`OUTPUT:`. We scan for all 2D-array-shaped substrings and return the LAST one
that parses as a valid int grid (models commonly emit reasoning first and
final answer last).
"""

import json
import re

from shared.types import Grid


# Match a 2D-array literal: `[` then `[` then any content (non-greedy) then
# `]` then `]`. Adequate for ARC grids — rows are flat, not nested deeper.
_ARRAY_RE = re.compile(r"\[\s*\[[\s\S]*?\]\s*\]")


def _is_int_grid(obj) -> bool:
    if not isinstance(obj, list) or not obj:
        return False
    if not all(isinstance(row, list) and row for row in obj):
        return False
    return all(isinstance(v, int) for row in obj for v in row)


def extract_response(response: str) -> Grid:
    """Return the predicted output grid. Raises ValueError if none found."""
    text = response.strip()

    # Clean-response fast path: the whole thing parses as a grid.
    try:
        parsed = json.loads(text)
        if _is_int_grid(parsed):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Scan for embedded 2D-array substrings; prefer the LAST valid one.
    for chunk in reversed(_ARRAY_RE.findall(text)):
        try:
            parsed = json.loads(chunk)
            if _is_int_grid(parsed):
                return parsed
        except (json.JSONDecodeError, ValueError):
            continue

    raise ValueError(f"No 2D int grid found in response: {text[:200]!r}")
