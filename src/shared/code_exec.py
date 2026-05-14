"""Safe-ish execution of generated transform() code in a subprocess."""

import json
import subprocess
import sys
import textwrap
from typing import List

Grid = List[List[int]]

_RUNNER_TEMPLATE = textwrap.dedent("""\
    import json, sys

    {code}

    grid = json.loads(sys.stdin.read())
    result = transform(grid)
    print(json.dumps(result))
""")


def execute_transformation(
    code: str,
    grid: Grid,
    timeout: float = 5.0,
) -> tuple[Grid | None, str | None]:
    """Run generated transform() code in a subprocess.

    Returns (result_grid, None) on success or (None, error_string) on failure.
    """
    script = _RUNNER_TEMPLATE.format(code=code)
    grid_json = json.dumps(grid)

    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            input=grid_json,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, f"Execution timed out after {timeout}s"

    if proc.returncode != 0:
        return None, proc.stderr.strip()

    try:
        result = json.loads(proc.stdout.strip())
    except (json.JSONDecodeError, ValueError) as e:
        return None, f"Failed to parse output as grid: {e}\nstdout: {proc.stdout.strip()}"

    return result, None
