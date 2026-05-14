"""Render pipeline artifacts as human-readable markdown.

Usage:
    python -m pipeline.render <json_path>                # print to stdout
    python -m pipeline.render <json_path> --write        # write .md sibling file
    python -m pipeline.render <json_path> -o <out_path>  # write to specific path
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline.io import load_pattern_document, load_transformation_result


def render_json(json_path: Path | str) -> str:
    """Render a pipeline JSON artifact as markdown, auto-detecting the type."""
    name = Path(json_path).name
    if name.startswith("pattern_explorer"):
        return load_pattern_document(json_path).to_markdown()
    elif name.startswith("transformation_definer"):
        return load_transformation_result(json_path).to_markdown()
    else:
        raise ValueError(f"Unknown artifact type: {name}")


def _cli():
    parser = argparse.ArgumentParser(description="Render a pipeline JSON artifact as markdown.")
    parser.add_argument("json_path", type=Path, help="Path to a pipeline JSON file")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Write markdown to this path")
    parser.add_argument("--write", action="store_true", help="Write .md alongside the input file")
    args = parser.parse_args()

    if not args.json_path.exists():
        print(f"File not found: {args.json_path}", file=sys.stderr)
        sys.exit(1)

    markdown = render_json(args.json_path)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown)
        print(f"Wrote {args.output}")
    elif args.write:
        md_path = args.json_path.with_suffix(".md")
        md_path.write_text(markdown)
        print(f"Wrote {md_path}")
    else:
        print(markdown)


if __name__ == "__main__":
    _cli()
