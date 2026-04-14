#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


def extract_project_id(text: str, fallback: str) -> str:
    match = re.search(r"^project_id:\s*(.+)$", text, flags=re.MULTILINE)
    if not match:
        return fallback
    value = match.group(1).strip().strip("'\"")
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value) or fallback


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy repo-local resume-project markdown files into a central data/projects directory."
    )
    parser.add_argument(
        "--source-root",
        required=True,
        help="Root directory to search for resume-project.md files",
    )
    parser.add_argument(
        "--output-dir",
        default="data/projects",
        help="Destination directory for normalized project markdown",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for path in sorted(source_root.glob("**/resume-project.md")):
        text = path.read_text(encoding="utf-8")
        fallback = path.parent.name.lower().replace("__", "-")
        project_id = extract_project_id(text, fallback)
        destination = output_dir / f"{project_id}.md"
        destination.write_text(text, encoding="utf-8")
        count += 1
        print(f"synced\t{path}\t->\t{destination}")

    print(f"synced_total\t{count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
