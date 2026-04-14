#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from semantha_core import SemanthaWorkspace


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-shot SEmantha CLI: bundle candidates, create a default resume plan, render TeX, and compile PDF."
    )
    parser.add_argument("--label", required=True, help="Output label prefix")
    parser.add_argument("--role-family")
    parser.add_argument("--query")
    parser.add_argument("--target-text")
    parser.add_argument("--target-file")
    parser.add_argument("--top", type=int, default=6)
    parser.add_argument("--plan-top-n", type=int, default=4)
    parser.add_argument("--max-projects", type=int, default=4)
    parser.add_argument("--max-bullets-per-project", type=int, default=3)
    args = parser.parse_args()

    workspace = SemanthaWorkspace(Path(__file__).resolve().parent)
    bundle = workspace.build_resume_bundle(
        label=args.label,
        query=args.query,
        target_text=args.target_text,
        target_file=args.target_file,
        role_family=args.role_family,
        top=args.top,
    )
    plan = workspace.create_resume_plan(label=args.label, top_n=args.plan_top_n)
    render = workspace.render_resume_tex(
        label=args.label,
        max_projects=args.max_projects,
        max_bullets_per_project=args.max_bullets_per_project,
    )
    compile_result = workspace.compile_resume_pdf(label=args.label)

    print(bundle["selected_path"])
    print(plan["resume_plan_path"])
    print(render["tex_path"])
    print(compile_result["pdf_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
