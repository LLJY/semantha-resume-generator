from __future__ import annotations

import math
from typing import Any


SUMMARY_WRAP_THRESHOLD = 128
BULLET_WRAP_THRESHOLD = 112


def estimate_text_lines(text: str, line_width: int) -> int:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return 0
    return max(1, math.ceil(len(cleaned) / max(1, line_width)))


def estimate_project_budget(
    project: dict[str, Any], bullets: list[str]
) -> dict[str, Any]:
    summary = str(
        project.get("synthesis")
        or project.get("motivation_summary")
        or project.get("elevator_summary")
        or project.get("cosine_title")
        or project.get("display_name")
        or ""
    )
    summary_lines = estimate_text_lines(summary, SUMMARY_WRAP_THRESHOLD)
    bullet_lines = [
        estimate_text_lines(bullet, BULLET_WRAP_THRESHOLD) for bullet in bullets
    ]
    warnings: list[str] = []
    if summary_lines > 2:
        warnings.append("summary-likely-wraps")
    if any(lines > 2 for lines in bullet_lines):
        warnings.append("bullet-likely-wraps")
    return {
        "summary_lines": summary_lines,
        "bullet_lines": bullet_lines,
        "total_lines": 2 + summary_lines + sum(bullet_lines),
        "warnings": warnings,
    }


def estimate_resume_budget(project_entries: list[dict[str, Any]]) -> dict[str, Any]:
    total_project_lines = sum(
        int(item["budget"]["total_lines"]) for item in project_entries
    )
    warnings: list[dict[str, Any]] = []
    for item in project_entries:
        for warning in item["budget"]["warnings"]:
            warnings.append({"project_id": item["project_id"], "warning": warning})
    if total_project_lines > 30:
        warnings.append({"project_id": None, "warning": "project-section-over-budget"})
    return {
        "project_lines_estimate": total_project_lines,
        "warnings": warnings,
    }
