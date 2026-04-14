#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rank_projects import (
    attach_family,
    attach_context,
    build_match_report,
    extract_ranking_metadata,
    load_record,
    load_family_map,
    normalize_label,
    load_target_text,
    parse_sections,
    rank_projects,
    should_include_record,
    split_frontmatter,
)
from resume_budget import estimate_project_budget, estimate_resume_budget
from semantic_features import SemanticConfig


RESUME_PLAN_SCHEMA_VERSION = "resume-plan/v1"
DEFAULT_PLAN_PROJECT_COUNT = 4
PLAN_OVERRIDE_FIELDS = {
    "display_name",
    "source_display_name",
    "family_name",
    "family_role",
    "motivation_summary",
    "problem_trigger",
    "personal_connection",
    "why_now",
    "constraints_or_stakes",
    "synthesis",
    "resume_keywords",
    "elevator_summary",
    "why_it_matters",
    "technical_highlights",
    "suggested_bullet_points",
    "bullet_candidates",
    "caveats",
    "role_family_targets",
}
PLAN_LIST_OVERRIDE_FIELDS = {
    "suggested_bullet_points",
    "bullet_candidates",
    "caveats",
    "role_family_targets",
}


def load_profile(
    profile_file: Path | None,
) -> tuple[dict[str, Any], dict[str, str], str]:
    if profile_file is None or not profile_file.exists():
        return {}, {}, ""
    text = profile_file.read_text(encoding="utf-8")
    frontmatter, body = split_frontmatter(text)
    return frontmatter, parse_sections(body), text


def build_project_bundle(
    results: list[dict[str, Any]],
    projects_dir: Path,
    context_dir: Path,
    *,
    top_limit: int | None = None,
    unique_families: bool = True,
) -> list[dict[str, Any]]:
    bundle: list[dict[str, Any]] = []
    seen_families: set[str] = set()
    for item in results:
        family_id = str(item.get("family_id") or "").strip()
        if unique_families and family_id and family_id in seen_families:
            continue
        record = load_record(projects_dir / Path(item["path"]).name)
        context_frontmatter: dict[str, Any] = {}
        context_sections: dict[str, str] = {}
        for context_path in [
            context_dir / f"{item['project_id']}.md",
            context_dir / f"{Path(item['path']).stem}.md",
        ]:
            if context_path.exists():
                context_frontmatter, context_body = split_frontmatter(
                    context_path.read_text(encoding="utf-8")
                )
                context_sections = parse_sections(context_body)
                break
        project = {
            "project_id": item["project_id"],
            "display_name": item["display_name"],
            "source_display_name": item.get(
                "source_display_name", item["display_name"]
            ),
            "repo": item["repo"],
            "score": item.get("diversified_score", item["final_score"]),
            "base_score": item["final_score"],
            "technical_impressiveness": item.get("technical_impressiveness", 1.0),
            "cosine_title": item.get("cosine_title", ""),
            "family_id": item.get("family_id"),
            "family_name": item.get("family_name"),
            "family_role": item.get("family_role"),
            "motivation_summary": item.get("motivation_summary", ""),
            "problem_trigger": context_frontmatter.get("problem_trigger", ""),
            "personal_connection": context_frontmatter.get("personal_connection", ""),
            "why_now": context_frontmatter.get("why_now", ""),
            "constraints_or_stakes": context_frontmatter.get(
                "constraints_or_stakes", ""
            ),
            "context_keywords": context_frontmatter.get("context_keywords", []),
            "why_this_existed": context_sections.get("why this existed", ""),
            "personal_context": context_sections.get("personal context", ""),
            "resume_intent": context_sections.get("resume intent", ""),
            "synthesis": context_sections.get("synthesis", ""),
            "suggested_bullet_points": extract_list(
                context_sections.get("suggested bullet points", "")
            ),
            "non_obvious_context": context_sections.get(
                "things not obvious from the code", ""
            ),
            "resume_keywords": record.sections.get("resume keywords", ""),
            "elevator_summary": record.sections.get("elevator summary", ""),
            "why_it_matters": record.sections.get("why it matters", ""),
            "technical_highlights": record.sections.get("technical highlights", ""),
            "bullet_candidates": extract_list(
                record.sections.get("resume bullet candidates", "")
            ),
            "caveats": extract_list(record.sections.get("caveats", "")),
            "role_family_targets": record.frontmatter.get("role_family_targets", []),
            "lexical_score": item.get("lexical_score", 0.0),
            "semantic_score": item.get("semantic_score", 0.0),
            "chunk_score": item.get("chunk_score", 0.0),
            "keyword_score": item.get("keyword_score", 0.0),
            "heuristic_score": item.get("heuristic_score", 0.0),
            "rerank_score": item.get("rerank_score", 0.0),
            "score_breakdown": item.get("score_breakdown", {}),
            "match_report": item.get("match_report", {}),
            "ranking_diagnostics": item.get("ranking_diagnostics", []),
        }
        chosen_bullets = (
            project["suggested_bullet_points"] or project["bullet_candidates"][:3]
        )
        project["budget_estimate"] = estimate_project_budget(project, chosen_bullets)
        bundle.append(project)
        if family_id:
            seen_families.add(family_id)
        if top_limit is not None and len(bundle) >= top_limit:
            break
    return bundle


def extract_list(text: str) -> list[str]:
    items: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            items.append(stripped[2:].strip())
    return items


def load_selected_projects(selected_file: Path) -> list[dict[str, Any]]:
    payload = json.loads(selected_file.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        selected = payload.get("selected_projects")
        if isinstance(selected, list):
            return selected
    raise ValueError("selected_file must contain a selected project list")


def select_ranked_results_for_bundle_report(
    ranked: list[dict[str, Any]], selected_projects: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    ranked_index = {
        str(item.get("project_id") or "").strip(): item
        for item in ranked
        if str(item.get("project_id") or "").strip()
    }
    ordered: list[dict[str, Any]] = []
    for project in selected_projects:
        project_id = str(project.get("project_id") or "").strip()
        matched = ranked_index.get(project_id)
        if matched is not None:
            ordered.append(matched)
    return ordered


def validate_plan_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    validated: dict[str, Any] = {}
    for key, value in overrides.items():
        if key not in PLAN_OVERRIDE_FIELDS:
            raise ValueError(f"Unsupported resume-plan override field: {key}")
        if key in PLAN_LIST_OVERRIDE_FIELDS:
            if not isinstance(value, list) or any(
                not isinstance(item, str) for item in value
            ):
                raise ValueError(f"{key} override must be a list of strings")
        elif not isinstance(value, str):
            raise ValueError(f"{key} override must be a string")
        validated[key] = value
    return validated


def build_resume_plan(
    *,
    selected_projects: list[dict[str, Any]],
    label: str,
    chosen_project_ids: list[str] | None = None,
    project_overrides: dict[str, dict[str, Any]] | None = None,
    top_n: int = DEFAULT_PLAN_PROJECT_COUNT,
    source_selected_path: str | None = None,
) -> dict[str, Any]:
    if not isinstance(selected_projects, list) or not selected_projects:
        raise ValueError("selected_projects must be a non-empty list")

    project_index = {
        str(project.get("project_id") or "").strip(): (idx, project)
        for idx, project in enumerate(selected_projects)
        if str(project.get("project_id") or "").strip()
    }
    if not project_index:
        raise ValueError("selected_projects is missing project_id values")

    if chosen_project_ids is None:
        chosen_ids = [
            project["project_id"] for project in selected_projects[: max(1, top_n)]
        ]
    else:
        if not chosen_project_ids:
            raise ValueError("chosen_project_ids must not be empty")
        chosen_ids = chosen_project_ids
    ordered_plan_projects: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    project_overrides = project_overrides or {}

    for project_id in chosen_ids:
        normalized_id = str(project_id or "").strip()
        if not normalized_id:
            raise ValueError("chosen_project_ids must not contain empty values")
        if normalized_id in seen_ids:
            raise ValueError(f"Duplicate project_id in resume plan: {normalized_id}")
        source = project_index.get(normalized_id)
        if source is None:
            raise ValueError(f"Project not found in selected bundle: {normalized_id}")
        source_rank, _project = source
        overrides = project_overrides.get(normalized_id) or {}
        if not isinstance(overrides, dict):
            raise ValueError(f"Overrides for {normalized_id} must be an object")
        ordered_plan_projects.append(
            {
                "project_id": normalized_id,
                "source_rank": source_rank + 1,
                "overrides": validate_plan_overrides(overrides),
            }
        )
        seen_ids.add(normalized_id)

    selection_mode = "manual" if chosen_project_ids else "auto-top-ranked"
    return {
        "schema_version": RESUME_PLAN_SCHEMA_VERSION,
        "label": label,
        "selection_mode": selection_mode,
        "source_selected_path": source_selected_path,
        "project_count": len(ordered_plan_projects),
        "projects": ordered_plan_projects,
    }


def render_prompt(
    *,
    label: str,
    role_family: str | None,
    target_text: str,
    profile_frontmatter: dict[str, Any],
    profile_sections: dict[str, str],
    template_text: str,
    selected_projects: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append(f"# Resume drafting prompt: {label}")
    lines.append("")
    lines.append("## Task for the LLM")
    lines.append("")
    lines.append(
        "Draft a tailored one-page resume in LaTeX using the supplied template, selected project records, and user/profile context."
    )
    lines.append("")
    lines.append("## Hard constraints")
    lines.append("")
    constraints = [
        "Use only facts grounded in the supplied profile, project records, and context overlays.",
        "Do not invent metrics, users, employers, timelines, or production claims.",
        "Prefer the strongest 4-6 projects; do not stuff the page with everything.",
        "Treat project families as one body of work with multiple surfaces; do not waste space repeating near-duplicate family members unless they add distinct value.",
        "Respect caveats on team ownership, school origin, prototype status, or design-only scope.",
        "Optimize wording for relevance and clarity, but stay technically truthful.",
        "Output a complete LaTeX document that fills the given template placeholders with concrete content.",
    ]
    for item in constraints:
        lines.append(f"- {item}")
    lines.append("")
    if role_family:
        lines.append(f"## Role family bias\n\n- {role_family}\n")

    lines.append("## Target description")
    lines.append("")
    lines.append(target_text.strip() or "No target provided.")
    lines.append("")

    lines.append("## User profile")
    lines.append("")
    if profile_frontmatter:
        for key in ["name", "headline", "location", "email", "github", "phone"]:
            if profile_frontmatter.get(key):
                lines.append(f"- {key}: {profile_frontmatter[key]}")
        lines.append("")
    for section_name in ["summary", "experience", "education", "skills"]:
        value = profile_sections.get(section_name, "").strip()
        if value:
            lines.append(f"### {section_name.title()}")
            lines.append("")
            lines.append(value)
            lines.append("")

    lines.append("## Selected projects")
    lines.append("")
    for idx, project in enumerate(selected_projects, start=1):
        lines.append(f"### {idx}. {project['display_name']}")
        lines.append("")
        if project["source_display_name"] != project["display_name"]:
            lines.append(f"- Source display name: {project['source_display_name']}")
        lines.append(f"- Project ID: {project['project_id']}")
        lines.append(f"- Repo: {project['repo']}")
        lines.append(f"- Ranking score: {project['score']}")
        if project["base_score"] != project["score"]:
            lines.append(
                f"- Base score before family diversification: {project['base_score']}"
            )
        if project["technical_impressiveness"] != 1.0:
            lines.append(
                f"- Technical impressiveness bias: {project['technical_impressiveness']}"
            )
        if project["family_name"]:
            lines.append(f"- Project family: {project['family_name']}")
        if project.get("score_breakdown"):
            breakdown = project["score_breakdown"]
            lines.append(
                "- Score breakdown: "
                f"lexical={breakdown.get('lexical', 0.0)}, "
                f"embedding={breakdown.get('embedding', 0.0)}, "
                f"chunk={breakdown.get('job_chunk', 0.0)}, "
                f"keyword={breakdown.get('keyword_overlap', 0.0)}, "
                f"heuristic={breakdown.get('heuristic_adjustment', 0.0)}, "
                f"cross-encoder={breakdown.get('cross_encoder', 0.0)}"
            )
        keyword_hits = project.get("match_report", {}).get("keyword_hits") or []
        if keyword_hits:
            lines.append(f"- Keyword hits: {', '.join(keyword_hits)}")
        missing_keywords = project.get("match_report", {}).get("missing_keywords") or []
        if missing_keywords:
            lines.append(
                f"- Missing target keywords: {', '.join(missing_keywords[:8])}"
            )
        if project["family_role"]:
            lines.append(f"- Family role: {project['family_role']}")
        if project["cosine_title"]:
            lines.append(f"- Retrieval title: {project['cosine_title']}")
        if project["motivation_summary"]:
            lines.append(f"- Why it existed: {project['motivation_summary']}")
        if project["problem_trigger"]:
            lines.append(f"- Problem trigger: {project['problem_trigger']}")
        if project["personal_connection"]:
            lines.append(f"- Personal connection: {project['personal_connection']}")
        if project["why_now"]:
            lines.append(f"- Why now: {project['why_now']}")
        if project["constraints_or_stakes"]:
            lines.append(f"- Constraints/stakes: {project['constraints_or_stakes']}")
        if project["context_keywords"]:
            lines.append(
                f"- Context keywords: {', '.join(project['context_keywords'])}"
            )
        if project["role_family_targets"]:
            lines.append(f"- Role targets: {', '.join(project['role_family_targets'])}")
        if project["resume_keywords"]:
            lines.append(f"- Resume keywords: {project['resume_keywords']}")
        if project["elevator_summary"]:
            lines.append(f"- Summary: {project['elevator_summary']}")
        if project["why_it_matters"]:
            lines.append(f"- Why it matters: {project['why_it_matters']}")
        if project["synthesis"]:
            lines.append(f"- Synthesis: {project['synthesis']}")
        if project["why_this_existed"]:
            lines.append(f"- Expanded why this existed: {project['why_this_existed']}")
        if project["personal_context"]:
            lines.append(f"- Expanded personal context: {project['personal_context']}")
        if project["resume_intent"]:
            lines.append(f"- Resume intent: {project['resume_intent']}")
        if project["suggested_bullet_points"]:
            lines.append("- Suggested bullet points:")
            for bullet in project["suggested_bullet_points"]:
                lines.append(f"  - {bullet}")
        if project["non_obvious_context"]:
            lines.append(
                f"- Things not obvious from the code: {project['non_obvious_context']}"
            )
        if project["technical_highlights"]:
            lines.append("- Technical highlights:")
            for line in project["technical_highlights"].splitlines():
                stripped = line.strip()
                if stripped:
                    lines.append(f"  {stripped}")
        if project["bullet_candidates"]:
            lines.append("- Resume bullet candidates:")
            for bullet in project["bullet_candidates"]:
                lines.append(f"  - {bullet}")
        if project["caveats"]:
            lines.append("- Caveats:")
            for caveat in project["caveats"]:
                lines.append(f"  - {caveat}")
        budget_estimate = project.get("budget_estimate") or {}
        if budget_estimate.get("warnings"):
            lines.append(f"- Layout warnings: {', '.join(budget_estimate['warnings'])}")
        lines.append("")

    lines.append("## LaTeX template to fill")
    lines.append("")
    lines.append("```tex")
    lines.append(template_text.rstrip())
    lines.append("```")
    lines.append("")
    lines.append("## Required output")
    lines.append("")
    lines.append("Return only the completed LaTeX document.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build an LLM-ready resume drafting prompt from ranked markdown project records."
    )
    parser.add_argument("--projects-dir", default="data/projects")
    parser.add_argument("--context-dir", default="data/context")
    parser.add_argument("--family-dir", default="data/families")
    parser.add_argument("--profile-file", type=Path)
    parser.add_argument("--template-file", type=Path, required=True)
    parser.add_argument("--query")
    parser.add_argument("--target-file", type=Path)
    parser.add_argument("--role-family")
    parser.add_argument("--top", type=int, default=6)
    parser.add_argument("--allow-family-duplicates", action="store_true")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--label", default="resume-prompt")
    args = parser.parse_args()
    args.label = normalize_label(args.label)

    target_text = load_target_text(args.query, args.target_file, args.role_family)
    if not target_text:
        raise SystemExit("Provide --query, --target-file, or --role-family")

    projects_dir = Path(args.projects_dir)
    context_dir = Path(args.context_dir)
    family_dir = Path(args.family_dir)
    project_paths = sorted(projects_dir.glob("*.md"))
    if not project_paths:
        raise SystemExit(f"No markdown project files found in {projects_dir}")

    family_map = load_family_map(family_dir)
    records = [
        attach_family(attach_context(record, context_dir), family_map)
        for path in project_paths
        if (record := load_record(path)) and should_include_record(record)
    ]
    ranked = rank_projects(records, target_text, args.role_family)
    metadata = extract_ranking_metadata(ranked, target_text)
    selected_projects = build_project_bundle(
        ranked,
        projects_dir,
        context_dir,
        top_limit=args.top,
        unique_families=not args.allow_family_duplicates,
    )
    report_results = select_ranked_results_for_bundle_report(ranked, selected_projects)

    profile_frontmatter, profile_sections, _ = load_profile(args.profile_file)
    template_text = args.template_file.read_text(encoding="utf-8")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / f"{args.label}-prompt.md"
    selected_path = output_dir / f"{args.label}-selected.json"
    report_path = output_dir / f"{args.label}-match-report.json"

    prompt_text = render_prompt(
        label=args.label,
        role_family=args.role_family,
        target_text=target_text,
        profile_frontmatter=profile_frontmatter,
        profile_sections=profile_sections,
        template_text=template_text,
        selected_projects=selected_projects,
    )
    prompt_path.write_text(prompt_text, encoding="utf-8")
    selected_path.write_text(json.dumps(selected_projects, indent=2), encoding="utf-8")
    semantic_config = SemanticConfig.from_env()
    match_report = build_match_report(
        label=args.label,
        target_text=target_text,
        role_family=args.role_family,
        semantic_config=semantic_config,
        target_keywords=metadata["target_keywords"],
        expanded_keywords=metadata["expanded_keywords"],
        diagnostics=metadata["diagnostics"],
        results=report_results,
    )
    match_report["selected_bundle_budget"] = estimate_resume_budget(
        [
            {
                "project_id": project["project_id"],
                "budget": project.get("budget_estimate") or {},
            }
            for project in selected_projects
        ]
    )
    report_path.write_text(json.dumps(match_report, indent=2), encoding="utf-8")

    print(prompt_path)
    print(selected_path)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
