#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from build_resume_prompt import (
    RESUME_PLAN_SCHEMA_VERSION,
    load_profile,
    load_selected_projects,
)


LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(text: str) -> str:
    return "".join(LATEX_ESCAPES.get(ch, ch) for ch in text)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sentenceify(text: str) -> str:
    text = normalize_whitespace(text)
    return text


def normalize_compare_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def compare_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def text_similarity(a: str, b: str) -> float:
    a_norm = normalize_compare_text(a)
    b_norm = normalize_compare_text(b)
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0
    a_tokens = compare_tokens(a_norm)
    b_tokens = compare_tokens(b_norm)
    if not a_tokens or not b_tokens:
        return 0.0
    overlap = len(a_tokens & b_tokens) / max(1, min(len(a_tokens), len(b_tokens)))
    if a_norm in b_norm or b_norm in a_norm:
        return max(overlap, 0.85)
    return overlap


def visible_project_title(project: dict[str, Any]) -> str:
    return str(
        project.get("display_name") or project.get("source_display_name") or ""
    ).strip()


def first_paragraph(text: str) -> str:
    for part in text.split("\n\n"):
        candidate = normalize_whitespace(part)
        if candidate:
            return candidate
    return normalize_whitespace(text)


def render_paragraph(text: str) -> str:
    cleaned = first_paragraph(text)
    return latex_escape(cleaned)


def render_markdown_bullets(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    output: list[str] = []
    indent_stack: list[int] = []

    def open_level() -> None:
        output.append(r"\begin{itemize}[leftmargin=*,itemsep=0.18em,topsep=0.18em]")

    def close_level() -> None:
        output.append(r"\end{itemize}")

    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if not stripped.startswith("- "):
            output.append(latex_escape(stripped) + r"\\")
            continue

        while indent_stack and indent < indent_stack[-1]:
            close_level()
            indent_stack.pop()
        if not indent_stack or indent > indent_stack[-1]:
            open_level()
            indent_stack.append(indent)
        content = latex_escape(stripped[2:].strip())
        output.append(rf"\item {content}")

    while indent_stack:
        close_level()
        indent_stack.pop()

    return "\n".join(output)


def parse_markdown_outline(text: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        stripped = raw_line.lstrip()
        indent = len(raw_line) - len(stripped)
        if not stripped.startswith("- "):
            continue
        content = stripped[2:].strip()
        if indent == 0:
            current = {"text": content, "children": []}
            entries.append(current)
        elif current is not None:
            current["children"].append(content)
    return entries


def render_simple_bullets(items: list[str]) -> str:
    if not items:
        return ""
    lines = [r"\begin{itemize}"]
    for item in items:
        lines.append(rf"\item {latex_escape(sentenceify(item))}")
    lines.append(r"\end{itemize}")
    return "\n".join(lines)


def split_role_line(text: str) -> tuple[str, str, str]:
    match = re.match(r"^(.*?),\s*(.*?)\s*\(([^()]*)\)$", text)
    if match:
        role = match.group(1).strip()
        org = match.group(2).strip()
        dates = match.group(3).strip()
        return role, org, dates
    return text, "", ""


def render_experience(text: str) -> str:
    entries = parse_markdown_outline(text)
    output: list[str] = []
    for entry in entries:
        role, org, dates = split_role_line(entry["text"])
        subtitle = org
        body = render_simple_bullets(entry["children"])
        output.append(
            rf"\resumeSubheading{{{latex_escape(role)}}}{{{latex_escape(dates)}}}{{{latex_escape(subtitle)}}}{{{body}}}"
        )
    return "\n".join(output).strip()


def render_education(text: str) -> str:
    entries = parse_markdown_outline(text)
    output: list[str] = []
    for entry in entries:
        degree, org, dates = split_role_line(entry["text"])
        subtitle = org
        output.append(
            rf"\resumeSubheading{{{latex_escape(degree)}}}{{{latex_escape(dates)}}}{{{latex_escape(subtitle)}}}{{}}"
        )
    return "\n".join(output).strip()


def render_skills(text: str) -> str:
    entries = parse_markdown_outline(text)
    if not entries:
        return ""
    lines: list[str] = [r"\begin{itemize}"]
    for entry in entries:
        label, sep, value = entry["text"].partition(":")
        if sep:
            lines.append(
                rf"\item \textbf{{{latex_escape(label.strip())}}}: {latex_escape(normalize_whitespace(value.strip()))}"
            )
        else:
            lines.append(rf"\item {latex_escape(sentenceify(entry['text']))}")
    lines.append(r"\end{itemize}")
    return "\n".join(lines).strip()


def select_projects(
    selected: list[dict[str, Any]], max_projects: int, unique_families: bool
) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    seen_families: set[str] = set()
    for project in selected:
        family_id = project.get("family_id")
        if unique_families and family_id and family_id in seen_families:
            continue
        chosen.append(project)
        if family_id:
            seen_families.add(str(family_id))
        if len(chosen) >= max_projects:
            break
    return chosen


def load_resume_plan(plan_file: Path) -> dict[str, Any]:
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("resume-plan must be a JSON object")
    if plan.get("schema_version") != RESUME_PLAN_SCHEMA_VERSION:
        raise ValueError(
            f"resume-plan schema_version must be {RESUME_PLAN_SCHEMA_VERSION}"
        )
    projects = plan.get("projects")
    if not isinstance(projects, list) or not projects:
        raise ValueError("resume-plan projects must be a non-empty list")
    return plan


def materialize_plan_projects(
    plan: dict[str, Any], selected_projects: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    selected_index = {
        str(project.get("project_id") or "").strip(): project
        for project in selected_projects
        if str(project.get("project_id") or "").strip()
    }
    resolved_projects: list[dict[str, Any]] = []
    for item in plan["projects"]:
        if not isinstance(item, dict):
            raise ValueError("resume-plan projects entries must be objects")
        project_id = str(item.get("project_id") or "").strip()
        if not project_id:
            raise ValueError("resume-plan projects entries must include project_id")
        source_project = selected_index.get(project_id)
        if source_project is None:
            raise ValueError(f"Project not found in selected bundle: {project_id}")
        merged = dict(source_project)
        overrides = item.get("overrides") or {}
        if not isinstance(overrides, dict):
            raise ValueError(
                f"resume-plan overrides for {project_id} must be an object"
            )
        merged.update(overrides)
        resolved_projects.append(merged)
    return resolved_projects


def choose_project_summary(project: dict[str, Any]) -> str:
    for key in ["synthesis", "motivation_summary", "elevator_summary", "cosine_title"]:
        value = str(project.get(key, "") or "").strip()
        if value:
            return sentenceify(first_paragraph(value))
    return sentenceify(str(project["display_name"]))


def choose_project_bullets(project: dict[str, Any], max_bullets: int) -> list[str]:
    summary = choose_project_summary(project)
    unique_candidates: list[str] = []
    seen: set[str] = set()
    for key in ["suggested_bullet_points", "bullet_candidates"]:
        for item in project.get(key) or []:
            candidate = sentenceify(str(item))
            if not candidate:
                continue
            normalized = normalize_compare_text(candidate)
            if normalized in seen:
                continue
            seen.add(normalized)
            unique_candidates.append(candidate)

    if not unique_candidates:
        return []

    filtered = [
        candidate
        for candidate in unique_candidates
        if text_similarity(candidate, summary) < 0.50
    ]
    chosen = sorted(
        filtered or unique_candidates,
        key=lambda candidate: (len(candidate), unique_candidates.index(candidate)),
    )
    return chosen[:max_bullets]


def project_subtitle(project: dict[str, Any]) -> str:
    family_name = str(project.get("family_name") or "").strip()
    title = visible_project_title(project)
    if family_name and family_name != title:
        return family_name
    return ""


def render_projects(
    selected: list[dict[str, Any]],
    max_projects: int,
    max_bullets: int,
    unique_families: bool,
    authoritative_order: bool = False,
) -> str:
    projects = (
        selected
        if authoritative_order
        else select_projects(
            selected, max_projects=max_projects, unique_families=unique_families
        )
    )
    entries: list[str] = []
    for project in projects:
        title = latex_escape(visible_project_title(project))
        subtitle = latex_escape(project_subtitle(project))
        bullets = choose_project_bullets(project, max_bullets=max_bullets)
        summary = latex_escape(choose_project_summary(project))
        if bullets:
            body = [r"\begin{itemize}"]
            for bullet in bullets:
                body.append(rf"\item {latex_escape(bullet)}")
            body.append(r"\end{itemize}")
            body_text = "\n".join(body)
        else:
            body_text = ""
        entries.append(
            rf"\resumeProject{{{title}}}{{{subtitle}}}{{{summary}}}{{{body_text}}}"
        )
    return "\n".join(entries).strip()


def replace_section(template: str, section_name: str, content: str) -> str:
    pattern = re.compile(
        rf"\{{\{{#{section_name}\}}\}}(.*?)\{{\{{/{section_name}\}}\}}",
        re.DOTALL,
    )

    def replacement(match: re.Match[str]) -> str:
        if not content.strip():
            return ""
        block = match.group(1)
        return block.replace(f"{{{{{section_name}}}}}", content).strip() + "\n"

    return pattern.sub(replacement, template)


def render_template(template_text: str, mapping: dict[str, str]) -> str:
    rendered = template_text
    for key in ["summary", "experience", "projects", "education", "skills"]:
        rendered = replace_section(rendered, key, mapping.get(key, ""))

    rendered = rendered.replace("{{{name}}}", "{" + mapping.get("name", "") + "}")
    rendered = rendered.replace(
        "{{{headline}}}", "{" + mapping.get("headline", "") + "}"
    )
    rendered = rendered.replace(
        "{{{social_row}}}", "{" + mapping.get("social_row", "") + "}"
    )
    rendered = rendered.replace(
        "{{{contact_row}}}", "{" + mapping.get("contact_row", "") + "}"
    )
    return rendered


def render_social_row(profile_frontmatter: dict[str, Any]) -> str:
    github = str(profile_frontmatter.get("github", "") or "").strip()
    linkedin = str(profile_frontmatter.get("linkedin", "") or "").strip()
    parts: list[str] = []
    if github:
        github_url = github if github.startswith("http") else f"https://{github}"
        github_label_match = re.search(
            r"github\.com/([^/?#]+)", github_url, flags=re.IGNORECASE
        )
        github_label = github_label_match.group(1) if github_label_match else github
        parts.append(
            rf"\href{{{latex_escape(github_url)}}}{{\faGithub\enspace {latex_escape(github_label)}}}"
        )
    if linkedin:
        linkedin_url = (
            linkedin if linkedin.startswith("http") else f"https://{linkedin}"
        )
        linkedin_label_match = re.search(
            r"linkedin\.com/(?:in|company)/([^/?#]+)",
            linkedin_url,
            flags=re.IGNORECASE,
        )
        linkedin_label = (
            linkedin_label_match.group(1)
            if linkedin_label_match
            else re.sub(r"^https?://(www\.)?", "", linkedin_url).rstrip("/")
        )
        parts.append(
            rf"\href{{{latex_escape(linkedin_url)}}}{{\faLinkedin\enspace {latex_escape(linkedin_label)}}}"
        )
    return r" \enspace\textbar\enspace ".join(parts)


def render_contact_row(profile_frontmatter: dict[str, Any]) -> str:
    parts = [
        str(profile_frontmatter.get("email", "") or "").strip(),
        str(profile_frontmatter.get("phone", "") or "").strip(),
    ]
    escaped = [latex_escape(part) for part in parts if part]
    return r" \enspace\textbar\enspace ".join(escaped)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render a deterministic LaTeX resume draft from a selected project bundle."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--selected-file", type=Path)
    input_group.add_argument("--plan-file", type=Path)
    parser.add_argument("--profile-file", type=Path, required=True)
    parser.add_argument("--template-file", type=Path, required=True)
    parser.add_argument("--output-tex", type=Path, required=True)
    parser.add_argument("--max-projects", type=int, default=4)
    parser.add_argument("--max-bullets-per-project", type=int, default=3)
    parser.add_argument("--allow-family-duplicates", action="store_true")
    args = parser.parse_args()

    authoritative_order = False
    if args.plan_file:
        plan = load_resume_plan(args.plan_file)
        source_selected_path_raw = str(plan.get("source_selected_path") or "").strip()
        if not source_selected_path_raw:
            raise SystemExit("resume-plan is missing source_selected_path")
        source_selected_path = Path(source_selected_path_raw)
        selected = materialize_plan_projects(
            plan,
            load_selected_projects(source_selected_path),
        )
        authoritative_order = True
    else:
        selected = load_selected_projects(args.selected_file)
    profile_frontmatter, profile_sections, _ = load_profile(args.profile_file)
    template_text = args.template_file.read_text(encoding="utf-8")

    social_row = render_social_row(profile_frontmatter)
    contact_row = render_contact_row(profile_frontmatter)

    mapping = {
        "name": latex_escape(str(profile_frontmatter.get("name", ""))),
        "headline": latex_escape(str(profile_frontmatter.get("headline", ""))),
        "social_row": social_row,
        "contact_row": contact_row,
        "summary": render_paragraph(profile_sections.get("summary", "")),
        "experience": render_experience(profile_sections.get("experience", "")),
        "education": render_education(profile_sections.get("education", "")),
        "skills": render_skills(profile_sections.get("skills", "")),
        "projects": render_projects(
            selected,
            max_projects=args.max_projects,
            max_bullets=args.max_bullets_per_project,
            unique_families=not args.allow_family_duplicates,
            authoritative_order=authoritative_order,
        ),
    }

    rendered = render_template(template_text, mapping)
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text(rendered, encoding="utf-8")
    print(args.output_tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
