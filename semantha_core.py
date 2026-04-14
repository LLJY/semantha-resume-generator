#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import mimetypes
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from build_resume_prompt import (
    DEFAULT_PLAN_PROJECT_COUNT,
    build_project_bundle,
    build_resume_plan,
    load_profile,
    load_selected_projects,
    render_prompt,
    select_ranked_results_for_bundle_report,
)
from draft_resume_tex import (
    analyze_render_budget,
    choose_project_bullets,
    choose_project_summary,
    latex_escape,
    load_resume_plan,
    materialize_plan_projects,
    render_contact_row,
    render_education,
    render_experience,
    render_paragraph,
    render_projects,
    render_skills,
    render_social_row,
    render_template,
    text_similarity,
)
from rank_projects import (
    SECTION_RE,
    ProjectRecord,
    attach_context,
    attach_family,
    build_match_report,
    extract_ranking_metadata,
    load_family_map,
    load_record,
    load_target_text,
    normalize_label,
    parse_sections,
    rank_projects,
    should_include_record,
    split_frontmatter,
)
from resume_budget import estimate_resume_budget
from semantic_features import SemanticConfig
from sync_context import extract_project_id as extract_context_project_id
from sync_projects import extract_project_id as extract_project_record_id


DEFAULT_PROTOCOL_VERSION = "2025-06-18"
DEFAULT_MAX_PROJECTS = 4
DEFAULT_MAX_BULLETS_PER_PROJECT = 3
PROJECT_SECTION_TITLES = {
    "resume_keywords": "Resume Keywords",
    "elevator_summary": "Elevator Summary",
    "why_it_matters": "Why It Matters",
    "what_was_built": "What Was Built",
    "technical_highlights": "Technical Highlights",
    "architecture_notes": "Architecture Notes",
    "evidence": "Evidence",
    "resume_bullet_candidates": "Resume Bullet Candidates",
    "caveats": "Caveats",
}
CONTEXT_SECTION_TITLES = {
    "why_this_existed": "Why This Existed",
    "personal_context": "Personal Context",
    "resume_intent": "Resume Intent",
    "synthesis": "Synthesis",
    "suggested_bullet_points": "Suggested Bullet Points",
    "things_not_obvious_from_the_code": "Things Not Obvious From The Code",
}
PROFILE_SECTION_ORDER = ["summary", "experience", "education", "skills"]
PROFILE_SECTION_TITLES = {
    "summary": "Summary",
    "experience": "Experience",
    "education": "Education",
    "skills": "Skills",
}


class SemanthaError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkspacePaths:
    generator_dir: Path
    workspace_root: Path
    projects_dir: Path
    context_dir: Path
    families_dir: Path
    output_dir: Path
    template_file: Path
    profile_file: Path
    portfolio_resume_dir: Path
    portfolio_context_dir: Path
    portfolio_repos_dir: Path

    @classmethod
    def from_generator_dir(cls, generator_dir: Path) -> "WorkspacePaths":
        generator_dir = generator_dir.resolve()
        workspace_root = generator_dir.parent
        return cls(
            generator_dir=generator_dir,
            workspace_root=workspace_root,
            projects_dir=generator_dir / "data/projects",
            context_dir=generator_dir / "data/context",
            families_dir=generator_dir / "data/families",
            output_dir=generator_dir / "output",
            template_file=generator_dir / "templates/modern-onepage.tex",
            profile_file=workspace_root / "portfolio/resume/profile.md",
            portfolio_resume_dir=workspace_root / "portfolio/resume",
            portfolio_context_dir=workspace_root / "portfolio/context",
            portfolio_repos_dir=workspace_root / "portfolio/repos",
        )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _ensure_within(path: Path, roots: list[Path]) -> Path:
    resolved = path.resolve()
    for root in roots:
        if _is_relative_to(resolved, root.resolve()):
            return resolved
    raise SemanthaError(f"Path is outside allowed roots: {path}")


def _resolve_existing_path(
    raw: str, roots: list[Path], search_roots: list[Path] | None = None
) -> Path:
    candidate = Path(raw)
    search_paths = [candidate] if candidate.is_absolute() else [candidate]
    if not candidate.is_absolute():
        search_paths.extend(root / candidate for root in (search_roots or roots))
    for search_path in search_paths:
        if search_path.exists():
            return _ensure_within(search_path, roots)
    raise SemanthaError(f"Path not found: {raw}")


def _resolve_output_path(
    raw: str,
    default_root: Path,
    allowed_roots: list[Path],
    search_roots: list[Path] | None = None,
) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        return _ensure_within(candidate, allowed_roots)
    for candidate_path in (
        [candidate]
        + [root / candidate for root in (search_roots or [])]
        + [default_root / candidate]
    ):
        try:
            return _ensure_within(candidate_path, allowed_roots)
        except SemanthaError:
            continue
    raise SemanthaError(f"Path is outside allowed roots: {raw}")


def _validate_project_id(project_id: str) -> str:
    project_id = project_id.strip()
    if not project_id:
        raise SemanthaError("project_id is required")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", project_id):
        raise SemanthaError(
            "project_id must only contain letters, numbers, dot, underscore, and hyphen"
        )
    return project_id


def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "":
        return '""'
    if re.fullmatch(r"[A-Za-z0-9_./:+#@'-]+(?: [A-Za-z0-9_./:+#@'-]+)*", text):
        return text
    return json.dumps(text, ensure_ascii=False)


def _render_frontmatter(frontmatter: dict[str, Any], preferred_order: list[str]) -> str:
    keys = []
    seen: set[str] = set()
    for key in preferred_order:
        if key in frontmatter:
            keys.append(key)
            seen.add(key)
    for key in sorted(frontmatter):
        if key not in seen:
            keys.append(key)
    lines = ["---"]
    for key in keys:
        value = frontmatter[key]
        if isinstance(value, list):
            if not value:
                lines.append(f"{key}: []")
            else:
                lines.append(f"{key}:")
                for item in value:
                    lines.append(f"  - {_format_scalar(item)}")
        else:
            lines.append(f"{key}: {_format_scalar(value)}")
    lines.append("---")
    return "\n".join(lines)


def _normalize_section_key(key: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    return normalized


def _render_section_block(title: str, content: Any) -> str:
    if isinstance(content, list):
        body = "\n".join(
            f"- {str(item).strip()}" for item in content if str(item).strip()
        )
    else:
        body = str(content or "").strip()
    return f"## {title}\n\n{body}".rstrip()


def _has_meaningful_content(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, list):
        return any(str(item).strip() for item in value)
    return bool(str(value).strip())


def _render_markdown_document(
    *,
    frontmatter: dict[str, Any],
    sections: dict[str, Any],
    title_map: dict[str, str],
    preferred_frontmatter_order: list[str],
) -> str:
    fm = _render_frontmatter(frontmatter, preferred_frontmatter_order)
    rendered_sections: list[str] = []
    used_keys: set[str] = set()
    for key, title in title_map.items():
        if key in sections and _has_meaningful_content(sections[key]):
            rendered_sections.append(_render_section_block(title, sections[key]))
            used_keys.add(key)
    for key in sorted(sections):
        if key in used_keys:
            continue
        value = sections[key]
        if not _has_meaningful_content(value):
            continue
        title = key.replace("_", " ").title()
        rendered_sections.append(_render_section_block(title, value))
    body = "\n\n".join(rendered_sections).strip()
    return (fm + "\n\n" + body + "\n").strip() + "\n"


def _parse_profile_sections_with_order(
    body: str,
) -> tuple[list[str], dict[str, str], dict[str, list[str]]]:
    order: list[str] = []
    titles: dict[str, str] = {}
    sections: dict[str, list[str]] = {}
    current = ""
    for line in body.splitlines():
        match = SECTION_RE.match(line)
        if match:
            title = match.group(1).strip()
            key = _normalize_section_key(title)
            current = key
            if key not in sections:
                sections[key] = []
                order.append(key)
            titles[key] = title
        elif current:
            sections.setdefault(current, []).append(line)
    return order, titles, sections


def _render_profile(
    frontmatter: dict[str, Any], body_sections: list[tuple[str, str]]
) -> str:
    fm = _render_frontmatter(
        frontmatter,
        ["name", "headline", "location", "email", "github", "linkedin", "phone"],
    )
    blocks = [fm]
    for title, content in body_sections:
        blocks.append(f"## {title}\n\n{content.strip()}".rstrip())
    return "\n\n".join(blocks).strip() + "\n"


def _extract_metrics_signal(text: str) -> bool:
    return bool(
        re.search(
            r"\d|percent|ms|seconds|weeks|months|users|routes|stops|deployed",
            text,
            re.IGNORECASE,
        )
    )


class SemanthaWorkspace:
    def __init__(self, generator_dir: Path):
        self.paths = WorkspacePaths.from_generator_dir(generator_dir)

    def _load_records(
        self,
        projects_dir: Path | None = None,
        context_dir: Path | None = None,
        family_dir: Path | None = None,
    ) -> list[ProjectRecord]:
        projects_dir = projects_dir or self.paths.projects_dir
        context_dir = context_dir or self.paths.context_dir
        family_dir = family_dir or self.paths.families_dir
        project_paths = sorted(projects_dir.glob("*.md"))
        family_map = load_family_map(family_dir)
        records: list[ProjectRecord] = []
        for path in project_paths:
            record = load_record(path)
            if not should_include_record(record):
                continue
            records.append(
                attach_family(attach_context(record, context_dir), family_map)
            )
        return records

    def _project_path(self, project_id: str) -> Path:
        project_id = _validate_project_id(project_id)
        candidate = self.paths.projects_dir / f"{project_id}.md"
        if candidate.exists():
            return candidate
        for path in self.paths.projects_dir.glob("*.md"):
            record = load_record(path)
            if record.project_id == project_id:
                return path
        raise SemanthaError(f"Unknown project_id: {project_id}")

    def _context_path(self, project_id: str) -> Path:
        project_id = _validate_project_id(project_id)
        candidate = self.paths.context_dir / f"{project_id}.md"
        if candidate.exists():
            return candidate
        candidate = self.paths.portfolio_context_dir / f"{project_id}.md"
        if candidate.exists():
            return candidate
        raise SemanthaError(f"Unknown context overlay: {project_id}")

    def _family_path(self, family_id: str) -> Path:
        family_id = _validate_project_id(family_id)
        candidate = self.paths.families_dir / f"{family_id}.md"
        if candidate.exists():
            return candidate
        raise SemanthaError(f"Unknown family_id: {family_id}")

    def _selected_bundle_path(
        self, *, label: str | None, selected_file: str | None
    ) -> Path:
        if selected_file:
            resolved = _resolve_existing_path(
                selected_file,
                [self.paths.output_dir],
                [
                    self.paths.generator_dir,
                    self.paths.output_dir,
                    self.paths.workspace_root,
                ],
            )
            if not resolved.name.endswith("-selected.json"):
                raise SemanthaError(
                    "selected_file must point to an output/*-selected.json bundle"
                )
            return resolved
        if not label:
            raise SemanthaError("Provide label or selected_file")
        candidate = self.paths.output_dir / f"{normalize_label(label)}-selected.json"
        if not candidate.exists():
            raise SemanthaError(f"Selected bundle not found: {candidate}")
        return candidate

    def _resume_plan_path(
        self, *, label: str | None, resume_plan_file: str | None
    ) -> Path:
        if resume_plan_file:
            resolved = _resolve_existing_path(
                resume_plan_file,
                [self.paths.output_dir],
                [
                    self.paths.generator_dir,
                    self.paths.output_dir,
                    self.paths.workspace_root,
                ],
            )
            if not resolved.name.endswith("-resume-plan.json"):
                raise SemanthaError(
                    "resume_plan_file must point to an output/*-resume-plan.json file"
                )
            return resolved
        if not label:
            raise SemanthaError("Provide label or resume_plan_file")
        candidate = self.paths.output_dir / f"{normalize_label(label)}-resume-plan.json"
        if not candidate.exists():
            raise SemanthaError(f"Resume plan not found: {candidate}")
        return candidate

    def _tex_path(self, *, label: str | None, tex_file: str | None) -> Path:
        if tex_file:
            resolved = _resolve_existing_path(
                tex_file,
                [self.paths.output_dir],
                [
                    self.paths.generator_dir,
                    self.paths.output_dir,
                    self.paths.workspace_root,
                ],
            )
            if resolved.suffix.lower() != ".tex":
                raise SemanthaError("tex_file must point to an output/*.tex file")
            return resolved
        if not label:
            raise SemanthaError("Provide label or tex_file")
        candidate = self.paths.output_dir / f"{normalize_label(label)}.tex"
        if not candidate.exists():
            raise SemanthaError(f"TeX file not found: {candidate}")
        return candidate

    def project_index(self) -> list[dict[str, Any]]:
        family_map = load_family_map(self.paths.families_dir)
        items: list[dict[str, Any]] = []
        for path in sorted(self.paths.projects_dir.glob("*.md")):
            record = attach_family(load_record(path), family_map)
            items.append(
                {
                    "project_id": record.project_id,
                    "display_name": record.title,
                    "source_display_name": record.source_title,
                    "repo": record.frontmatter.get("repo", ""),
                    "source_type": record.frontmatter.get("source_type", ""),
                    "family_id": record.family_id,
                    "family_name": record.family_name,
                    "path": str(path),
                }
            )
        return items

    def semantic_search_projects(
        self,
        *,
        query: str | None = None,
        target_text: str | None = None,
        target_file: str | None = None,
        role_family: str | None = None,
        top: int = 10,
    ) -> dict[str, Any]:
        target_path = (
            _resolve_existing_path(
                target_file,
                [
                    self.paths.portfolio_resume_dir,
                    self.paths.generator_dir,
                    self.paths.workspace_root,
                ],
                [
                    self.paths.generator_dir,
                    self.paths.workspace_root,
                    self.paths.portfolio_resume_dir,
                ],
            )
            if target_file
            else None
        )
        combined_target = target_text or load_target_text(
            query, target_path, role_family
        )
        if not combined_target:
            raise SemanthaError("Provide query, target_text, or target_file")
        records = self._load_records()
        if not records:
            raise SemanthaError(
                f"No eligible markdown project files found in {self.paths.projects_dir}"
            )
        ranked = rank_projects(records, combined_target, role_family)[: max(1, top)]
        metadata = extract_ranking_metadata(ranked, combined_target)
        match_report = build_match_report(
            label="semantic-search",
            target_text=combined_target,
            role_family=role_family,
            semantic_config=SemanticConfig.from_env(),
            target_keywords=metadata["target_keywords"],
            expanded_keywords=metadata["expanded_keywords"],
            diagnostics=metadata["diagnostics"],
            results=ranked,
        )
        return {
            "target_text": combined_target,
            "role_family": role_family,
            "results": ranked,
            "match_report": match_report,
        }

    def inspect_project(self, project_id: str) -> dict[str, Any]:
        path = self._project_path(project_id)
        record = attach_family(
            attach_context(load_record(path), self.paths.context_dir),
            load_family_map(self.paths.families_dir),
        )
        context_path = self.paths.context_dir / f"{record.project_id}.md"
        return {
            "project_id": record.project_id,
            "path": str(path),
            "frontmatter": record.frontmatter,
            "sections": record.sections,
            "family": {
                "family_id": record.family_id,
                "family_name": record.family_name,
                "family_role": record.family_role,
                "family_keywords": record.family_keywords or [],
            },
            "context_path": str(context_path) if context_path.exists() else None,
            "context_frontmatter": record.context_frontmatter or {},
            "context_sections": record.context_sections or {},
        }

    def recommend_follow_up_questions(
        self,
        *,
        query: str | None = None,
        target_text: str | None = None,
        target_file: str | None = None,
        role_family: str | None = None,
        top: int = 5,
    ) -> dict[str, Any]:
        search = self.semantic_search_projects(
            query=query,
            target_text=target_text,
            target_file=target_file,
            role_family=role_family,
            top=max(3, top),
        )
        questions: list[dict[str, Any]] = []
        if not role_family:
            questions.append(
                {
                    "scope": "global",
                    "question": "Which role family should SEmantha bias toward for this resume variant?",
                    "why": "Role-family bias materially changes ranking and project framing.",
                }
            )
        if len(search["target_text"].split()) < 25:
            questions.append(
                {
                    "scope": "global",
                    "question": "Can you paste the full job description or the strongest requirement bullets?",
                    "why": "A short query weakens semantic matching and recommendation quality.",
                }
            )

        for item in search["results"]:
            inspected = self.inspect_project(item["project_id"])
            context_fm = inspected["context_frontmatter"]
            context_sections = inspected["context_sections"]
            sections = inspected["sections"]
            bullets = [
                bullet
                for bullet in sections.get("resume bullet candidates", "").splitlines()
                if bullet.strip().startswith("- ")
            ]
            if not context_fm.get("motivation_summary"):
                questions.append(
                    {
                        "scope": item["project_id"],
                        "question": f"Why did {item['display_name']} need to exist in the first place?",
                        "why": "Missing motivation makes ranking and tailored framing weaker.",
                    }
                )
            if not context_sections.get("synthesis"):
                questions.append(
                    {
                        "scope": item["project_id"],
                        "question": f"What is the shortest high-signal synthesis line for {item['display_name']}?",
                        "why": "The deterministic renderer and prompt bundle both prefer synthesis when it exists.",
                    }
                )
            if not bullets:
                questions.append(
                    {
                        "scope": item["project_id"],
                        "question": f"What are the 2-4 best ATS bullets for {item['display_name']}?",
                        "why": "The project record is missing bullet candidates.",
                    }
                )
            elif not _extract_metrics_signal(" ".join(bullets)):
                questions.append(
                    {
                        "scope": item["project_id"],
                        "question": f"What measurable result, performance change, scale, or deployment fact should be attached to {item['display_name']}?",
                        "why": "The current bullets are factual but light on outcomes or scale.",
                    }
                )
            if (
                str(inspected["frontmatter"].get("authorship_confidence", "")).lower()
                != "high"
            ):
                questions.append(
                    {
                        "scope": item["project_id"],
                        "question": f"How should authorship for {item['display_name']} be framed on the resume?",
                        "why": "The current record is not high-confidence solo ownership.",
                    }
                )
            if len(questions) >= 8:
                break

        return {
            "target_text": search["target_text"],
            "questions": questions[:8],
            "top_projects": search["results"][: max(3, min(top, 5))],
        }

    def build_resume_bundle(
        self,
        *,
        query: str | None = None,
        target_text: str | None = None,
        target_file: str | None = None,
        role_family: str | None = None,
        top: int = 6,
        allow_family_duplicates: bool = False,
        label: str = "resume",
    ) -> dict[str, Any]:
        target_path = (
            _resolve_existing_path(
                target_file,
                [
                    self.paths.portfolio_resume_dir,
                    self.paths.generator_dir,
                    self.paths.workspace_root,
                ],
                [
                    self.paths.generator_dir,
                    self.paths.workspace_root,
                    self.paths.portfolio_resume_dir,
                ],
            )
            if target_file
            else None
        )
        combined_target = target_text or load_target_text(
            query, target_path, role_family
        )
        if not combined_target:
            raise SemanthaError("Provide query, target_text, or target_file")

        records = self._load_records()
        if not records:
            raise SemanthaError(
                f"No eligible markdown project files found in {self.paths.projects_dir}"
            )
        ranked = rank_projects(records, combined_target, role_family)
        metadata = extract_ranking_metadata(ranked, combined_target)
        selected_projects = build_project_bundle(
            ranked,
            self.paths.projects_dir,
            self.paths.context_dir,
            top_limit=max(1, top),
            unique_families=not allow_family_duplicates,
        )
        report_results = select_ranked_results_for_bundle_report(
            ranked, selected_projects
        )
        profile_frontmatter, profile_sections, _ = load_profile(self.paths.profile_file)
        template_text = self.paths.template_file.read_text(encoding="utf-8")
        label = normalize_label(label or "resume")
        prompt_path = self.paths.output_dir / f"{label}-prompt.md"
        selected_path = self.paths.output_dir / f"{label}-selected.json"
        report_path = self.paths.output_dir / f"{label}-match-report.json"
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        prompt_text = render_prompt(
            label=label,
            role_family=role_family,
            target_text=combined_target,
            profile_frontmatter=profile_frontmatter,
            profile_sections=profile_sections,
            template_text=template_text,
            selected_projects=selected_projects,
        )
        prompt_path.write_text(prompt_text, encoding="utf-8")
        selected_path.write_text(
            json.dumps(selected_projects, indent=2), encoding="utf-8"
        )
        match_report = build_match_report(
            label=label,
            target_text=combined_target,
            role_family=role_family,
            semantic_config=SemanticConfig.from_env(),
            target_keywords=metadata["target_keywords"],
            expanded_keywords=metadata["expanded_keywords"],
            diagnostics=metadata["diagnostics"],
            results=report_results,
        )
        match_report["selected_bundle_budget"] = estimate_resume_budget(
            [
                {
                    "project_id": str(project.get("project_id") or ""),
                    "budget": project.get("budget_estimate") or {},
                }
                for project in selected_projects
            ]
        )
        report_path.write_text(json.dumps(match_report, indent=2), encoding="utf-8")
        return {
            "label": label,
            "target_text": combined_target,
            "prompt_path": str(prompt_path),
            "selected_path": str(selected_path),
            "match_report_path": str(report_path),
            "selected_projects": selected_projects,
        }

    def create_resume_plan(
        self,
        *,
        label: str | None = None,
        selected_file: str | None = None,
        chosen_project_ids: list[str] | None = None,
        project_overrides: dict[str, dict[str, Any]] | None = None,
        top_n: int = DEFAULT_PLAN_PROJECT_COUNT,
    ) -> dict[str, Any]:
        if chosen_project_ids is not None:
            if not isinstance(chosen_project_ids, list) or any(
                not isinstance(item, str) for item in chosen_project_ids
            ):
                raise SemanthaError("chosen_project_ids must be a list of strings")
        if project_overrides is not None:
            if not isinstance(project_overrides, dict) or any(
                not isinstance(key, str) or not isinstance(value, dict)
                for key, value in project_overrides.items()
            ):
                raise SemanthaError(
                    "project_overrides must be an object keyed by project_id"
                )

        selected_path = self._selected_bundle_path(
            label=label, selected_file=selected_file
        )
        resolved_label = label or selected_path.name.removesuffix("-selected.json")
        plan_path = self.paths.output_dir / (
            f"{normalize_label(resolved_label)}-resume-plan.json"
        )
        try:
            plan = build_resume_plan(
                selected_projects=load_selected_projects(selected_path),
                label=normalize_label(resolved_label),
                chosen_project_ids=chosen_project_ids,
                project_overrides=project_overrides,
                top_n=max(1, top_n),
                source_selected_path=str(selected_path),
            )
        except ValueError as exc:
            raise SemanthaError(str(exc)) from exc
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

        selected_index = {
            str(project.get("project_id") or ""): project
            for project in load_selected_projects(selected_path)
        }
        planned_projects = []
        for item in plan["projects"]:
            project = dict(selected_index[item["project_id"]])
            project.update(item["overrides"])
            planned_projects.append(project)
        return {
            "label": normalize_label(resolved_label),
            "selected_path": str(selected_path),
            "resume_plan_path": str(plan_path),
            "selection_mode": plan["selection_mode"],
            "project_count": plan["project_count"],
            "projects": planned_projects,
        }

    def recommend_bundle_edits(
        self, *, label: str | None = None, selected_file: str | None = None
    ) -> dict[str, Any]:
        path = self._selected_bundle_path(label=label, selected_file=selected_file)
        selected = json.loads(path.read_text(encoding="utf-8"))
        recommendations: list[dict[str, Any]] = []
        family_counter = Counter(
            str(item.get("family_id") or "")
            for item in selected
            if item.get("family_id")
        )
        for project in selected:
            summary = choose_project_summary(project)
            bullets = choose_project_bullets(project, max_bullets=5)
            if len(summary) > 220:
                recommendations.append(
                    {
                        "project_id": project["project_id"],
                        "severity": "medium",
                        "type": "summary-length",
                        "recommendation": "Compress the description line; it is likely to wrap and waste vertical space.",
                    }
                )
            if not bullets:
                recommendations.append(
                    {
                        "project_id": project["project_id"],
                        "severity": "high",
                        "type": "missing-bullets",
                        "recommendation": "Add bullet candidates or suggested bullet points for ATS coverage.",
                    }
                )
            for bullet in bullets:
                if len(bullet) > 135:
                    recommendations.append(
                        {
                            "project_id": project["project_id"],
                            "severity": "medium",
                            "type": "bullet-length",
                            "recommendation": "Shorten at least one bullet; it is likely to wrap to a second line.",
                            "example": bullet,
                        }
                    )
                if text_similarity(summary, bullet) >= 0.5:
                    recommendations.append(
                        {
                            "project_id": project["project_id"],
                            "severity": "medium",
                            "type": "summary-bullet-duplication",
                            "recommendation": "Swap at least one bullet for a more distinct achievement or systems detail.",
                            "example": bullet,
                        }
                    )
                    break
            family_id = str(project.get("family_id") or "")
            if family_id and family_counter[family_id] > 1:
                recommendations.append(
                    {
                        "project_id": project["project_id"],
                        "severity": "low",
                        "type": "family-duplication",
                        "recommendation": "This bundle includes multiple members of the same project family; verify that the duplication is intentional.",
                    }
                )
        return {
            "selected_path": str(path),
            "recommendations": recommendations,
        }

    def patch_selected_bundle(
        self,
        *,
        project_id: str,
        patch: dict[str, Any],
        label: str | None = None,
        selected_file: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(patch, dict) or not patch:
            raise SemanthaError("patch must be a non-empty object")
        path = self._selected_bundle_path(label=label, selected_file=selected_file)
        selected = json.loads(path.read_text(encoding="utf-8"))
        mutable_fields = {
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
        target = None
        for item in selected:
            if item.get("project_id") == project_id:
                target = item
                break
        if target is None:
            raise SemanthaError(f"Project not found in selected bundle: {project_id}")
        list_fields = {
            "suggested_bullet_points",
            "bullet_candidates",
            "caveats",
            "role_family_targets",
        }
        for key, value in patch.items():
            if key not in mutable_fields:
                raise SemanthaError(f"Unsupported selected bundle patch field: {key}")
            if key in list_fields:
                if not isinstance(value, list) or any(
                    not isinstance(item, str) for item in value
                ):
                    raise SemanthaError(f"{key} must be a list of strings")
            elif not isinstance(value, str):
                raise SemanthaError(f"{key} must be a string")
            target[key] = value
        path.write_text(json.dumps(selected, indent=2), encoding="utf-8")
        return {
            "selected_path": str(path),
            "project": target,
        }

    def render_resume_tex(
        self,
        *,
        label: str | None = None,
        selected_file: str | None = None,
        resume_plan_file: str | None = None,
        output_tex: str | None = None,
        max_projects: int = DEFAULT_MAX_PROJECTS,
        max_bullets_per_project: int = DEFAULT_MAX_BULLETS_PER_PROJECT,
        allow_family_duplicates: bool = False,
    ) -> dict[str, Any]:
        auto_plan_path = None
        if not resume_plan_file and label and not selected_file:
            candidate = (
                self.paths.output_dir / f"{normalize_label(label)}-resume-plan.json"
            )
            if candidate.exists():
                auto_plan_path = candidate

        if resume_plan_file or auto_plan_path:
            plan_path = self._resume_plan_path(
                label=label, resume_plan_file=resume_plan_file or str(auto_plan_path)
            )
            try:
                plan = load_resume_plan(plan_path)
            except ValueError as exc:
                raise SemanthaError(str(exc)) from exc
            source_selected_path_raw = str(
                plan.get("source_selected_path") or ""
            ).strip()
            if not source_selected_path_raw:
                raise SemanthaError("resume-plan is missing source_selected_path")
            selected_path = _resolve_existing_path(
                source_selected_path_raw,
                [self.paths.output_dir],
                [
                    self.paths.generator_dir,
                    self.paths.output_dir,
                    self.paths.workspace_root,
                ],
            )
            try:
                selected = materialize_plan_projects(
                    plan,
                    load_selected_projects(selected_path),
                )
            except ValueError as exc:
                raise SemanthaError(str(exc)) from exc
            label = label or plan_path.name.removesuffix("-resume-plan.json")
            authoritative_order = True
        else:
            selected_path = self._selected_bundle_path(
                label=label, selected_file=selected_file
            )
            try:
                selected = load_selected_projects(selected_path)
            except ValueError as exc:
                raise SemanthaError(str(exc)) from exc
            label = label or selected_path.name.removesuffix("-selected.json")
            authoritative_order = False
        output_path = (
            _resolve_output_path(
                output_tex,
                self.paths.output_dir,
                [self.paths.output_dir],
                [self.paths.generator_dir, self.paths.workspace_root],
            )
            if output_tex
            else self.paths.output_dir / f"{normalize_label(label)}.tex"
        )
        profile_frontmatter, profile_sections, _ = load_profile(self.paths.profile_file)
        template_text = self.paths.template_file.read_text(encoding="utf-8")
        mapping = {
            "name": latex_escape(str(profile_frontmatter.get("name", ""))),
            "headline": latex_escape(str(profile_frontmatter.get("headline", ""))),
            "social_row": render_social_row(profile_frontmatter),
            "contact_row": render_contact_row(profile_frontmatter),
            "summary": render_paragraph(profile_sections.get("summary", "")),
            "experience": render_experience(profile_sections.get("experience", "")),
            "education": render_education(profile_sections.get("education", "")),
            "skills": render_skills(profile_sections.get("skills", "")),
            "projects": render_projects(
                selected,
                max_projects=max_projects,
                max_bullets=max_bullets_per_project,
                unique_families=not allow_family_duplicates,
                authoritative_order=authoritative_order,
            ),
        }
        rendered = render_template(template_text, mapping)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        budget = analyze_render_budget(
            selected,
            max_projects=max_projects,
            max_bullets=max_bullets_per_project,
            unique_families=not allow_family_duplicates,
            authoritative_order=authoritative_order,
        )
        return {
            "label": normalize_label(label),
            "selected_path": str(selected_path),
            "resume_plan_path": str(plan_path)
            if (resume_plan_file or auto_plan_path)
            else None,
            "tex_path": str(output_path),
            "budget": budget,
        }

    def compile_resume_pdf(
        self, *, label: str | None = None, tex_file: str | None = None
    ) -> dict[str, Any]:
        pdflatex = shutil.which("pdflatex")
        if not pdflatex:
            raise SemanthaError("pdflatex is not installed or not on PATH")
        tex_path = self._tex_path(label=label, tex_file=tex_file)
        output_dir = tex_path.parent
        command = [
            pdflatex,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-output-directory",
            str(output_dir),
            str(tex_path),
        ]
        result = subprocess.run(
            command,
            cwd=self.paths.generator_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        log_path = output_dir / f"{tex_path.stem}.log"
        log_text = (
            log_path.read_text(encoding="utf-8", errors="ignore")
            if log_path.exists()
            else ""
        )
        combined_output = result.stdout + "\n" + log_text
        page_match = re.search(
            r"Output written on .*?\((\d+) page(?:s)?(?:,|\))",
            combined_output,
            flags=re.DOTALL,
        )
        warnings = [
            line
            for line in log_text.splitlines()
            if line.startswith("!")
            or "Overfull \\hbox" in line
            or "Underfull \\hbox" in line
        ]
        pdf_path = output_dir / f"{tex_path.stem}.pdf"
        response = {
            "tex_path": str(tex_path),
            "pdf_path": str(pdf_path),
            "log_path": str(log_path),
            "return_code": result.returncode,
            "pages": int(page_match.group(1)) if page_match else None,
            "warnings": warnings[:20],
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
        if result.returncode != 0:
            raise SemanthaError(json.dumps(response, ensure_ascii=False))
        return response

    def sync_project_records(
        self, *, source_root: str | None = None, output_dir: str | None = None
    ) -> dict[str, Any]:
        source = (
            _resolve_existing_path(
                source_root,
                [
                    self.paths.portfolio_repos_dir,
                    self.paths.workspace_root,
                    self.paths.generator_dir,
                ],
                [self.paths.workspace_root, self.paths.generator_dir],
            )
            if source_root
            else self.paths.portfolio_repos_dir
        )
        destination = (
            _resolve_output_path(
                output_dir,
                self.paths.projects_dir,
                [self.paths.projects_dir],
                [self.paths.generator_dir, self.paths.workspace_root],
            )
            if output_dir
            else self.paths.projects_dir
        )
        if destination != self.paths.projects_dir:
            raise SemanthaError("output_dir must be exactly data/projects")
        destination.mkdir(parents=True, exist_ok=True)
        synced: list[dict[str, str]] = []
        for path in sorted(source.glob("**/resume-project.md")):
            text = path.read_text(encoding="utf-8")
            fallback = path.parent.name.lower().replace("__", "-")
            project_id = extract_project_record_id(text, fallback)
            out_path = destination / f"{project_id}.md"
            out_path.write_text(text, encoding="utf-8")
            synced.append({"source": str(path), "destination": str(out_path)})
        return {
            "source_root": str(source),
            "output_dir": str(destination),
            "synced_count": len(synced),
            "synced": synced,
        }

    def sync_context_overlays(
        self, *, source_root: str | None = None, output_dir: str | None = None
    ) -> dict[str, Any]:
        source = (
            _resolve_existing_path(
                source_root,
                [
                    self.paths.portfolio_context_dir,
                    self.paths.workspace_root,
                    self.paths.generator_dir,
                ],
                [self.paths.workspace_root, self.paths.generator_dir],
            )
            if source_root
            else self.paths.portfolio_context_dir
        )
        destination = (
            _resolve_output_path(
                output_dir,
                self.paths.context_dir,
                [self.paths.context_dir],
                [self.paths.generator_dir, self.paths.workspace_root],
            )
            if output_dir
            else self.paths.context_dir
        )
        if destination != self.paths.context_dir:
            raise SemanthaError("output_dir must be exactly data/context")
        destination.mkdir(parents=True, exist_ok=True)
        synced: list[dict[str, str]] = []
        for path in sorted(source.glob("*.md")):
            if path.name in {"README.md", "project-context-template.md"}:
                continue
            text = path.read_text(encoding="utf-8")
            fallback = path.stem.lower()
            project_id = extract_context_project_id(text, fallback)
            out_path = destination / f"{project_id}.md"
            out_path.write_text(text, encoding="utf-8")
            synced.append({"source": str(path), "destination": str(out_path)})
        return {
            "source_root": str(source),
            "output_dir": str(destination),
            "synced_count": len(synced),
            "synced": synced,
        }

    def upsert_project_record(
        self,
        *,
        project_id: str,
        frontmatter: dict[str, Any],
        sections: dict[str, Any],
        write: bool = True,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        project_id = _validate_project_id(project_id)
        path = self.paths.projects_dir / f"{project_id}.md"
        if path.exists() and not overwrite and write:
            raise SemanthaError(f"Project record already exists: {path}")
        frontmatter = dict(frontmatter or {})
        frontmatter["project_id"] = project_id
        markdown = _render_markdown_document(
            frontmatter=frontmatter,
            sections={
                _normalize_section_key(k): v for k, v in (sections or {}).items()
            },
            title_map=PROJECT_SECTION_TITLES,
            preferred_frontmatter_order=[
                "project_id",
                "canonical_name",
                "display_name",
                "resume_display_name",
                "repo",
                "url",
                "visibility",
                "source_type",
                "primary_class",
                "secondary_tags",
                "role_family_targets",
                "cosine_title",
                "cosine_summary",
                "resume_keywords",
                "technology_keywords",
                "domain_keywords",
                "impact_keywords",
                "collaboration_keywords",
                "evidence_keywords",
                "status_keywords",
                "technical_impressiveness",
                "authorship_confidence",
                "inspection_level",
            ],
        )
        if write:
            path.write_text(markdown, encoding="utf-8")
        return {
            "project_id": project_id,
            "path": str(path),
            "written": write,
            "markdown": markdown,
        }

    def upsert_context_overlay(
        self,
        *,
        project_id: str,
        frontmatter: dict[str, Any],
        sections: dict[str, Any],
        write: bool = True,
        overwrite: bool = False,
        write_to_portfolio: bool = True,
    ) -> dict[str, Any]:
        project_id = _validate_project_id(project_id)
        root = (
            self.paths.portfolio_context_dir
            if write_to_portfolio
            else self.paths.context_dir
        )
        path = root / f"{project_id}.md"
        synced_path = self.paths.context_dir / f"{project_id}.md"
        if path.exists() and not overwrite and write:
            raise SemanthaError(f"Context overlay already exists: {path}")
        frontmatter = dict(frontmatter or {})
        frontmatter["project_id"] = project_id
        markdown = _render_markdown_document(
            frontmatter=frontmatter,
            sections={
                _normalize_section_key(k): v for k, v in (sections or {}).items()
            },
            title_map=CONTEXT_SECTION_TITLES,
            preferred_frontmatter_order=[
                "project_id",
                "motivation_summary",
                "problem_trigger",
                "personal_connection",
                "why_now",
                "constraints_or_stakes",
                "preferred_role_family_targets",
                "context_keywords",
            ],
        )
        if write:
            path.write_text(markdown, encoding="utf-8")
            if synced_path != path:
                synced_path.write_text(markdown, encoding="utf-8")
        return {
            "project_id": project_id,
            "path": str(path),
            "synced_path": str(synced_path),
            "written": write,
            "markdown": markdown,
        }

    def update_profile_section(
        self,
        *,
        section_name: str,
        content: str,
        mode: str = "replace",
        title: str | None = None,
    ) -> dict[str, Any]:
        text = self.paths.profile_file.read_text(encoding="utf-8")
        frontmatter, body = split_frontmatter(text)
        order, titles, sections = _parse_profile_sections_with_order(body)
        key = _normalize_section_key(section_name)
        if key not in sections:
            sections[key] = []
            order.append(key)
        titles[key] = (
            title
            or titles.get(key)
            or PROFILE_SECTION_TITLES.get(key)
            or section_name.strip()
        )
        existing = "\n".join(sections.get(key, [])).strip()
        content = content.strip()
        if mode == "replace":
            updated = content
        elif mode == "append_line":
            updated = "\n".join(part for part in [existing, content] if part).strip()
        elif mode == "append_bullet":
            bullet = content if content.startswith("- ") else f"- {content}"
            updated = "\n".join(part for part in [existing, bullet] if part).strip()
        elif mode == "append_paragraph":
            updated = "\n\n".join(part for part in [existing, content] if part).strip()
        else:
            raise SemanthaError(
                "mode must be replace, append_line, append_bullet, or append_paragraph"
            )
        sections[key] = updated.splitlines()
        ordered_sections = [
            (titles[name], "\n".join(sections[name]).strip())
            for name in order
            if "\n".join(sections[name]).strip()
        ]
        rendered = _render_profile(frontmatter, ordered_sections)
        self.paths.profile_file.write_text(rendered, encoding="utf-8")
        return {
            "profile_path": str(self.paths.profile_file),
            "section_name": titles[key],
            "mode": mode,
            "content": "\n".join(sections[key]).strip(),
        }

    def list_resources(self) -> list[dict[str, Any]]:
        resources = [
            {
                "uri": "semantha://profile",
                "name": "Profile",
                "title": "Resume Profile",
                "description": "The canonical profile markdown used for resume generation.",
                "mimeType": "text/markdown",
            },
            {
                "uri": "semantha://projects/index",
                "name": "Projects Index",
                "title": "Project Index",
                "description": "List of known project records in the corpus.",
                "mimeType": "application/json",
            },
            {
                "uri": "semantha://families/index",
                "name": "Families Index",
                "title": "Project Families Index",
                "description": "List of project families used for ranking diversification.",
                "mimeType": "application/json",
            },
            {
                "uri": "semantha://outputs/index",
                "name": "Outputs Index",
                "title": "Generated Outputs Index",
                "description": "List of generated selected bundles, match reports, resume plans, prompts, TeX files, and PDFs.",
                "mimeType": "application/json",
            },
            {
                "uri": "semantha://targets/index",
                "name": "Targets Index",
                "title": "Resume Targets Index",
                "description": "Target role markdown files from portfolio/resume.",
                "mimeType": "application/json",
            },
        ]
        return resources

    def list_resource_templates(self) -> list[dict[str, Any]]:
        return [
            {
                "uriTemplate": "semantha://projects/{project_id}",
                "name": "Project Record",
                "title": "Project Record",
                "description": "Read a specific project markdown record by project_id.",
                "mimeType": "text/markdown",
            },
            {
                "uriTemplate": "semantha://context/{project_id}",
                "name": "Context Overlay",
                "title": "Context Overlay",
                "description": "Read a specific project context overlay by project_id.",
                "mimeType": "text/markdown",
            },
            {
                "uriTemplate": "semantha://families/{family_id}",
                "name": "Family Overlay",
                "title": "Family Overlay",
                "description": "Read a project family overlay by family_id.",
                "mimeType": "text/markdown",
            },
            {
                "uriTemplate": "semantha://outputs/{label}/{kind}",
                "name": "Generated Output",
                "title": "Generated Output",
                "description": "Read a generated selected bundle, match report, resume plan, prompt, TeX, or PDF by label and kind.",
            },
        ]

    def read_resource(self, uri: str) -> dict[str, Any]:
        parsed = urlparse(uri)
        if parsed.scheme != "semantha":
            raise SemanthaError(f"Unsupported resource URI: {uri}")
        host = parsed.netloc
        path = [segment for segment in parsed.path.split("/") if segment]
        if host == "profile":
            return self._read_path_resource(self.paths.profile_file, uri=uri)
        if host == "projects" and path == ["index"]:
            return self._text_resource(
                uri, "application/json", json.dumps(self.project_index(), indent=2)
            )
        if host == "projects" and len(path) == 1:
            return self._read_path_resource(
                self._project_path(unquote(path[0])), uri=uri
            )
        if host == "context" and len(path) == 1:
            return self._read_path_resource(
                self._context_path(unquote(path[0])), uri=uri
            )
        if host == "families" and path == ["index"]:
            families = [
                {"family_id": p.stem, "path": str(p)}
                for p in sorted(self.paths.families_dir.glob("*.md"))
                if p.name != "README.md"
            ]
            return self._text_resource(
                uri, "application/json", json.dumps(families, indent=2)
            )
        if host == "families" and len(path) == 1:
            return self._read_path_resource(
                self._family_path(unquote(path[0])), uri=uri
            )
        if host == "targets" and path == ["index"]:
            targets = [
                {"name": p.name, "path": str(p)}
                for p in sorted(self.paths.portfolio_resume_dir.glob("*-target.md"))
            ]
            return self._text_resource(
                uri, "application/json", json.dumps(targets, indent=2)
            )
        if host == "targets" and len(path) == 1:
            target_path = _resolve_existing_path(
                unquote(path[0]),
                [self.paths.portfolio_resume_dir],
                [self.paths.workspace_root, self.paths.portfolio_resume_dir],
            )
            return self._read_path_resource(target_path, uri=uri)
        if host == "outputs" and path == ["index"]:
            outputs = sorted(self.paths.output_dir.glob("*"))
            items = []
            for output in outputs:
                if output.is_dir():
                    continue
                items.append(
                    {
                        "name": output.name,
                        "path": str(output),
                        "mimeType": mimetypes.guess_type(output.name)[0]
                        or "text/plain",
                    }
                )
            return self._text_resource(
                uri, "application/json", json.dumps(items, indent=2)
            )
        if host == "outputs" and len(path) == 2:
            label = normalize_label(unquote(path[0]))
            kind = unquote(path[1]).lower()
            suffix_map = {
                "selected": f"{label}-selected.json",
                "report": f"{label}-match-report.json",
                "plan": f"{label}-resume-plan.json",
                "prompt": f"{label}-prompt.md",
                "tex": f"{label}.tex",
                "pdf": f"{label}.pdf",
            }
            if kind not in suffix_map:
                raise SemanthaError(f"Unknown output kind: {kind}")
            return self._read_path_resource(
                self.paths.output_dir / suffix_map[kind], uri=uri
            )
        raise SemanthaError(f"Unknown resource URI: {uri}")

    def _text_resource(self, uri: str, mime_type: str, text: str) -> dict[str, Any]:
        return {
            "uri": uri,
            "mimeType": mime_type,
            "text": text,
        }

    def _read_path_resource(self, path: Path, uri: str | None = None) -> dict[str, Any]:
        if not path.exists():
            raise SemanthaError(f"Resource not found: {path}")
        uri = uri or path.as_uri()
        mime_type = mimetypes.guess_type(path.name)[0] or "text/plain"
        if path.suffix.lower() == ".pdf":
            return {
                "uri": uri,
                "mimeType": "application/pdf",
                "blob": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
        return {
            "uri": uri,
            "mimeType": mime_type,
            "text": path.read_text(encoding="utf-8"),
        }

    def list_prompts(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "tailor_resume",
                "title": "Tailor Resume End-to-End",
                "description": "Guide the LLM through retrieval, selection, packaging, rendering, and compilation.",
                "arguments": [
                    {
                        "name": "target_text",
                        "description": "Target job description or role brief",
                        "required": False,
                    },
                    {
                        "name": "role_family",
                        "description": "Optional role-family bias such as backend or linux-systems",
                        "required": False,
                    },
                    {
                        "name": "label",
                        "description": "Output label prefix",
                        "required": False,
                    },
                ],
            },
            {
                "name": "refine_resume",
                "title": "Refine Existing Resume Variant",
                "description": "Inspect a selected bundle, create an editorial plan, and regenerate the PDF.",
                "arguments": [
                    {
                        "name": "label",
                        "description": "Existing resume label prefix",
                        "required": True,
                    }
                ],
            },
        ]

    def get_prompt(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        arguments = arguments or {}
        if name == "tailor_resume":
            target_text = str(arguments.get("target_text", "") or "").strip()
            role_family = str(arguments.get("role_family", "") or "").strip()
            label = str(
                arguments.get("label", "resume-variant") or "resume-variant"
            ).strip()
            text = (
                "Use SEmantha to tailor the resume end-to-end.\n\n"
                "1. Call `semantic_search_projects` with the target text and optional role family.\n"
                "2. Call `build_resume_bundle` with label `"
                + label
                + "` to create the broad ranked `selected.json` retrieval bundle.\n"
                "3. Choose the final project subset in the LLM, optionally prepare per-project text overrides, then call `create_resume_plan`.\n"
                "4. Render from the resume plan with `render_resume_tex`, then compile with `compile_resume_pdf`.\n"
                "5. Report the output paths and any warnings.\n\n"
                f"Target text: {target_text or '[ask the user for a target description]'}\n"
                f"Role family: {role_family or '[optional]'}"
            )
            return self._prompt_response("Tailor resume workflow", text)
        if name == "refine_resume":
            label = str(arguments.get("label", "") or "").strip()
            if not label:
                raise SemanthaError("label is required for refine_resume prompt")
            text = (
                f"Refine the existing resume variant `{label}`.\n\n"
                "1. Read the selected bundle and current resume plan resources for the label.\n"
                "2. Ask the user which projects to keep, reorder, drop, or rewrite.\n"
                "3. Recreate the editorial intermediary with `create_resume_plan`.\n"
                "4. Re-render with `render_resume_tex` and recompile with `compile_resume_pdf`.\n"
                "6. Report page count, warnings, and output paths."
            )
            return self._prompt_response("Resume refinement workflow", text)
        raise SemanthaError(f"Unknown prompt: {name}")

    def _prompt_response(self, description: str, text: str) -> dict[str, Any]:
        return {
            "description": description,
            "messages": [
                {
                    "role": "user",
                    "content": {"type": "text", "text": text},
                }
            ],
        }
