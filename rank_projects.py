#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+.#/-]*")
SECTION_RE = re.compile(r"^##\s+(.*)$")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "both",
    "build",
    "built",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "with",
    "using",
    "use",
    "used",
    "user",
    "project",
    "projects",
    "role",
    "roles",
    "resume",
    "engineering",
    "engineer",
    "software",
    "systems",
    "system",
    "work",
    "working",
    "experience",
    "company",
    "job",
    "description",
    "responsibilities",
    "qualifications",
    "skills",
    "team",
    "teams",
    "code",
    "development",
    "design",
    "maintain",
    "maintaining",
    "develop",
    "developing",
    "services",
    "service",
}

ROLE_SIGNAL_KEYWORDS = {
    "backend": {
        "positive": {
            "backend",
            "api",
            "apis",
            "database",
            "databases",
            "postgresql",
            "clickhouse",
            "mariadb",
            "rabbitmq",
            "grpc",
            "rpc",
            "service",
            "services",
            "microservice",
            "microservices",
            "distributed",
            "queue",
            "queues",
            "auth",
            "oauth",
            "oidc",
            "identity",
            "server",
            "servers",
            "scalable",
            "scaling",
            "sql",
            "postgres",
            "valkey",
            "redis",
            "customer-facing",
            "platform",
        },
        "negative": {
            "firmware",
            "ble",
            "beacon",
            "esp32",
            "kernel",
            "ladspa",
            "arcore",
            "wear",
            "ws2812",
            "flutter",
            "angular",
            "react",
            "desktop",
            "compose",
            "ui",
            "frontend",
            "mobile",
        },
    },
    "linux-systems": {
        "positive": {
            "linux",
            "kernel",
            "systemd",
            "daemon",
            "pam",
            "sysfs",
            "filesystem",
            "pipewire",
            "udev",
            "packaging",
            "attestation",
        },
        "negative": {"flutter", "angular", "beacon", "arcore"},
    },
    "embedded": {
        "positive": {
            "firmware",
            "embedded",
            "esp32",
            "freertos",
            "ble",
            "wifi",
            "beacon",
            "microcontroller",
            "atecc608a",
            "pico",
            "esp-idf",
        },
        "negative": {"angular", "flutter", "postgresql", "clickhouse", "rabbitmq"},
    },
}


@dataclass
class ProjectRecord:
    path: Path
    frontmatter: dict[str, Any]
    sections: dict[str, str]
    context_frontmatter: dict[str, Any] | None = None
    context_sections: dict[str, str] | None = None
    family_id: str | None = None
    family_name: str | None = None
    family_role: str | None = None
    family_keywords: list[str] | None = None

    @property
    def project_id(self) -> str:
        return str(self.frontmatter.get("project_id") or self.path.stem)

    @property
    def title(self) -> str:
        return str(
            self.frontmatter.get("resume_display_name")
            or self.frontmatter.get("display_name")
            or self.frontmatter.get("canonical_name")
            or self.path.stem
        )

    @property
    def source_title(self) -> str:
        return str(
            self.frontmatter.get("display_name")
            or self.frontmatter.get("canonical_name")
            or self.path.stem
        )

    @property
    def technical_impressiveness(self) -> float:
        raw = self.frontmatter.get("technical_impressiveness", 1.0)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 1.0


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text
    lines = text.splitlines()
    end = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end = idx
            break
    if end is None:
        return {}, text
    frontmatter_text = "\n".join(lines[1:end])
    body = "\n".join(lines[end + 1 :])
    return parse_simple_yaml(frontmatter_text), body


def parse_simple_yaml(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        if not raw.strip() or raw.lstrip().startswith("#"):
            i += 1
            continue
        if ":" not in raw:
            i += 1
            continue
        key, value = raw.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            items: list[str] = []
            j = i + 1
            while j < len(lines):
                child = lines[j]
                if not child.startswith("  - "):
                    break
                items.append(child[4:].strip())
                j += 1
            data[key] = items
            i = j
            continue
        data[key] = parse_scalar(value)
        i += 1
    return data


def parse_scalar(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [part.strip().strip("'\"") for part in inner.split(",")]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)", value):
        return float(value)
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]
    return value


def parse_sections(body: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current = "_body"
    sections[current] = []
    for line in body.splitlines():
        match = SECTION_RE.match(line)
        if match:
            current = match.group(1).strip().lower()
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(line)
    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def load_record(path: Path) -> ProjectRecord:
    text = path.read_text(encoding="utf-8")
    frontmatter, body = split_frontmatter(text)
    return ProjectRecord(
        path=path, frontmatter=frontmatter, sections=parse_sections(body)
    )


def should_include_record(record: ProjectRecord) -> bool:
    source_type = normalize_label(str(record.frontmatter.get("source_type", "")))
    if source_type == "example":
        return False
    statuses = {
        normalize_label(str(x))
        for x in (record.frontmatter.get("status_keywords") or [])
    }
    return "example-only" not in statuses


def attach_context(record: ProjectRecord, context_root: Path | None) -> ProjectRecord:
    if context_root is None or not context_root.exists():
        return record

    candidates = [
        context_root / f"{record.project_id}.md",
        context_root / f"{record.path.stem}.md",
    ]
    for candidate in candidates:
        if candidate.exists():
            text = candidate.read_text(encoding="utf-8")
            frontmatter, body = split_frontmatter(text)
            record.context_frontmatter = frontmatter
            record.context_sections = parse_sections(body)
            break
    return record


def load_family_map(family_root: Path | None) -> dict[str, dict[str, Any]]:
    if family_root is None or not family_root.exists():
        return {}

    family_map: dict[str, dict[str, Any]] = {}
    for path in sorted(family_root.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        frontmatter, _ = split_frontmatter(text)
        family_id = str(frontmatter.get("family_id") or path.stem)
        family_name = str(frontmatter.get("family_name") or family_id)
        family_keywords = list(frontmatter.get("family_keywords") or [])
        members = list(frontmatter.get("members") or [])
        roles = frontmatter.get("member_roles") or {}
        if not isinstance(roles, dict):
            roles = {}
        for member in members:
            family_map[str(member)] = {
                "family_id": family_id,
                "family_name": family_name,
                "family_role": roles.get(member),
                "family_keywords": family_keywords,
            }
    return family_map


def attach_family(
    record: ProjectRecord, family_map: dict[str, dict[str, Any]]
) -> ProjectRecord:
    family = family_map.get(record.project_id)
    if not family:
        return record
    record.family_id = str(family.get("family_id") or "") or None
    record.family_name = str(family.get("family_name") or "") or None
    record.family_role = str(family.get("family_role") or "") or None
    family_keywords = family.get("family_keywords") or []
    record.family_keywords = [str(item) for item in family_keywords]
    return record


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS]


def role_keyword_adjustment(
    tokens: set[str], role_family: str | None
) -> tuple[float, list[str]]:
    if not role_family:
        return 0.0, []
    family = normalize_label(role_family)
    signals = ROLE_SIGNAL_KEYWORDS.get(family)
    if not signals:
        return 0.0, []

    notes: list[str] = []
    positive_hits = len(tokens & signals["positive"])
    negative_hits = len(tokens & signals["negative"])
    score = min(positive_hits * 0.015, 0.09) - min(negative_hits * 0.015, 0.06)
    if positive_hits:
        notes.append("role-keyword-fit")
    if negative_hits:
        notes.append("role-keyword-mismatch")
    if family == "backend" and positive_hits == 0:
        score -= 0.1
        notes.append("backend-signal-missing")
    return score, notes


def weight_text(record: ProjectRecord) -> str:
    fm = record.frontmatter
    parts: list[str] = []

    def repeat(value: Any, times: int) -> None:
        if not value:
            return
        if isinstance(value, list):
            text = " ".join(str(v) for v in value)
        else:
            text = str(value)
        parts.extend([text] * times)

    repeat(fm.get("cosine_title"), 4)
    repeat(fm.get("cosine_summary"), 4)
    repeat(fm.get("resume_keywords"), 3)
    repeat(fm.get("technology_keywords"), 3)
    repeat(fm.get("domain_keywords"), 2)
    repeat(fm.get("impact_keywords"), 2)
    repeat(fm.get("role_family_targets"), 2)
    repeat(record.sections.get("resume keywords"), 3)
    repeat(record.sections.get("elevator summary"), 3)
    repeat(record.sections.get("why it matters"), 2)
    repeat(record.sections.get("technical highlights"), 2)
    repeat(record.sections.get("resume bullet candidates"), 2)
    repeat(record.sections.get("evidence"), 1)
    repeat(record.family_name, 1)
    repeat(record.family_role, 1)
    repeat(record.family_keywords, 1)

    context_fm = record.context_frontmatter or {}
    context_sections = record.context_sections or {}
    repeat(context_fm.get("motivation_summary"), 3)
    repeat(context_fm.get("problem_trigger"), 2)
    repeat(context_fm.get("personal_connection"), 2)
    repeat(context_fm.get("why_now"), 1)
    repeat(context_fm.get("constraints_or_stakes"), 1)
    repeat(context_fm.get("context_keywords"), 2)
    repeat(context_fm.get("preferred_role_family_targets"), 2)
    repeat(context_sections.get("why this existed"), 3)
    repeat(context_sections.get("personal context"), 2)
    repeat(context_sections.get("resume intent"), 2)
    repeat(context_sections.get("things not obvious from the code"), 1)
    return "\n".join(parts)


def compute_idf(projects: list[ProjectRecord]) -> dict[str, float]:
    doc_freq: Counter[str] = Counter()
    for project in projects:
        doc_freq.update(set(tokenize(weight_text(project))))
    total = max(len(projects), 1)
    return {
        term: math.log((1 + total) / (1 + count)) + 1.0
        for term, count in doc_freq.items()
    }


def tfidf_vector(text: str, idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(tokenize(text))
    total = sum(counts.values()) or 1
    return {
        term: (count / total) * idf.get(term, 1.0) for term, count in counts.items()
    }


def cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    shared = set(a) & set(b)
    numerator = sum(a[t] * b[t] for t in shared)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if not norm_a or not norm_b:
        return 0.0
    return numerator / (norm_a * norm_b)


def normalize_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def role_match_strength(target: str | None, role_targets: set[str]) -> float:
    if not target or not role_targets:
        return 0.0

    target_norm = normalize_label(target)
    target_tokens = set(target_norm.split("-"))
    best = 0.0
    for candidate in role_targets:
        cand_tokens = set(candidate.split("-"))
        if candidate == target_norm:
            best = max(best, 1.0)
            continue
        if target_norm in candidate or candidate in target_norm:
            best = max(best, 0.8)
            continue
        overlap = len(target_tokens & cand_tokens)
        if overlap >= max(2, min(len(target_tokens), len(cand_tokens)) - 1):
            best = max(best, 0.6)
        elif overlap >= 2:
            best = max(best, 0.4)
    return best


def heuristic_score(
    record: ProjectRecord, target_text: str, role_family: str | None
) -> tuple[float, list[str]]:
    fm = record.frontmatter
    context_fm = record.context_frontmatter or {}
    notes: list[str] = []
    score = 0.0

    role_targets = {
        normalize_label(str(x)) for x in (fm.get("role_family_targets") or [])
    }
    role_targets.update(
        normalize_label(str(x))
        for x in (context_fm.get("preferred_role_family_targets") or [])
    )
    role_strength = role_match_strength(role_family, role_targets)
    if role_strength >= 1.0:
        score += 0.12
        notes.append("role-family-match")
    elif role_strength >= 0.8:
        score += 0.09
        notes.append("role-family-near-match")
    elif role_strength >= 0.6:
        score += 0.06
        notes.append("role-family-fuzzy-match")

    authorship = str(fm.get("authorship_confidence", "")).lower()
    if authorship == "high":
        score += 0.05
        notes.append("high-authorship-confidence")
    elif authorship == "medium":
        score += 0.02

    evidence = {normalize_label(str(x)) for x in (fm.get("evidence_keywords") or [])}
    if any(
        x in evidence for x in {"tests-inspected", "code-inspected", "source-inspected"}
    ):
        score += 0.04
        notes.append("code-evidence")
    if any(
        x in evidence
        for x in {
            "ci-inspected",
            "workflow-inspected",
            "github-metadata-inspected",
            "github-metadata-checked",
        }
    ):
        score += 0.02

    penalties = 0.0
    status = {normalize_label(str(x)) for x in (fm.get("status_keywords") or [])}
    primary_class = normalize_label(str(fm.get("primary_class", "")))
    primary_tokens = set(primary_class.split("-"))

    if any(
        token in primary_tokens
        for token in {
            "systems",
            "kernel",
            "security",
            "attestation",
            "embedded",
            "platform",
        }
    ):
        score += 0.03
        notes.append("high-signal-primary-class")

    if "design-only" in primary_class or "spec" in primary_class:
        penalties += 0.04
        notes.append("design-only-penalty")
    if any(token in primary_tokens for token in {"packaging", "prototype"}):
        penalties += 0.03
        notes.append("small-scope-penalty")
    if "school" in primary_class and "salvageable" not in primary_class:
        penalties += 0.04
    if "prototype" in status or "wip" in status:
        penalties += 0.01

    if any(x in evidence for x in {"no-readme", "no-readme-found"}):
        penalties += 0.02
        notes.append("documentation-gap")
    if "no-tests" in evidence or "no-tests" in status:
        penalties += 0.02
        notes.append("test-gap")
    if "no-ci" in status:
        penalties += 0.015
        notes.append("ci-gap")
    if any(
        x in status
        for x in {
            "single-visible-commit",
            "two-commit-history",
            "low-external-traction",
            "initial-prototype",
        }
    ):
        penalties += 0.015

    target_tokens = set(tokenize(target_text))
    project_tokens = set(tokenize(weight_text(record)))
    role_adjustment, role_notes = role_keyword_adjustment(project_tokens, role_family)
    score += role_adjustment
    notes.extend(role_notes)
    overlap = len(target_tokens & project_tokens)
    if overlap >= 8:
        score += 0.05
        notes.append("strong-keyword-overlap")
    elif overlap >= 4:
        score += 0.02

    if context_fm:
        score += 0.02
        notes.append("user-context-present")

    impressiveness = max(record.technical_impressiveness - 1.0, 0.0)
    if impressiveness > 0:
        score += min(impressiveness * 0.5, 0.06)
        notes.append("technical-impressiveness")

    return score - penalties, notes


def rank_projects(
    projects: list[ProjectRecord], target_text: str, role_family: str | None
) -> list[dict[str, Any]]:
    idf = compute_idf(projects)
    target_vector = tfidf_vector(target_text, idf)
    ranked: list[dict[str, Any]] = []
    for project in projects:
        base_text = weight_text(project)
        semantic = cosine_similarity(tfidf_vector(base_text, idf), target_vector)
        heuristic, notes = heuristic_score(project, target_text, role_family)
        final_score = semantic + heuristic
        ranked.append(
            {
                "project_id": project.project_id,
                "display_name": project.title,
                "source_display_name": project.source_title,
                "repo": project.frontmatter.get("repo", project.path.stem),
                "path": str(project.path),
                "semantic_score": round(semantic, 4),
                "heuristic_score": round(heuristic, 4),
                "final_score": round(final_score, 4),
                "technical_impressiveness": round(project.technical_impressiveness, 3),
                "role_family_targets": project.frontmatter.get(
                    "role_family_targets", []
                ),
                "cosine_title": project.frontmatter.get("cosine_title", ""),
                "family_id": project.family_id,
                "family_name": project.family_name,
                "family_role": project.family_role,
                "motivation_summary": (project.context_frontmatter or {}).get(
                    "motivation_summary", ""
                ),
                "notes": notes,
                "top_bullets": extract_bullets(
                    project.sections.get("resume bullet candidates", ""), limit=3
                ),
            }
        )
    ranked.sort(key=lambda item: item["final_score"], reverse=True)

    family_seen: Counter[str] = Counter()
    for item in ranked:
        base_score = item["final_score"]
        family_id = item.get("family_id")
        if family_id:
            penalty = 0.12 * family_seen[family_id]
            item["family_penalty"] = round(penalty, 4)
            item["diversified_score"] = round(base_score - penalty, 4)
            family_seen[family_id] += 1
            if penalty > 0:
                item["notes"].append("family-duplicate-penalty")
        else:
            item["family_penalty"] = 0.0
            item["diversified_score"] = round(base_score, 4)

    ranked.sort(key=lambda item: item["diversified_score"], reverse=True)
    return ranked


def extract_bullets(text: str, limit: int = 3) -> list[str]:
    bullets = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            bullets.append(stripped[2:].strip())
    return bullets[:limit]


def load_target_text(
    query: str | None, target_file: Path | None, role_family: str | None
) -> str:
    pieces: list[str] = []
    if role_family:
        pieces.append(role_family)
    if query:
        pieces.append(query)
    if target_file:
        pieces.append(target_file.read_text(encoding="utf-8"))
    return "\n\n".join(piece for piece in pieces if piece).strip()


def write_outputs(results: list[dict[str, Any]], output_dir: Path, label: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{label}.json"
    md_path = output_dir / f"{label}.md"

    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [f"# Ranking results: {label}", ""]
    for idx, item in enumerate(results, start=1):
        lines.append(f"## {idx}. {item['display_name']} ({item['repo']})")
        lines.append("")
        lines.append(f"- Final score: `{item['final_score']}`")
        if item.get("diversified_score") is not None:
            lines.append(f"- Diversified score: `{item['diversified_score']}`")
        lines.append(f"- Semantic score: `{item['semantic_score']}`")
        lines.append(f"- Heuristic score: `{item['heuristic_score']}`")
        if item.get("family_name"):
            lines.append(f"- Project family: {item['family_name']}")
        if item.get("family_role"):
            lines.append(f"- Family role: {item['family_role']}")
        if item["cosine_title"]:
            lines.append(f"- Retrieval title: {item['cosine_title']}")
        if item["motivation_summary"]:
            lines.append(f"- Why it existed: {item['motivation_summary']}")
        if item["notes"]:
            lines.append(f"- Notes: {', '.join(item['notes'])}")
        if item["top_bullets"]:
            lines.append("- Top bullets:")
            for bullet in item["top_bullets"]:
                lines.append(f"  - {bullet}")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rank markdown project records against a target query."
    )
    parser.add_argument(
        "--projects-dir",
        default="data/projects",
        help="Directory containing markdown project records",
    )
    parser.add_argument(
        "--context-dir",
        default="data/context",
        help="Directory containing optional per-project context overlays",
    )
    parser.add_argument(
        "--family-dir",
        default="data/families",
        help="Directory containing optional project family overlays",
    )
    parser.add_argument("--query", help="Inline target query or job description")
    parser.add_argument(
        "--target-file",
        type=Path,
        help="Markdown/text file containing target description",
    )
    parser.add_argument("--role-family", help="Optional role family bias label")
    parser.add_argument(
        "--top", type=int, default=10, help="Number of top results to emit"
    )
    parser.add_argument(
        "--output-dir", default="output", help="Directory for markdown/json outputs"
    )
    parser.add_argument("--label", default="ranking", help="Output filename prefix")
    args = parser.parse_args()

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
    projects = [
        attach_family(attach_context(record, context_dir), family_map)
        for path in project_paths
        if (record := load_record(path)) and should_include_record(record)
    ]
    if not projects:
        raise SystemExit(f"No eligible markdown project files found in {projects_dir}")
    results = rank_projects(projects, target_text, args.role_family)[: args.top]
    write_outputs(results, Path(args.output_dir), args.label)

    for item in results:
        print(f"{item['final_score']:.4f}\t{item['repo']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
