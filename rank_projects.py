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

from semantic_features import (
    COMMON_MATCH_STOPWORDS,
    SemanticConfig,
    chunk_text,
    cosine_similarity_dense,
    embed_texts,
    expand_keywords_with_ollama,
    extract_keywords_locally,
    rerank_documents,
)


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+.#/-]*")
SECTION_RE = re.compile(r"^##\s+(.*)$")
STOPWORDS = set(COMMON_MATCH_STOPWORDS)

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
    for candidate in [
        context_root / f"{record.project_id}.md",
        context_root / f"{record.path.stem}.md",
    ]:
        if candidate.exists():
            frontmatter, body = split_frontmatter(candidate.read_text(encoding="utf-8"))
            record.context_frontmatter = frontmatter
            record.context_sections = parse_sections(body)
            break
    return record


def load_family_map(family_root: Path | None) -> dict[str, dict[str, Any]]:
    if family_root is None or not family_root.exists():
        return {}
    family_map: dict[str, dict[str, Any]] = {}
    for path in sorted(family_root.glob("*.md")):
        frontmatter, _ = split_frontmatter(path.read_text(encoding="utf-8"))
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
    record.family_keywords = [
        str(item) for item in (family.get("family_keywords") or [])
    ]
    return record


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS]


def normalize_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def weight_text(record: ProjectRecord) -> str:
    fm = record.frontmatter
    parts: list[str] = []

    def repeat(value: Any, times: int) -> None:
        if not value:
            return
        text = (
            " ".join(str(v) for v in value) if isinstance(value, list) else str(value)
        )
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


def semantic_text(record: ProjectRecord) -> str:
    fm = record.frontmatter
    context_fm = record.context_frontmatter or {}
    context_sections = record.context_sections or {}
    parts: list[str] = [
        record.title,
        str(fm.get("cosine_title") or ""),
        str(fm.get("cosine_summary") or ""),
        str(record.sections.get("elevator summary") or ""),
        str(record.sections.get("why it matters") or ""),
        str(context_fm.get("motivation_summary") or ""),
        str(context_sections.get("synthesis") or ""),
    ]
    for key in ["resume_keywords", "technology_keywords", "domain_keywords"]:
        value = fm.get(key)
        if isinstance(value, list):
            parts.append(" ".join(str(item) for item in value))
        elif value:
            parts.append(str(value))
    parts.extend(
        extract_bullets(record.sections.get("resume bullet candidates", ""), limit=4)
    )
    parts.extend(
        extract_bullets(context_sections.get("suggested bullet points", ""), limit=4)
    )
    return "\n".join(part for part in parts if part).strip()


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
        elif target_norm in candidate or candidate in target_norm:
            best = max(best, 0.8)
        else:
            overlap = len(target_tokens & cand_tokens)
            if overlap >= max(2, min(len(target_tokens), len(cand_tokens)) - 1):
                best = max(best, 0.6)
            elif overlap >= 2:
                best = max(best, 0.4)
    return best


def role_keyword_adjustment(
    tokens: set[str], role_family: str | None
) -> tuple[float, list[str]]:
    if not role_family:
        return 0.0, []
    signals = ROLE_SIGNAL_KEYWORDS.get(normalize_label(role_family))
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
    if normalize_label(role_family) == "backend" and positive_hits == 0:
        score -= 0.1
        notes.append("backend-signal-missing")
    return score, notes


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


def extract_bullets(text: str, limit: int = 3) -> list[str]:
    bullets = [
        line.strip()[2:].strip()
        for line in text.splitlines()
        if line.strip().startswith("- ")
    ]
    return bullets[:limit]


def collect_project_keywords(record: ProjectRecord) -> set[str]:
    values: list[str] = []
    for key in [
        "resume_keywords",
        "technology_keywords",
        "domain_keywords",
        "impact_keywords",
        "role_family_targets",
    ]:
        raw = record.frontmatter.get(key)
        if isinstance(raw, list):
            values.extend(str(item) for item in raw)
        elif raw:
            values.append(str(raw))
    for key in ["resume keywords", "technical highlights", "elevator summary"]:
        if record.sections.get(key):
            values.append(record.sections[key])
    if record.family_keywords:
        values.extend(record.family_keywords)
    context_fm = record.context_frontmatter or {}
    if context_fm.get("context_keywords"):
        values.extend(str(item) for item in context_fm.get("context_keywords") or [])
    return set(tokenize("\n".join(values)))


def keyword_match_score(
    record: ProjectRecord, target_keywords: list[str], expanded_keywords: list[str]
) -> tuple[float, dict[str, Any]]:
    project_keywords = collect_project_keywords(record)
    required = [keyword for keyword in target_keywords if keyword]
    expanded = [keyword for keyword in expanded_keywords if keyword]
    direct_hits = [keyword for keyword in required if keyword in project_keywords]
    expanded_hits = [
        keyword
        for keyword in expanded
        if keyword in project_keywords and keyword not in direct_hits
    ]
    combined_total = max(1, len(required) + min(len(expanded), 8))
    score = min((len(direct_hits) + 0.6 * len(expanded_hits)) / combined_total, 1.0)
    return score, {
        "direct_hits": direct_hits[:12],
        "expanded_hits": expanded_hits[:8],
        "missing_keywords": [
            keyword for keyword in required[:12] if keyword not in direct_hits
        ],
    }


def build_match_report(
    *,
    label: str,
    target_text: str,
    role_family: str | None,
    semantic_config: SemanticConfig,
    target_keywords: list[str],
    expanded_keywords: list[str],
    diagnostics: list[str],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "match-report/v1",
        "label": label,
        "role_family": role_family,
        "target_keywords": target_keywords,
        "expanded_keywords": expanded_keywords,
        "semantic_models": {
            "embedding_model": semantic_config.embedding_model,
            "cross_encoder_model": semantic_config.cross_encoder_model,
        },
        "diagnostics": diagnostics,
        "job_description_chunks": chunk_text(
            target_text,
            chunk_words=semantic_config.jd_chunk_words,
            overlap_words=semantic_config.jd_chunk_overlap,
        ),
        "results": [
            {
                "project_id": item["project_id"],
                "display_name": item["display_name"],
                "final_score": item["final_score"],
                "diversified_score": item["diversified_score"],
                "score_breakdown": item["score_breakdown"],
                "match_report": item["match_report"],
                "notes": item["notes"],
            }
            for item in results
        ],
    }


def extract_ranking_metadata(
    results: list[dict[str, Any]], target_text: str
) -> dict[str, Any]:
    if not results:
        return {
            "target_keywords": extract_keywords_locally(target_text),
            "expanded_keywords": [],
            "diagnostics": [],
        }
    first = results[0]
    return {
        "target_keywords": first.get("target_keywords_used")
        or extract_keywords_locally(target_text),
        "expanded_keywords": first.get("expanded_keywords_used") or [],
        "diagnostics": first.get("ranking_diagnostics") or [],
    }


def rank_projects(
    projects: list[ProjectRecord],
    target_text: str,
    role_family: str | None,
    semantic_config: SemanticConfig | None = None,
) -> list[dict[str, Any]]:
    semantic_config = semantic_config or SemanticConfig.from_env()
    diagnostics: list[str] = []
    idf = compute_idf(projects)
    target_vector = tfidf_vector(target_text, idf)
    target_keywords = extract_keywords_locally(target_text)
    expanded_keywords, expansion_error = expand_keywords_with_ollama(
        target_text, target_keywords, semantic_config
    )
    expanded_keywords = [
        keyword for keyword in expanded_keywords if keyword not in set(target_keywords)
    ]
    if expansion_error:
        diagnostics.append(expansion_error)
    target_chunks = chunk_text(
        target_text,
        chunk_words=semantic_config.jd_chunk_words,
        overlap_words=semantic_config.jd_chunk_overlap,
    )
    project_texts = [weight_text(project) for project in projects]
    project_semantic_texts = [semantic_text(project) for project in projects]
    query_embeddings, embedding_error = embed_texts(
        [target_text, *target_chunks, *project_semantic_texts], semantic_config
    )
    if embedding_error:
        diagnostics.append(embedding_error)
    dense_target: list[float] | None = None
    dense_chunks: list[list[float]] = []
    dense_projects: list[list[float]] = []
    if query_embeddings:
        dense_target = query_embeddings[0]
        dense_chunks = query_embeddings[1 : 1 + len(target_chunks)]
        dense_projects = query_embeddings[1 + len(target_chunks) :]

    ranked: list[dict[str, Any]] = []
    backend_core_terms = {"backend", "django", "api", "service", "services"}
    for index, project in enumerate(projects):
        lexical_score = cosine_similarity(
            tfidf_vector(project_texts[index], idf), target_vector
        )
        dense_score = (
            cosine_similarity_dense(dense_target, dense_projects[index])
            if dense_target and index < len(dense_projects)
            else lexical_score
        )
        chunk_scores = (
            [
                cosine_similarity_dense(chunk_vector, dense_projects[index])
                for chunk_vector in dense_chunks
            ]
            if dense_chunks and index < len(dense_projects)
            else [lexical_score]
        )
        best_chunk_score = max(chunk_scores) if chunk_scores else 0.0
        best_chunk_index = chunk_scores.index(best_chunk_score) if chunk_scores else 0
        keyword_score, keyword_details = keyword_match_score(
            project, target_keywords, expanded_keywords
        )
        heuristic, notes = heuristic_score(project, target_text, role_family)
        normalized_heuristic = max(min(0.5 + heuristic, 1.0), 0.0)
        semantic_available = bool(dense_target and index < len(dense_projects))
        if semantic_available:
            lexical_component = 0.20 * lexical_score
            embedding_component = 0.16 * dense_score
            chunk_component = 0.10 * best_chunk_score
            keyword_component = 0.25 * keyword_score
            heuristic_component = 0.29 * normalized_heuristic
            final_score = (
                lexical_component
                + embedding_component
                + chunk_component
                + keyword_component
                + heuristic_component
            )
        else:
            dense_score = 0.0
            best_chunk_score = 0.0
            lexical_component = 0.45 * lexical_score
            embedding_component = 0.0
            chunk_component = 0.0
            keyword_component = 0.25 * keyword_score
            heuristic_component = 0.30 * normalized_heuristic
            final_score = lexical_component + keyword_component + heuristic_component
            notes.append("lexical-fallback")
        if normalize_label(role_family or "") == "backend" and not (
            (
                set(keyword_details["direct_hits"])
                | set(keyword_details["expanded_hits"])
            )
            & backend_core_terms
        ):
            final_score -= 0.04
            notes.append("backend-core-signal-weak")
        ranked.append(
            {
                "project_id": project.project_id,
                "display_name": project.title,
                "source_display_name": project.source_title,
                "repo": project.frontmatter.get("repo", project.path.stem),
                "path": str(project.path),
                "semantic_score": round(dense_score, 4),
                "lexical_score": round(lexical_score, 4),
                "chunk_score": round(best_chunk_score, 4),
                "keyword_score": round(keyword_score, 4),
                "heuristic_score": round(heuristic, 4),
                "rerank_score": 0.0,
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
                "score_breakdown": {
                    "lexical": round(lexical_score, 4),
                    "embedding": round(dense_score, 4),
                    "job_chunk": round(best_chunk_score, 4),
                    "keyword_overlap": round(keyword_score, 4),
                    "heuristic_adjustment": round(heuristic, 4),
                    "lexical_component": round(lexical_component, 4),
                    "embedding_component": round(embedding_component, 4),
                    "job_chunk_component": round(chunk_component, 4),
                    "keyword_component": round(keyword_component, 4),
                    "heuristic_component": round(heuristic_component, 4),
                    "cross_encoder": 0.0,
                },
                "match_report": {
                    "keyword_hits": keyword_details["direct_hits"],
                    "expanded_keyword_hits": keyword_details["expanded_hits"],
                    "missing_keywords": keyword_details["missing_keywords"],
                    "best_job_chunk_index": best_chunk_index,
                    "best_job_chunk_excerpt": target_chunks[best_chunk_index][:240]
                    if target_chunks
                    else target_text[:240],
                },
                "target_keywords_used": target_keywords[:],
                "expanded_keywords_used": expanded_keywords[:],
                "ranking_diagnostics": diagnostics[:],
            }
        )

    ranked.sort(key=lambda item: item["final_score"], reverse=True)
    rerank_docs = [
        project_semantic_texts[
            projects.index(
                next(
                    project
                    for project in projects
                    if project.project_id == item["project_id"]
                )
            )
        ]
        for item in ranked[: semantic_config.rerank_top_k]
    ]
    rerank_results, rerank_error = rerank_documents(
        target_text, rerank_docs, semantic_config
    )
    if rerank_error:
        diagnostics.append(rerank_error)
    if rerank_results:
        for rerank in rerank_results:
            item = ranked[rerank["corpus_id"]]
            item["rerank_score"] = rerank["normalized_score"]
            item["score_breakdown"]["cross_encoder"] = rerank["normalized_score"]
            item["final_score"] = round(
                item["final_score"] + 0.10 * rerank["normalized_score"], 4
            )
            item["notes"].append("cross-encoder-rerank")

    family_seen: Counter[str] = Counter()
    ranked.sort(key=lambda item: item["final_score"], reverse=True)
    for item in ranked:
        base_score = float(item["final_score"])
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
        item["ranking_diagnostics"] = diagnostics[:]
    ranked.sort(key=lambda item: item["diversified_score"], reverse=True)
    return ranked


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


def write_outputs(
    results: list[dict[str, Any]],
    output_dir: Path,
    label: str,
    *,
    match_report: dict[str, Any] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{label}.json"
    md_path = output_dir / f"{label}.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if match_report is not None:
        (output_dir / f"{label}-match-report.json").write_text(
            json.dumps(match_report, indent=2), encoding="utf-8"
        )
    lines = [f"# Ranking results: {label}", ""]
    for idx, item in enumerate(results, start=1):
        lines.append(f"## {idx}. {item['display_name']} ({item['repo']})")
        lines.append("")
        lines.append(f"- Final score: `{item['final_score']}`")
        lines.append(f"- Diversified score: `{item['diversified_score']}`")
        lines.append(f"- Lexical score: `{item['lexical_score']}`")
        lines.append(f"- Embedding score: `{item['semantic_score']}`")
        lines.append(f"- Job chunk score: `{item['chunk_score']}`")
        lines.append(f"- Keyword score: `{item['keyword_score']}`")
        lines.append(f"- Heuristic score: `{item['heuristic_score']}`")
        if item.get("rerank_score"):
            lines.append(f"- Cross-encoder rerank score: `{item['rerank_score']}`")
        if item.get("family_name"):
            lines.append(f"- Project family: {item['family_name']}")
        if item["match_report"]["keyword_hits"]:
            lines.append(
                f"- Keyword hits: {', '.join(item['match_report']['keyword_hits'])}"
            )
        if item["match_report"]["expanded_keyword_hits"]:
            lines.append(
                f"- Expanded keyword hits: {', '.join(item['match_report']['expanded_keyword_hits'])}"
            )
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
    parser.add_argument("--projects-dir", default="data/projects")
    parser.add_argument("--context-dir", default="data/context")
    parser.add_argument("--family-dir", default="data/families")
    parser.add_argument("--query")
    parser.add_argument("--target-file", type=Path)
    parser.add_argument("--role-family")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--label", default="ranking")
    parser.add_argument("--embedding-model")
    parser.add_argument("--cross-encoder-model")
    parser.add_argument("--disable-embeddings", action="store_true")
    parser.add_argument("--disable-cross-encoder", action="store_true")
    parser.add_argument("--enable-ollama-expansion", action="store_true")
    parser.add_argument("--ollama-url")
    parser.add_argument("--ollama-model")
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

    config = SemanticConfig.from_env()
    if args.embedding_model:
        config = SemanticConfig(
            **{**config.__dict__, "embedding_model": args.embedding_model}
        )
    if args.cross_encoder_model:
        config = SemanticConfig(
            **{**config.__dict__, "cross_encoder_model": args.cross_encoder_model}
        )
    if args.disable_embeddings:
        config = SemanticConfig(**{**config.__dict__, "use_embeddings": False})
    if args.disable_cross_encoder:
        config = SemanticConfig(**{**config.__dict__, "use_cross_encoder": False})
    if args.enable_ollama_expansion:
        config = SemanticConfig(**{**config.__dict__, "use_ollama_expansion": True})
    if args.ollama_url:
        config = SemanticConfig(**{**config.__dict__, "ollama_url": args.ollama_url})
    if args.ollama_model:
        config = SemanticConfig(
            **{**config.__dict__, "ollama_model": args.ollama_model}
        )

    results = rank_projects(projects, target_text, args.role_family, config)
    metadata = extract_ranking_metadata(results, target_text)
    report = build_match_report(
        label=args.label,
        target_text=target_text,
        role_family=args.role_family,
        semantic_config=config,
        target_keywords=metadata["target_keywords"],
        expanded_keywords=metadata["expanded_keywords"],
        diagnostics=metadata["diagnostics"],
        results=results[: args.top],
    )
    write_outputs(
        results[: args.top], Path(args.output_dir), args.label, match_report=report
    )
    for item in results[: args.top]:
        print(f"{item['diversified_score']:.4f}\t{item['repo']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
