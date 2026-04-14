from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from urllib import error, request


WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+.#/-]*")
COMMON_MATCH_STOPWORDS = {
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
    "description",
    "design",
    "develop",
    "developing",
    "development",
    "engineer",
    "engineering",
    "experience",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "its",
    "maintain",
    "maintaining",
    "job",
    "jobs",
    "of",
    "on",
    "or",
    "project",
    "projects",
    "qualifications",
    "qualification",
    "responsibilities",
    "requirements",
    "role",
    "roles",
    "skills",
    "work",
    "working",
    "team",
    "teams",
    "that",
    "the",
    "their",
    "this",
    "to",
    "using",
    "use",
    "used",
    "user",
    "we",
    "will",
    "with",
    "you",
    "your",
}
KEYWORD_STOPWORDS = COMMON_MATCH_STOPWORDS | {
    "candidate",
    "candidates",
    "companies",
    "company",
    "deliveries",
    "delivery",
    "get",
    "have",
    "know",
    "markets",
    "our",
    "platform",
}


@dataclass(frozen=True)
class SemanticConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    use_embeddings: bool = True
    use_cross_encoder: bool = True
    rerank_top_k: int = 8
    jd_chunk_words: int = 120
    jd_chunk_overlap: int = 30
    ollama_url: str | None = None
    ollama_model: str | None = None
    use_ollama_expansion: bool = False
    ollama_timeout_seconds: float = 8.0

    @classmethod
    def from_env(cls) -> "SemanticConfig":
        ollama_raw = os.getenv("SEMANTHA_OLLAMA_URL") or os.getenv("OLLAMA_HOST")
        return cls(
            embedding_model=os.getenv(
                "SEMANTHA_EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            cross_encoder_model=os.getenv(
                "SEMANTHA_CROSS_ENCODER_MODEL",
                "cross-encoder/ms-marco-MiniLM-L6-v2",
            ),
            use_embeddings=os.getenv("SEMANTHA_DISABLE_EMBEDDINGS", "").lower()
            not in {"1", "true", "yes"},
            use_cross_encoder=os.getenv("SEMANTHA_DISABLE_CROSS_ENCODER", "").lower()
            not in {"1", "true", "yes"},
            rerank_top_k=_env_int("SEMANTHA_RERANK_TOP_K", 8),
            jd_chunk_words=_env_int("SEMANTHA_JD_CHUNK_WORDS", 120),
            jd_chunk_overlap=_env_int("SEMANTHA_JD_CHUNK_OVERLAP", 30),
            ollama_url=_normalize_ollama_url(ollama_raw),
            ollama_model=os.getenv("SEMANTHA_OLLAMA_MODEL"),
            use_ollama_expansion=os.getenv(
                "SEMANTHA_ENABLE_OLLAMA_EXPANSION", ""
            ).lower()
            in {"1", "true", "yes"},
            ollama_timeout_seconds=_env_float("SEMANTHA_OLLAMA_TIMEOUT", 8.0),
        )


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(1.0, float(raw))
    except ValueError:
        return default


def _normalize_ollama_url(raw: str | None) -> str | None:
    if not raw:
        return None
    value = raw.strip()
    if not value:
        return None
    if not re.match(r"^https?://", value, flags=re.IGNORECASE):
        value = f"http://{value}"
    return value.rstrip("/")


def chunk_text(text: str, *, chunk_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text.strip()]
    step = max(1, chunk_words - max(0, overlap_words))
    chunks: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_words]
        if not window:
            break
        chunks.append(" ".join(window).strip())
        if start + chunk_words >= len(words):
            break
    return chunks or [text.strip()]


def cosine_similarity_dense(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    numerator = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if not norm_a or not norm_b:
        return 0.0
    return numerator / (norm_a * norm_b)


def squash_score(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def embed_texts(
    texts: list[str], config: SemanticConfig
) -> tuple[list[list[float]] | None, str | None]:
    if not config.use_embeddings:
        return None, "embeddings-disabled"
    try:
        model = _load_sentence_transformer(config.embedding_model)
        vectors = model.encode(
            texts,
            convert_to_numpy=False,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as exc:  # pragma: no cover - optional dependency/runtime
        return None, f"embedding-unavailable: {exc}"

    normalized: list[list[float]] = []
    for vector in vectors:
        if hasattr(vector, "tolist"):
            normalized.append([float(item) for item in vector.tolist()])
        else:
            normalized.append([float(item) for item in vector])
    return normalized, None


def rerank_documents(
    query: str, documents: list[str], config: SemanticConfig
) -> tuple[list[dict[str, Any]] | None, str | None]:
    if not config.use_cross_encoder or not documents:
        return None, "cross-encoder-disabled"
    try:
        model = _load_cross_encoder(config.cross_encoder_model)
        ranks = model.rank(
            query, documents, top_k=min(len(documents), config.rerank_top_k)
        )
    except Exception as exc:  # pragma: no cover - optional dependency/runtime
        return None, f"cross-encoder-unavailable: {exc}"
    normalized: list[dict[str, Any]] = []
    for item in ranks:
        score = float(item.get("score", 0.0))
        normalized.append(
            {
                "corpus_id": int(item["corpus_id"]),
                "score": score,
                "normalized_score": round(squash_score(score / 4.0), 4),
            }
        )
    return normalized, None


def extract_keywords_locally(text: str, *, limit: int = 24) -> list[str]:
    counts: dict[str, int] = {}
    for token in WORD_RE.findall(text.lower()):
        token = token.strip(".,:;!?()[]{}\"'`")
        if not token:
            continue
        if token.isdigit():
            continue
        if len(token) < 2:
            continue
        if token in KEYWORD_STOPWORDS:
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
    return [token for token, _count in ranked[:limit]]


def expand_keywords_with_ollama(
    text: str, base_keywords: list[str], config: SemanticConfig
) -> tuple[list[str], str | None]:
    if not config.use_ollama_expansion:
        return [], "ollama-expansion-disabled"
    if not config.ollama_url or not config.ollama_model:
        return [], "ollama-not-configured"

    prompt = (
        "Extract concise ATS-style resume keywords from the job description. "
        "Return strict JSON with key 'keywords' containing up to 20 lowercase strings. "
        "Prefer specific technologies, domains, protocols, and responsibilities.\n\n"
        f"Existing keywords: {', '.join(base_keywords[:15])}\n\n"
        f"Job description:\n{text[:6000]}"
    )
    payload = json.dumps(
        {
            "model": config.ollama_model,
            "prompt": prompt,
            "stream": False,
            "format": {
                "type": "object",
                "properties": {
                    "keywords": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["keywords"],
            },
            "options": {"temperature": 0},
        }
    ).encode("utf-8")
    url = config.ollama_url.rstrip("/") + "/api/generate"
    req = request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with request.urlopen(req, timeout=config.ollama_timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return [], f"ollama-unavailable: {exc}"

    raw_response = str(body.get("response") or "").strip()
    if not raw_response:
        return [], "ollama-empty-response"
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        return [], "ollama-invalid-json"
    keywords = parsed.get("keywords")
    if not isinstance(keywords, list):
        return [], "ollama-missing-keywords"
    cleaned = []
    for item in keywords:
        if not isinstance(item, str):
            continue
        keyword = item.strip().lower()
        if not keyword:
            continue
        normalized_terms = extract_keywords_locally(keyword, limit=6) or [keyword]
        for term in normalized_terms:
            if term and term not in cleaned:
                cleaned.append(term)
    return cleaned[:20], None
