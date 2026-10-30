"""Chunking and local-embedding retrieval helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from urllib.parse import urlsplit

import numpy as np
import ollama
from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    """A text chunk and its cosine similarity to a search query."""

    text: str
    score: float


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    """Split text into overlapping, whitespace-trimmed character chunks."""
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    normalized = " ".join(text.split())
    if not normalized:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(start + 1, end - overlap)
    return chunks


def embed_texts(
    texts: Sequence[str],
    embed_model: str = "nomic-embed-text",
    embed_fn: Callable[..., object] = ollama.embed,
) -> list[list[float]]:
    """Embed text batches through an injectable Ollama-compatible function."""
    if not texts:
        return []
    response = embed_fn(model=embed_model, input=list(texts))
    embeddings = response.get("embeddings") if isinstance(response, dict) else response.embeddings
    return [[float(value) for value in embedding] for embedding in embeddings]


def rank_chunks(
    query: str,
    chunks: Sequence[str],
    *,
    embed_model: str = "nomic-embed-text",
    top_k: int = 5,
    embed_fn: Callable[..., object] = ollama.embed,
) -> list[RetrievedChunk]:
    """Return the top chunks by cosine similarity using local embeddings."""
    if not chunks or top_k < 1:
        return []
    vectors = np.asarray(embed_texts([query, *chunks], embed_model, embed_fn), dtype=float)
    query_vector = vectors[0]
    chunk_vectors = vectors[1:]
    denominator = np.linalg.norm(chunk_vectors, axis=1) * np.linalg.norm(query_vector)
    scores = np.divide(
        chunk_vectors @ query_vector,
        denominator,
        out=np.zeros(len(chunk_vectors), dtype=float),
        where=denominator != 0,
    )
    best = np.argsort(scores)[::-1][:top_k]
    return [RetrievedChunk(text=chunks[index], score=float(scores[index])) for index in best]


def filter_relevant_items(
    query: str,
    items: Sequence[Mapping[str, str]],
    *,
    threshold: float = 0.45,
    top_k: int = 5,
    embed_model: str = "nomic-embed-text",
    embed_fn: Callable[..., object] = ollama.embed,
    deduplicate_domains: bool = True,
) -> list[tuple[Mapping[str, str], float]]:
    """Keep the most relevant, domain-distinct candidates above a cosine threshold."""
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("Relevance threshold must be between -1 and 1")
    if top_k < 1 or not items:
        return []
    candidate_texts = [
        item.get("text") or " ".join((item.get("title", ""), item.get("snippet", "")))
        for item in items
    ]
    vectors = np.asarray(embed_texts([query, *candidate_texts], embed_model, embed_fn), dtype=float)
    query_vector = vectors[0]
    candidate_vectors = vectors[1:]
    denominator = np.linalg.norm(candidate_vectors, axis=1) * np.linalg.norm(query_vector)
    scores = np.divide(
        candidate_vectors @ query_vector,
        denominator,
        out=np.zeros(len(candidate_vectors), dtype=float),
        where=denominator != 0,
    )
    ranked_indices = np.argsort(scores)[::-1]
    unique_hosts: set[str] = set()
    relevant: list[tuple[Mapping[str, str], float]] = []
    for index in ranked_indices:
        score = float(scores[index])
        if score < threshold:
            continue
        host = (urlsplit(items[index].get("url", "")).hostname or "").lower()
        if not host:
            continue
        host = host.removeprefix("www.")
        if deduplicate_domains and host in unique_hosts:
            continue
        unique_hosts.add(host)
        relevant.append((items[index], score))
        if len(relevant) >= min(top_k, 5):
            break
    return relevant
