"""Gold-aware answer and citation metrics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def accuracy(outcomes: Iterable[bool | None]) -> float | None:
    """Return labelled accuracy, or ``None`` when no gold outcomes exist."""
    labelled = [outcome for outcome in outcomes if outcome is not None]
    return sum(labelled) / len(labelled) if labelled else None


def citation_metrics(
    citations: list[dict[str, Any]],
    retrieved_urls: set[str],
    gold_urls: set[str] | None = None,
) -> dict[str, float | int | None]:
    """Compute verifier precision, gold-source recall, and fabricated URL rate."""
    known = [citation for citation in citations if citation.get("supports_claim") is not None]
    precision = (
        sum(bool(citation["supports_claim"]) for citation in known) / len(known) if known else None
    )
    cited_urls = {str(citation.get("url", "")) for citation in citations}
    fabricated_count = sum(url not in retrieved_urls for url in cited_urls if url)
    hallucinated_rate = fabricated_count / len(cited_urls) if cited_urls else 0.0
    recall: float | None = None
    if gold_urls is not None:
        recall = len(cited_urls & gold_urls) / len(gold_urls) if gold_urls else 1.0
    return {
        "citation_precision": precision,
        "citation_recall": recall,
        "hallucinated_url_rate": hallucinated_rate,
        "citation_count": len(citations),
    }


def aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, float | int | None]:
    """Aggregate per-question metrics without treating missing labels as zero."""
    accuracy_value = accuracy(record.get("accuracy") for record in records)
    labelled_precision = [
        record["citation_precision"]
        for record in records
        if record.get("citation_precision") is not None
    ]
    labelled_recall = [
        record["citation_recall"] for record in records if record.get("citation_recall") is not None
    ]
    hallucination_rates = [float(record.get("hallucinated_url_rate", 0)) for record in records]
    return {
        "accuracy": accuracy_value,
        "citation_precision": sum(labelled_precision) / len(labelled_precision)
        if labelled_precision
        else None,
        "citation_recall": sum(labelled_recall) / len(labelled_recall) if labelled_recall else None,
        "hallucinated_url_rate": sum(hallucination_rates) / len(hallucination_rates)
        if hallucination_rates
        else None,
        "question_count": len(records),
        "accuracy_label_count": sum(record.get("accuracy") is not None for record in records),
    }
