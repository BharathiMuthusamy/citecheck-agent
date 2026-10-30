"""Tests for the embedding-based relevance filter."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from citecheck.retrieval import filter_relevant_items


def test_relevance_filter_drops_unrelated_snippets() -> None:
    """Candidates under the threshold never reach the filtered result list."""
    items = [
        {"title": "Relevant", "snippet": "Answer about AI", "url": "https://ai.example/a"},
        {
            "title": "Unrelated",
            "snippet": "A recipe for bread",
            "url": "https://food.example/b",
        },
    ]

    def fake_embed(**kwargs: Any) -> dict[str, list[list[float]]]:
        return {"embeddings": [[1, 0], [0.9, 0.1], [0, 1]]}

    filtered = filter_relevant_items("AI", items, threshold=0.45, embed_fn=fake_embed)
    assert [item["title"] for item, _ in filtered] == ["Relevant"]


def test_relevance_filter_deduplicates_host_and_caps_at_five() -> None:
    """Only one candidate per host and at most five candidates are returned."""
    items = [
        {"title": str(index), "text": "relevant passage", "url": f"https://site{index}.test/a"}
        for index in range(7)
    ]
    items.append(
        {"title": "duplicate host", "text": "relevant duplicate", "url": "https://www.site0.test/b"}
    )

    def fake_embed(**kwargs: Any) -> dict[str, list[list[float]]]:
        return {"embeddings": [[1, 0] for _ in kwargs["input"]]}

    filtered = filter_relevant_items("query", items, embed_fn=fake_embed)
    assert len(filtered) == 5
    hosts = {
        (urlsplit(item["url"]).hostname or "").removeprefix("www.") for item, _ in filtered
    }
    assert len(hosts) == 5