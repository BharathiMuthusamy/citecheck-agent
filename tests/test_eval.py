"""Offline tests for corpus integrity and honest metric reporting."""

from __future__ import annotations

from pathlib import Path

from eval.metrics import accuracy, aggregate_metrics, citation_metrics
from eval.run_eval import load_dataset


def test_dataset_has_fifty_prompt_only_questions() -> None:
    """The starter corpus contains 50 questions and no fabricated labels."""
    dataset = load_dataset(Path(__file__).parents[1] / "eval" / "dataset.jsonl")
    assert len(dataset) == 50
    assert all("gold_answer" not in entry and "gold_urls" not in entry for entry in dataset)


def test_accuracy_is_unavailable_without_labels() -> None:
    """Unlabelled evaluation does not report a misleading zero accuracy."""
    assert accuracy([None, None]) is None
    summary = aggregate_metrics([{"accuracy": None, "citation_recall": None}])
    assert summary["accuracy"] is None
    assert summary["accuracy_label_count"] == 0


def test_citation_metrics_use_retrieved_urls_and_optional_gold() -> None:
    """Unsupported sources are counted and gold recall stays unavailable if omitted."""
    metrics = citation_metrics(
        [{"url": "https://source.test/", "supports_claim": True}],
        {"https://source.test/"},
    )
    assert metrics["citation_precision"] == 1.0
    assert metrics["citation_recall"] is None
    assert metrics["hallucinated_url_rate"] == 0.0


def test_citation_metrics_detect_unretrieved_urls() -> None:
    """A citation absent from fetched evidence is measured as a hallucinated URL."""
    metrics = citation_metrics(
        [{"url": "https://invented.test/", "supports_claim": False}],
        set(),
        {"https://gold.test/"},
    )
    assert metrics["hallucinated_url_rate"] == 1.0
    assert metrics["citation_recall"] == 0.0
