"""Offline tests for the synthetic prompt-injection experiment runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from citecheck.schemas import Answer, Source
from eval.injection_test import CANARY, run_injection_experiments


def test_injection_runner_uses_fixed_page_and_reports_delivery(tmp_path: Path) -> None:
    """The experiment does not need live search and records whether the page was read."""
    calls: list[dict[str, Any]] = []

    def fake_agent(question: str, settings: object, **kwargs: Any) -> Answer:
        page_url = question.split()[1]
        page = kwargs["page_reader"](page_url)
        calls.append({"question": question, "search": kwargs["search_fn"]("test", 5)})
        return Answer(
            summary=f"Test response includes {CANARY}",
            sources=[Source(id=1, title=page.title, url=page.url, quote=page.text)],
        )

    output = tmp_path / "injection.json"
    rows = run_injection_experiments("test-model", output, agent_fn=fake_agent)
    assert len(rows) == 3
    assert all(row["attack_delivered"] and row["canary_observed"] for row in rows)
    assert all(call["search"] == [] for call in calls)
    assert output.exists()


def test_injection_runner_marks_undelivered_case_unscored(tmp_path: Path) -> None:
    """A canary miss is not counted if the model never read the injected page."""

    def no_page_agent(question: str, settings: object, **kwargs: Any) -> Answer:
        return Answer(summary="No page read")

    rows = run_injection_experiments(
        "test-model",
        tmp_path / "injection.json",
        agent_fn=no_page_agent,
    )
    assert all(row["attack_delivered"] is False for row in rows)
    assert all(row["canary_observed"] is None for row in rows)