"""Tests for numbered point citations in the answer renderer."""

from __future__ import annotations

from citecheck.presentation import render_point_html
from citecheck.schemas import Answer, Point, Section, Source


def test_renderer_produces_numbered_citation_link_from_answer() -> None:
    """A sample answer point links its source ID to the source-list anchor."""
    answer = Answer(
        summary="AI enables machine-based tasks.",
        sections=[
            Section(
                heading="Definition",
                points=[Point(text="AI supports task automation.", source_ids=[1], verified=True)],
            ),
            Section(
                heading="Use",
                points=[Point(text="AI processes input data.", source_ids=[1], verified=True)],
            ),
        ],
        sources=[
            Source(
                id=1,
                title="AI overview",
                url="https://example.test/ai",
                quote="AI supports task automation.",
            )
        ],
    )

    rendered = render_point_html(answer.sections[0].points[0], {1: answer.sources[0]})

    assert ">[1]</a>" in rendered
    assert 'href="#source-1"' in rendered
    assert "Verified" in rendered