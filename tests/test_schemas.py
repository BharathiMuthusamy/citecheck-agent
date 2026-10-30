"""Tests for the structured answer schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from citecheck.schemas import Answer


def test_answer_rejects_points_without_source_ids() -> None:
    """Every structured point must name at least one source ID."""
    with pytest.raises(ValidationError):
        Answer.model_validate(
            {
                "summary": "A short summary.",
                "sections": [
                    {
                        "heading": "Findings",
                        "points": [{"text": "An unsupported point.", "source_ids": []}],
                    },
                    {
                        "heading": "Additional findings",
                        "points": [{"text": "Another.", "source_ids": [1]}],
                    },
                ],
                "sources": [],
            }
        )