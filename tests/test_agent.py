"""Offline orchestration tests for structured research answers."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from citecheck.agent import run_agent
from citecheck.config import Settings
from citecheck.schemas import Answer
from citecheck.tools import PageContent


def _tool_message() -> SimpleNamespace:
    """Build the page-reader tool call used by agent flow tests."""
    return SimpleNamespace(
        function=SimpleNamespace(
            name="read_page", arguments={"url": "https://example.test/article"}
        )
    )


def _valid_answer(source_ids: list[int] | None = None) -> str:
    """Return a schema-valid answer fixture referencing supplied IDs."""
    return json.dumps(
        {
            "summary": "The article reports a result.",
            "sections": [
                {
                    "heading": "Finding",
                    "points": [
                        {
                            "text": "The article reports a result.",
                            "source_ids": source_ids if source_ids is not None else [1],
                        }
                    ],
                },
                {
                    "heading": "Context",
                    "points": [
                        {
                            "text": "A related finding.",
                            "source_ids": source_ids if source_ids is not None else [1],
                        }
                    ],
                },
            ],
            "sources": [],
        }
    )


def test_agent_uses_schema_output_and_fetched_source_ids() -> None:
    """The final call uses Answer's JSON schema and emits verified fetched sources."""
    messages_seen: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []

    def fake_chat(**kwargs: Any) -> SimpleNamespace:
        messages_seen.append(kwargs)
        if "format" in kwargs:
            if kwargs["format"].get("title") == "Answer":
                content = _valid_answer()
            else:
                content = json.dumps({"supported": True, "reason": "Passage supports point."})
            return SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=[]))
        if len([call for call in messages_seen if "format" not in call]) == 1:
            message = SimpleNamespace(content="", tool_calls=[_tool_message()])
            return SimpleNamespace(message=message)
        return SimpleNamespace(message=SimpleNamespace(content="Page read.", tool_calls=[]))

    def fake_embed(**kwargs: Any) -> dict[str, list[list[float]]]:
        return {"embeddings": [[1.0, 0.0] for _ in kwargs["input"]]}

    answer = run_agent(
        "What does the article report?",
        Settings(chunk_size=200),
        chat_fn=fake_chat,
        embed_fn=fake_embed,
        page_reader=lambda url: PageContent(
            url=url,
            title="A study",
            text="The article reports a result in its abstract.",
            injection_flags=[],
        ),
        trace_fn=lambda path, trace: traces.append(trace.model_dump()),
    )
    final_call = next(
        call for call in messages_seen if call.get("format", {}).get("title") == "Answer"
    )
    point = answer.sections[0].points[0]
    assert final_call["format"] == Answer.model_json_schema()
    assert point.source_ids == [1]
    assert point.verified is True
    assert answer.sources[0].url == "https://example.test/article"
    assert {trace["tool"] for trace in traces} >= {"read_page", "embed", "llm_final", "verifier"}


def test_agent_drops_point_with_unknown_source_id() -> None:
    """A point citing an ID absent from this run's fetched passages is removed."""
    calls = 0

    def fake_chat(**kwargs: Any) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        if "format" in kwargs:
            content = _valid_answer([999])
            return SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=[]))
        if calls == 1:
            message = SimpleNamespace(content="", tool_calls=[_tool_message()])
            return SimpleNamespace(message=message)
        return SimpleNamespace(message=SimpleNamespace(content="Page read.", tool_calls=[]))

    answer = run_agent(
        "What does the article report?",
        Settings(verifier_enabled=False),
        chat_fn=fake_chat,
        embed_fn=lambda **kwargs: {"embeddings": [[1.0, 0.0] for _ in kwargs["input"]]},
        page_reader=lambda url: PageContent(
            url=url,
            title="A study",
            text="A relevant article passage.",
            injection_flags=[],
        ),
        trace_fn=lambda path, trace: None,
    )
    assert answer.sections == []
    assert answer.sources == []


def test_agent_retries_invalid_structured_output_once() -> None:
    """Invalid JSON is retried once with the same Answer schema."""
    final_calls = 0

    def fake_chat(**kwargs: Any) -> SimpleNamespace:
        nonlocal final_calls
        if "format" not in kwargs:
            return SimpleNamespace(message=SimpleNamespace(content="Done.", tool_calls=[]))
        final_calls += 1
        content = "not JSON" if final_calls == 1 else json.dumps(
            {"summary": "Evidence is insufficient.", "sections": [], "sources": []}
        )
        return SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=[]))

    answer = run_agent(
        "Question?",
        Settings(),
        chat_fn=fake_chat,
        trace_fn=lambda path, trace: None,
    )
    assert final_calls == 2
    assert answer.summary == "Evidence is insufficient to provide a supported answer."
