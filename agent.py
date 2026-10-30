"""Backward-compatible command-line entry point for CiteCheck."""

from __future__ import annotations

import sys

from citecheck.agent import run_agent as run_research
from citecheck.config import Settings
from citecheck.tools import read_page, web_search

MODEL = Settings.from_environment().model
MAX_STEPS = Settings.from_environment().max_steps
FUNCTIONS = {"web_search": web_search, "read_page": read_page}


def run_agent(question: str) -> str:
    """Run the structured agent and return a concise legacy-style string."""
    answer = run_research(question)
    sections = "\n\n".join(
        f"{section.heading}\n"
        + "\n".join(
            f"- {point.text} [{', '.join(map(str, point.source_ids))}]"
            for point in section.points
        )
        for section in answer.sections
    )
    sources = "\n".join(f"[{source.id}] {source.title}: {source.url}" for source in answer.sources)
    output = "\n\n".join(part for part in (answer.summary, sections) if part)
    return f"{output}\n\nSources:\n{sources}" if sources else output


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or input("Ask a research question: ")
    print("\n" + run_agent(question))
