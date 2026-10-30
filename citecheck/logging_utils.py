"""JSONL trace logging for research-agent steps."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class StepTrace(BaseModel):
    """One tool or model operation in the agent trace."""

    timestamp: str = ""
    step: int
    tool: str
    args: dict[str, Any]
    result_length: int
    latency_ms: float


def append_trace(path: str | Path, trace: StepTrace) -> None:
    """Append a serialized trace record, creating its parent directory."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not trace.timestamp:
        trace = trace.model_copy(update={"timestamp": datetime.now(UTC).isoformat()})
    with target.open("a", encoding="utf-8") as file:
        file.write(json.dumps(trace.model_dump(mode="json"), ensure_ascii=False) + "\n")
