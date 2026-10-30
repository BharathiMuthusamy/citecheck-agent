"""Validated runtime settings for CiteCheck."""

from __future__ import annotations

import os

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Configuration values with environment-based overrides."""

    model: str = "qwen2.5:3b"
    embed_model: str = "nomic-embed-text"
    max_steps: int = Field(default=6, ge=1, le=30)
    chunk_size: int = Field(default=1000, ge=100, le=10000)
    top_k: int = Field(default=5, ge=1, le=20)
    relevance_threshold: float = Field(default=0.45, ge=0, le=1)
    retrieval_enabled: bool = True
    verifier_enabled: bool = True
    trace_path: str = "traces/agent.jsonl"

    @classmethod
    def from_environment(cls) -> Settings:
        """Build settings from ``CITECHECK_*`` environment variables."""
        values: dict[str, object] = {}
        for field in ("model", "embed_model", "trace_path"):
            value = os.getenv(f"CITECHECK_{field.upper()}")
            if value is not None:
                values[field] = value
        for field in ("max_steps", "chunk_size", "top_k"):
            value = os.getenv(f"CITECHECK_{field.upper()}")
            if value is not None:
                values[field] = int(value)
        relevance_threshold = os.getenv("CITECHECK_RELEVANCE_THRESHOLD")
        if relevance_threshold is not None:
            values["relevance_threshold"] = float(relevance_threshold)
        for field in ("retrieval_enabled", "verifier_enabled"):
            value = os.getenv(f"CITECHECK_{field.upper()}")
            if value is not None:
                values[field] = value.strip().lower() in {"1", "true", "yes", "on"}
        return cls.model_validate(values)
