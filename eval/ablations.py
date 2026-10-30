"""Predefined configuration grid for retrieval/verifier comparisons."""

from __future__ import annotations

CONFIGURATIONS = [
    {"name": "baseline", "retrieval_enabled": False, "verifier_enabled": False},
    {"name": "retrieval", "retrieval_enabled": True, "verifier_enabled": False},
    {"name": "verifier", "retrieval_enabled": False, "verifier_enabled": True},
    {"name": "retrieval_verifier", "retrieval_enabled": True, "verifier_enabled": True},
]
