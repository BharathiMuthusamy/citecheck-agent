"""Pure rendering helpers for source-linked answer points."""

from __future__ import annotations

import html
from collections.abc import Mapping

from citecheck.schemas import Point, Source


def render_point_html(point: Point, sources: Mapping[int, Source]) -> str:
    """Render a point with verifier status and links to matching source entries."""
    citation_links = " ".join(
        f'<a class="cc-citation" href="#source-{source_id}">[{source_id}]</a>'
        for source_id in point.source_ids
        if source_id in sources
    )
    status = "Verified" if point.verified is True else "Not supported"
    status_class = "cc-verified" if point.verified is True else "cc-unverified"
    return (
        f'<div class="cc-point"><span class="cc-point-status {status_class}">'
        f"{status}</span> {html.escape(point.text)} {citation_links}</div>"
    )