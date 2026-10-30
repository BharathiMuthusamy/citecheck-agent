"""Structured answer and evidence models."""

from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl


class Citation(BaseModel):
    """A source passage used to support a claim."""

    url: HttpUrl
    title: str = ""
    passage: str = Field(min_length=1)
    supports_claim: bool | None = None
    verifier_reason: str = ""


class Claim(BaseModel):
    """A factual statement and the evidence cited for it."""

    text: str = Field(min_length=1)
    citations: list[Citation] = Field(default_factory=list)
    supported: bool | None = None
    verifier_reason: str = ""


class Point(BaseModel):
    """One answer bullet with source references and verifier output."""

    text: str = Field(min_length=1)
    source_ids: list[int] = Field(min_length=1)
    verified: bool | None = None
    verifier_reason: str = ""


class Section(BaseModel):
    """A titled group of answer points."""

    heading: str = Field(min_length=1)
    points: list[Point] = Field(min_length=1)


class Source(BaseModel):
    """A fetched source and the exact passage used as evidence."""

    id: int = Field(ge=1)
    title: str
    url: str
    quote: str = Field(min_length=1)


class Answer(BaseModel):
    """Structured answer whose points reference fetched source entries."""

    summary: str = Field(min_length=1)
    sections: list[Section] = Field(default_factory=list, max_length=4)
    sources: list[Source] = Field(default_factory=list)
