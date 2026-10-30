"""Mockable DuckDuckGo search and safely fetched page tools."""

from __future__ import annotations

import time
from collections.abc import Callable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from pydantic import BaseModel

from citecheck.security import detect_prompt_injection, validate_url

_SEARCH_CACHE: dict[tuple[str, int], list[dict[str, str]]] = {}
_PAGE_CACHE: dict[str, PageContent] = {}


class PageContent(BaseModel):
    """Readable text and metadata extracted from one web page."""

    url: str
    title: str
    text: str
    injection_flags: list[str]


def search_results(
    query: str,
    max_results: int = 5,
    search_fn: Callable[[str, int], list[dict[str, str]]] | None = None,
) -> list[dict[str, str]]:
    """Search DuckDuckGo, returning cached result dictionaries."""
    cache_key = (query, max_results)
    if cache_key in _SEARCH_CACHE:
        return [dict(result) for result in _SEARCH_CACHE[cache_key]]
    search = search_fn or (lambda text, limit: DDGS().text(text, max_results=limit))
    results = [
        {
            "title": str(item.get("title", "")),
            "url": str(item.get("href", item.get("url", ""))),
            "snippet": str(item.get("body", item.get("snippet", ""))),
        }
        for item in search(query, max_results)
    ]
    _SEARCH_CACHE[cache_key] = results
    return [dict(result) for result in results]


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return readable titles, URLs, and snippets."""
    try:
        results = search_results(query, max_results)
    except Exception as exc:
        return f"Search failed: {exc}"
    if not results:
        return "No results found."
    return "\n\n".join(f"{item['title']}\n{item['url']}\n{item['snippet']}" for item in results)


def read_page_document(
    url: str,
    session: requests.Session | None = None,
    *,
    retries: int = 2,
    timeout: float = 10,
    max_chars: int = 12000,
    url_validator: Callable[[str], str] = validate_url,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> PageContent:
    """Fetch, sanitize, and cache a page while validating each redirect."""
    if url in _PAGE_CACHE:
        return _PAGE_CACHE[url]
    client = session or requests.Session()
    current_url = url
    response: requests.Response | None = None
    for redirect_number in range(6):
        current_url = url_validator(current_url)
        for attempt in range(retries + 1):
            try:
                response = client.get(
                    current_url,
                    timeout=timeout,
                    headers={"User-Agent": "CiteCheck/0.1 (research assistant)"},
                    allow_redirects=False,
                )
                break
            except requests.RequestException:
                if attempt == retries:
                    raise
                sleep_fn(0.25 * (attempt + 1))
        if response is None:
            raise requests.RequestException("No response received")
        if response.is_redirect:
            location = response.headers.get("Location")
            if not location or redirect_number == 5:
                raise requests.RequestException("Too many or invalid redirects")
            current_url = urljoin(current_url, location)
            response = None
            continue
        response.raise_for_status()
        break

    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    text = " ".join(soup.get_text(" ").split())[:max_chars]
    document = PageContent(
        url=current_url,
        title=title,
        text=text,
        injection_flags=detect_prompt_injection(text),
    )
    _PAGE_CACHE[url] = document
    return document


def read_page(url: str) -> str:
    """Read a web page and return its sanitized text for the agent tool."""
    try:
        document = read_page_document(url)
    except Exception as exc:
        return f"Could not fetch page: {exc}"
    warning = (
        " [Possible prompt injection detected in page text.]" if document.injection_flags else ""
    )
    return f"{warning}\n{document.text}".strip()
