"""URL safety and prompt-injection checks for retrieved web content."""

from __future__ import annotations

import ipaddress
import re
import socket
from collections.abc import Callable
from urllib.parse import urlsplit

AddressInfo = list[tuple[int, int, int, str, tuple[str, int]]]
Resolver = Callable[..., AddressInfo]

_INJECTION_PATTERNS = (
    re.compile(r"\bignore (?:all )?(?:previous|prior|above) instructions\b", re.I),
    re.compile(r"\bdisregard (?:all )?(?:previous|prior|above) instructions\b", re.I),
    re.compile(r"\b(reveal|print|repeat|show) (?:the )?(?:system|developer) prompt\b", re.I),
    re.compile(r"\b(system|developer) prompt\s*[:=]", re.I),
    re.compile(r"\b(do not follow|override) (?:the )?(?:user|system) instructions\b", re.I),
)


def validate_url(url: str, resolver: Resolver = socket.getaddrinfo) -> str:
    """Return a safe HTTP(S) URL or raise ``ValueError`` for unsafe targets."""
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise ValueError("Invalid URL") from exc

    if parsed.scheme not in {"http", "https"} or not hostname:
        raise ValueError("Only absolute HTTP(S) URLs are allowed")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URLs containing credentials are not allowed")

    normalized_host = hostname.rstrip(".").lower()
    if normalized_host == "localhost" or normalized_host.endswith(".localhost"):
        raise ValueError("Localhost URLs are not allowed")

    try:
        addresses = [ipaddress.ip_address(normalized_host)]
    except ValueError:
        try:
            records = resolver(normalized_host, port, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise ValueError("URL host could not be resolved") from exc
        addresses = [ipaddress.ip_address(record[4][0].split("%", 1)[0]) for record in records]
        if not addresses:
            raise ValueError("URL host did not resolve to an address") from None

    if any(not address.is_global for address in addresses):
        raise ValueError("Private, loopback, and reserved IP addresses are not allowed")
    return url


def detect_prompt_injection(text: str) -> list[str]:
    """Return matched suspicious instruction patterns in untrusted page text."""
    return [pattern.pattern for pattern in _INJECTION_PATTERNS if pattern.search(text)]
