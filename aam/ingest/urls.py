"""Parse a text field with one URL per line and compare URLs loosely."""

from __future__ import annotations

import re
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

_SCHEME = re.compile(r"^https?://", re.IGNORECASE)


def url_key(url: str) -> str:
    """Comparison key: scheme, host case and a trailing slash do not matter."""
    parts = urlsplit(url.strip())
    key = parts.netloc.lower() + parts.path.rstrip("/")
    return f"{key}?{parts.query}" if parts.query else key


def parse_urls(text: Optional[str]) -> list[str]:
    """URLs in input order, without fragments and duplicates; tokens without a dotted host are dropped."""
    urls: list[str] = []
    seen: set[str] = set()
    for token in (text or "").split():
        token = token.strip(",;")
        if not token:
            continue
        if not _SCHEME.match(token):
            token = f"https://{token}"
        parts = urlsplit(token)
        if "." not in parts.netloc:
            continue
        url = urlunsplit((parts.scheme.lower(), parts.netloc, parts.path, parts.query, ""))
        key = url_key(url)
        if key in seen:
            continue
        seen.add(key)
        urls.append(url)
    return urls
