"""Parse a text field with one URL per line and compare URLs loosely."""

from __future__ import annotations

import re
from typing import Optional
from urllib.parse import unquote, urlsplit, urlunsplit

_SCHEME = re.compile(r"^https?://", re.IGNORECASE)
_ANY_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)
_LEADING = "<>\"'\u201e\u201c,;"  # round brackets are handled separately in _strip_wrappers
_TRAILING = "<>\"'\u201e\u201c,;."
_TRACKING_PARAMS = frozenset({"srsltid", "gclid", "fbclid", "msclkid", "ref"})


def _is_tracking(parameter: str) -> bool:
    name = parameter.split("=", 1)[0].lower()
    return name.startswith("utm_") or name in _TRACKING_PARAMS


def url_key(url: str) -> str:
    """Comparison key: scheme, host case, a trailing slash, percent-encoding of the path and tracking
    parameters (utm_*, srsltid, gclid, ...) do not matter. Never raises."""
    url = url.strip()
    try:
        parts = urlsplit(url if _ANY_SCHEME.match(url) else f"https://{url}")
    except ValueError:
        return url.lower()
    key = parts.netloc.lower() + unquote(parts.path.rstrip("/"))
    query = "&".join(p for p in parts.query.split("&") if p and not _is_tracking(p))
    return f"{key}?{query}" if query else key


def _strip_wrappers(token: str) -> str:
    """Strip quotes, angle brackets and trailing punctuation; round brackets only when they do not belong to the URL.

    A leading "(" goes when it is closed by the trailing ")" (surrounding pair) or when "(" outnumbers ")";
    a trailing ")" goes when ")" outnumbers "(". So `Dach_(Bauwerk)` keeps its bracket.
    """
    while True:
        stripped = token.lstrip(_LEADING).rstrip(_TRAILING)
        if stripped.startswith("(") and (stripped.endswith(")") or stripped.count("(") > stripped.count(")")):
            stripped = stripped[1:]
            if stripped.count(")") > stripped.count("("):
                stripped = stripped[:-1]
        elif stripped.endswith(")") and stripped.count(")") > stripped.count("("):
            stripped = stripped[:-1]
        if stripped == token:
            return token
        token = stripped


def _is_mail(token: str) -> bool:
    return token.lower().startswith("mailto:") or ("@" in token and "/" not in token)


def parse_urls(text: Optional[str]) -> list[str]:
    """URLs in input order, without fragments and duplicates; tokens without a dotted host are dropped."""
    urls: list[str] = []
    seen: set[str] = set()
    for token in (text or "").split():
        token = _strip_wrappers(token)
        if not token or _is_mail(token):
            continue
        if not _SCHEME.match(token):
            token = f"https://{token}"
        try:
            parts = urlsplit(token)
            if "." not in parts.netloc:
                continue
            url = urlunsplit((parts.scheme.lower(), parts.netloc, parts.path, parts.query, ""))
        except ValueError:
            continue
        key = url_key(url)
        if key in seen:
            continue
        seen.add(key)
        urls.append(url)
    return urls
