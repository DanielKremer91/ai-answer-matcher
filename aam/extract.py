"""Fetch pages and extract their main content. Failures are collected, never raised."""

from __future__ import annotations

import re
from typing import Callable, Optional, Sequence

import httpx
import trafilatura

from aam.models import Document, FailedUrl

_TEXT_TYPES = ("html", "xml", "text/plain")
_BOLD = re.compile(r"\*\*|__")
USER_AGENT = "Mozilla/5.0 (compatible; AIAnswerMatcher/1.0)"


def extract_text(html) -> str:
    """Main content as Markdown, one paragraph per line; "" when nothing usable is found.

    Headings arrive as `## Title` lines. Bold markers are removed, empty lines dropped.
    """
    text = trafilatura.extract(html, include_comments=False, include_tables=True, favor_recall=True,
                               output_format="markdown")
    if not text:
        return ""
    lines = (_BOLD.sub("", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def fetch_documents(urls: Sequence[str], origin: str, client: httpx.Client, *, timeout: float = 20.0,
                    progress: Optional[Callable[[int, int], None]] = None) -> tuple[list[Document], list[FailedUrl]]:
    docs: list[Document] = []
    failed: list[FailedUrl] = []
    for position, url in enumerate(urls, start=1):
        try:
            response = client.get(url, timeout=timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT})
        except httpx.TimeoutException:
            failed.append(FailedUrl(url, origin, f"Timeout nach {timeout:.0f} s"))
        except (httpx.HTTPError, httpx.InvalidURL, UnicodeError) as exc:
            failed.append(FailedUrl(url, origin, f"Verbindungsfehler ({type(exc).__name__})"))
        else:
            content_type = response.headers.get("content-type", "")
            if response.status_code >= 400:
                failed.append(FailedUrl(url, origin, f"HTTP {response.status_code}"))
            elif content_type and not any(kind in content_type.lower() for kind in _TEXT_TYPES):
                failed.append(FailedUrl(url, origin, f"kein HTML ({content_type.split(';')[0].strip()})"))
            else:
                try:
                    text = extract_text(response.content)
                except Exception:  # malformed or binary responses must not abort the run
                    text = ""
                if text:
                    docs.append(Document(url, text, origin))
                else:
                    failed.append(FailedUrl(url, origin, "kein Main Content gefunden"))
        if progress:
            progress(position, len(urls))
    return docs, failed
