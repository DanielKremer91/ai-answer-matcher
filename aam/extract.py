"""Fetch pages and extract their main content. Failures are collected, never raised."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import httpx
import trafilatura

from aam.models import Document, FailedUrl

USER_AGENT = "Mozilla/5.0 (compatible; AIAnswerMatcher/1.0)"


def extract_text(html) -> str:
    """Main content with one paragraph per line; "" when nothing usable is found."""
    text = trafilatura.extract(html, include_comments=False, include_tables=True, favor_recall=True)
    return text.strip() if text else ""


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
            if response.status_code >= 400:
                failed.append(FailedUrl(url, origin, f"HTTP {response.status_code}"))
            else:
                text = extract_text(response.content)
                if text:
                    docs.append(Document(url, text, origin))
                else:
                    failed.append(FailedUrl(url, origin, "kein Main Content gefunden"))
        if progress:
            progress(position, len(urls))
    return docs, failed
