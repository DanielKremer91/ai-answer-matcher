"""Fetch pages and extract their main content. Failures are collected, never raised."""

from __future__ import annotations

import re
from typing import Callable, Optional, Sequence

import httpx
import trafilatura

from aam.dom_extract import extract_dom
from aam.models import Document, FailedUrl
from aam.text import is_separator_line

_TEXT_TYPES = ("html", "xml", "text/plain")
_ESCAPABLE = "\\*_`~#[]().!|<>+-{}"
_ESCAPE = re.compile(r"\\([" + re.escape(_ESCAPABLE) + r"])")
_PLACEHOLDER = {chr(0xE000 + i): char for i, char in enumerate(_ESCAPABLE)}
_PLACEHOLDER_OF = {char: key for key, char in _PLACEHOLDER.items()}
_EMPHASIS = (
    re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*"),
    re.compile(r"(?<!\w)__(?=\S)(.+?)(?<=\S)__(?!\w)"),
    re.compile(r"(?<![\w*])\*(?=[^\s*])(.+?)(?<=[^\s*])\*(?![\w*])"),
    re.compile(r"(?<!\w)_(?=[^\s_])(.+?)(?<=[^\s_])_(?!\w)"),
)
_SUP_SUB = re.compile(r"</?su[bp]>")
_LIST_MARKER = re.compile(r"^[-*]\s+")
USER_AGENT = "Mozilla/5.0 (compatible; AIAnswerMatcher/1.0)"


def _clean_inline(line: str) -> str:
    """Remove inline Markdown residue (emphasis, code, strike, escapes, sup/sub, list marker) from one line.

    Heading markers `#` and table pipes stay: the sections need them.
    """
    line = _ESCAPE.sub(lambda m: _PLACEHOLDER_OF[m.group(1)], line)  # shield escaped characters
    line = _SUP_SUB.sub("", line).replace("~~", "").replace("`", "")
    for pattern in _EMPHASIS:
        line = pattern.sub(r"\1", line)
    line = _LIST_MARKER.sub("", line.strip())
    return "".join(_PLACEHOLDER.get(char, char) for char in line).strip()


def extract_text(html) -> str:
    """Main content, one block per line, headings as `## Title`; "" when nothing usable is found.

    First choice is the DOM walk (aam.dom_extract): it keeps collapsed accordions and every heading
    marker. When it finds no usable container (very short pages, unusual markup), trafilatura's
    Markdown output is used instead, with inline residue (emphasis, code, escapes, list markers)
    removed and table separator rows (`|---|---|`) dropped.
    """
    try:
        text = extract_dom(html)
    except Exception:  # malformed markup: fall through to trafilatura
        text = ""
    if text:
        return text
    text = trafilatura.extract(html, include_comments=False, include_tables=True, favor_recall=True,
                               output_format="markdown")
    if not text:
        return ""
    lines = (_clean_inline(line) for line in text.split("\n"))
    return "\n".join(line for line in lines if line and not is_separator_line(line))


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
