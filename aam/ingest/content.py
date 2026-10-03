"""Read a table with a URL column and a main-content column (e.g. a Screaming Frog export)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

from aam.ingest.tables import read_table_info
from aam.ingest.urls import url_key
from aam.models import OWN, Document

URL_COLUMNS = ("url", "address", "adresse", "page", "seite")
CONTENT_COLUMNS = ("extract main content 1", "main content", "content", "text", "body text",
                   "seiteninhalt", "inhalt")
_CONTENT_HINTS = ("content", "inhalt", "text")


@dataclass(frozen=True)
class ContentInfo:
    url_column: str
    content_column: str
    rows: int
    documents: int
    empty: int  # documents without text
    duplicates: int  # rows dropped because the URL appeared before


def _exact(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    by_lower = {c.strip().lower(): c for c in reversed(columns)}  # first column wins on ties
    for candidate in candidates:
        if candidate in by_lower:
            return by_lower[candidate]
    return None


def _content_column(columns: Sequence[str], url_column: Optional[str]) -> Optional[str]:
    exact = _exact(columns, CONTENT_COLUMNS)
    if exact:
        return exact
    for column in columns:
        if column != url_column and any(hint in column.lower() for hint in _CONTENT_HINTS):
            return column
    return None


def documents_from_frame(df: pd.DataFrame, *, origin: str = OWN) -> tuple[list[Document], ContentInfo]:
    columns = [str(c) for c in df.columns]
    url_column = _exact(columns, URL_COLUMNS)
    content_column = _content_column(columns, url_column)
    if url_column is None or content_column is None:
        raise ValueError(
            "Spalten nicht gefunden. Erwartet werden eine URL-Spalte (z. B. 'Address' oder 'URL') und eine "
            "Content-Spalte (z. B. 'Extract Main Content 1' oder 'Main Content'). "
            f"Gefunden: {', '.join(columns)}"
        )
    docs: list[Document] = []
    seen: set[str] = set()
    empty = duplicates = 0
    for url, text in zip(df[url_column], df[content_column]):
        url = str(url).strip()
        text = "" if text is None else str(text).strip()
        if not url:
            continue
        key = url_key(url)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        if not text:
            empty += 1
        docs.append(Document(url, text, origin))
    return docs, ContentInfo(url_column, content_column, len(df), len(docs), empty, duplicates)


def read_content_table(source, filename: Optional[str] = None, *, origin: str = OWN) -> tuple[list[Document], ContentInfo]:
    df, _ = read_table_info(source, filename)
    return documents_from_frame(df, origin=origin)
