"""Cut documents into chunks whose size follows the statements of the AI answer."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from aam.models import Chunk, ChunkStats, Document
from aam.text import count_words, normalize_space, split_sentences


def _windows(paragraph: str, target_words: int) -> list[str]:
    """Sentence windows of at most 2 x `target_words` words (a single longer sentence stays whole).

    Sentences are grouped greedily without overlap; every later window is preceded by the last
    sentence of the previous group when that still fits the limit.
    """
    limit = 2 * target_words
    groups: list[list[str]] = []
    current: list[str] = []
    current_words = 0
    for sentence in split_sentences(paragraph):
        words = count_words(sentence)
        if current and current_words + words > limit:
            groups.append(current)
            current, current_words = [], 0
        current.append(sentence)
        current_words += words
        if current_words >= target_words:
            groups.append(current)
            current, current_words = [], 0
    if current:
        groups.append(current)

    def size(group: list[str]) -> int:
        return sum(count_words(s) for s in group)

    if len(groups) >= 2 and size(groups[-1]) < target_words / 2 and size(groups[-2]) + size(groups[-1]) <= limit:
        last = groups.pop()
        groups[-1] = groups[-1] + last

    windows: list[str] = []
    for i, group in enumerate(groups):
        overlap = groups[i - 1][-1] if i else None
        if overlap is not None and count_words(overlap) + size(group) <= limit:
            group = [overlap, *group]
        windows.append(" ".join(group))
    return windows


def chunk_document(doc: Document, target_words: int) -> tuple[list[Chunk], int, int]:
    """Return (chunks, merged paragraphs, split paragraphs) for one document."""
    limit = 2 * target_words
    half = target_words / 2
    pieces: list[tuple[str, bool]] = []  # (text, is a continuation window of a split paragraph)
    split = 0
    for line in doc.text.split("\n"):
        paragraph = normalize_space(line)
        if not paragraph:
            continue
        if count_words(paragraph) > limit:
            windows = _windows(paragraph, target_words)
            if len(windows) > 1:
                split += 1
            pieces.extend((w, i > 0) for i, w in enumerate(windows))
        else:
            pieces.append((paragraph, False))

    texts: list[str] = []
    continuation: list[bool] = []  # flag of the first piece of each text
    merged = 0
    for piece, is_continuation in pieces:
        if (texts and not is_continuation and count_words(texts[-1]) < half
                and count_words(texts[-1]) + count_words(piece) <= limit):
            texts[-1] = f"{texts[-1]} {piece}"
            merged += 1
        else:
            texts.append(piece)
            continuation.append(is_continuation)
    if (len(texts) >= 2 and not continuation[-1] and count_words(texts[-1]) < half
            and count_words(texts[-2]) + count_words(texts[-1]) <= limit):
        tail = texts.pop()
        texts[-1] = f"{texts[-1]} {tail}"
        merged += 1

    chunks = [Chunk(doc.url, doc.origin, i, text, count_words(text)) for i, text in enumerate(texts)]
    return chunks, merged, split


def stats_for(chunks: Sequence[Chunk], *, documents: int, merged: int = 0, split: int = 0,
              empty_urls: Sequence[str] = ()) -> ChunkStats:
    words = [c.words for c in chunks]
    if not words:
        return ChunkStats(documents, 0, 0.0, 0.0, 0.0, merged, split, tuple(empty_urls))
    p10, median, p90 = (float(v) for v in np.percentile(words, [10, 50, 90]))
    return ChunkStats(documents, len(words), median, p10, p90, merged, split, tuple(empty_urls))


def chunk_documents(docs: Sequence[Document], target_words: int) -> tuple[list[Chunk], ChunkStats]:
    chunks: list[Chunk] = []
    merged = split = 0
    empty: list[str] = []
    for doc in docs:
        doc_chunks, doc_merged, doc_split = chunk_document(doc, target_words)
        if not doc_chunks:
            empty.append(doc.url)
        chunks.extend(doc_chunks)
        merged += doc_merged
        split += doc_split
    return chunks, stats_for(chunks, documents=len(docs), merged=merged, split=split, empty_urls=empty)
