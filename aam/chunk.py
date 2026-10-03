"""Cut documents into chunks whose size follows the statements of the AI answer."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from aam.models import Chunk, ChunkStats, Document
from aam.text import count_words, normalize_space, split_sentences


def _windows(paragraph: str, target_words: int) -> list[str]:
    """Sentence windows of about `target_words` words; each starts with the previous window's last sentence."""
    limit = 2 * target_words
    out: list[str] = []
    current: list[str] = []
    fresh = 0  # sentences in `current` that are not part of an emitted window yet
    for sentence in split_sentences(paragraph):
        if count_words(sentence) > limit:
            if fresh:
                out.append(" ".join(current))
            out.append(sentence)  # cannot be cut any further
            current, fresh = [], 0
            continue
        current.append(sentence)
        fresh += 1
        if sum(count_words(s) for s in current) >= target_words:
            out.append(" ".join(current))
            current, fresh = [sentence], 0
    if fresh:
        out.append(" ".join(current))
    return out


def chunk_document(doc: Document, target_words: int) -> tuple[list[Chunk], int, int]:
    """Return (chunks, merged paragraphs, split paragraphs) for one document."""
    limit = 2 * target_words
    half = target_words / 2
    pieces: list[str] = []
    split = 0
    for line in doc.text.split("\n"):
        paragraph = normalize_space(line)
        if not paragraph:
            continue
        if count_words(paragraph) > limit:
            windows = _windows(paragraph, target_words)
            if len(windows) > 1:
                split += 1
            pieces.extend(windows)
        else:
            pieces.append(paragraph)

    texts: list[str] = []
    merged = 0
    for piece in pieces:
        if texts and count_words(texts[-1]) < half and count_words(texts[-1]) + count_words(piece) <= limit:
            texts[-1] = f"{texts[-1]} {piece}"
            merged += 1
        else:
            texts.append(piece)
    if len(texts) >= 2 and count_words(texts[-1]) < half and count_words(texts[-2]) + count_words(texts[-1]) <= limit:
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
