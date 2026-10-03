"""One run in two phases: prepare (segment and chunk, free) and analyze (embed and match)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import date
from typing import Callable, Optional, Sequence

from aam.chunk import chunk_documents, stats_for
from aam.embeddings import EmbeddingProvider, embed_cached
from aam.ingest.chunks_file import check_dimension
from aam.ingest.urls import url_key
from aam.match import match
from aam.models import Chunk, ChunkStats, Document, FailedUrl, Result, RunMeta, SegmentedAnswer
from aam.segment import Embed, segment_answer

MAX_TARGET_DEVIATION = 0.5


@dataclass
class Prepared:
    answer: SegmentedAnswer
    own_chunks: list
    source_chunks: list
    own_stats: ChunkStats
    source_stats: Optional[ChunkStats]
    failed: tuple
    warnings: tuple
    prechunked: bool

    @property
    def texts_to_embed(self) -> int:
        """Distinct texts the next analysis sends to the provider at most."""
        texts = {s.text for s in self.answer.statements}
        texts.update(c.text for c in self.own_chunks + self.source_chunks if c.vector is None)
        return len(texts)


def combine_documents(table_docs: Sequence[Document], fetched_docs: Sequence[Document]) -> list[Document]:
    """Documents from an uploaded table win over fetched ones for the same URL."""
    docs = list(table_docs)
    seen = {url_key(d.url) for d in docs}
    for doc in fetched_docs:
        key = url_key(doc.url)
        if key not in seen:
            seen.add(key)
            docs.append(doc)
    return docs


def prepare(keyword: str, answer_text: str, *, own_docs: Sequence[Document] = (),
            source_docs: Sequence[Document] = (), own_chunks: Optional[Sequence[Chunk]] = None,
            stored_target: Optional[int] = None, failed: Sequence[FailedUrl] = (),
            target_override: Optional[int] = None, embed: Optional[Embed] = None) -> Prepared:
    if not keyword.strip():
        raise ValueError("Bitte ein Keyword oder einen Prompt eintragen.")
    if not answer_text.strip():
        raise ValueError("Bitte die AI-Antwort einfügen.")
    answer = segment_answer(keyword, answer_text, target_override=target_override, embed=embed)
    warnings: list[str] = []
    if own_chunks is not None:
        chunks = list(own_chunks)
        own_stats = stats_for(chunks, documents=len({c.url for c in chunks}))
        target = answer.target_words
        if stored_target and abs(stored_target - target) / target > MAX_TARGET_DEVIATION:
            warnings.append(
                f"Die hochgeladenen Chunks wurden mit Zielgröße {stored_target} Wörter geschnitten, "
                f"die aktuelle AI-Antwort ergibt {target}. Die Scores sind dadurch nur eingeschränkt vergleichbar.")
    else:
        chunks, own_stats = chunk_documents(own_docs, answer.target_words)
    if not chunks:
        raise ValueError("Kein eigener Content mit Text vorhanden. Bitte Export oder URL-Liste prüfen.")
    own_keys = {url_key(d.url) for d in own_docs} if own_chunks is None else {url_key(c.url) for c in chunks}
    benchmarks = []
    for doc in source_docs:
        if url_key(doc.url) in own_keys:
            warnings.append(f"{doc.url} gehört zum eigenen Content und wird nicht als Quelle gewertet.")
        else:
            benchmarks.append(doc)
    source_chunks: list[Chunk] = []
    source_stats = None
    if benchmarks:
        source_chunks, source_stats = chunk_documents(benchmarks, answer.target_words)
    return Prepared(answer, chunks, source_chunks, own_stats, source_stats, tuple(failed), tuple(warnings),
                    own_chunks is not None)


def analyze(prepared: Prepared, provider: EmbeddingProvider, cache, *, covered_delta: float = 0.05,
            weak_delta: float = 0.15, progress: Optional[Callable[[int, int], None]] = None,
            today: Optional[date] = None) -> Result:
    statements = prepared.answer.statements
    pending = [c for c in prepared.own_chunks + prepared.source_chunks if c.vector is None]
    texts = [s.text for s in statements] + [c.text for c in pending]
    vectors = embed_cached(provider, texts, cache, progress=progress)
    statement_vectors = vectors[:len(statements)]
    if any(v is None for v in statement_vectors):
        raise ValueError("Mindestens eine Aussage der AI-Antwort konnte nicht eingebettet werden.")
    for chunk, vector in zip(pending, vectors[len(statements):]):
        chunk.vector = vector
    dimension = len(statement_vectors[0])
    if prepared.prechunked:
        check_dimension(prepared.own_chunks, dimension)

    own = [c for c in prepared.own_chunks if c.vector is not None]
    sources = [c for c in prepared.source_chunks if c.vector is not None]
    warnings = list(prepared.warnings)
    dropped = len(prepared.own_chunks) + len(prepared.source_chunks) - len(own) - len(sources)
    if dropped:
        if dropped == 1:
            warnings.append("1 Chunk konnte nicht eingebettet werden und fehlt in der Auswertung.")
        else:
            warnings.append(f"{dropped} Chunks konnten nicht eingebettet werden und fehlen in der Auswertung.")
    matched = match(prepared.answer, statement_vectors, own + sources,
                    covered_delta=covered_delta, weak_delta=weak_delta)
    meta = RunMeta(
        keyword=prepared.answer.keyword,
        date=(today or date.today()).isoformat(),
        provider=provider.name,
        model=provider.model,
        dimension=dimension,
        target_words=prepared.answer.target_words,
        median_words=prepared.answer.median_words,
        method=prepared.answer.method,
        covered_delta=covered_delta,
        weak_delta=weak_delta,
        failed=tuple(prepared.failed),
        warnings=tuple(warnings),
        prechunked=prepared.prechunked,
    )
    return Result(meta, matched, prepared.answer, own, sources, prepared.own_stats, prepared.source_stats,
                  list(statement_vectors))


def rematch(result: Result, *, covered_delta: float, weak_delta: float) -> Result:
    """New thresholds on an existing result; no embedding call."""
    matched = match(result.answer, result.statement_vectors, result.own_chunks + result.source_chunks,
                    covered_delta=covered_delta, weak_delta=weak_delta)
    meta = dataclasses.replace(result.meta, covered_delta=covered_delta, weak_delta=weak_delta)
    return dataclasses.replace(result, meta=meta, match=matched)
