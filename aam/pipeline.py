"""One run in two phases: prepare (segment and chunk, free) and analyze (embed and match)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import date
from typing import Callable, Optional, Sequence

import numpy as np

from aam.chunk import chunk_documents, stats_for
from aam.embeddings import EmbeddingProvider, embed_cached, fits_whole_text
from aam.ingest.chunks_file import check_dimension
from aam.ingest.urls import url_key
from aam.match import match
from aam.models import OWN, SOURCE, Chunk, ChunkStats, Document, FailedUrl, Result, RunMeta, SegmentedAnswer
from aam.sections import plain_text
from aam.segment import Embed, segment_answer

MAX_TARGET_DEVIATION = 0.5


@dataclass
class Prepared:
    answer: SegmentedAnswer
    own_chunks: list[Chunk]
    source_chunks: list[Chunk]
    own_stats: ChunkStats
    source_stats: Optional[ChunkStats]
    failed: tuple[FailedUrl, ...]
    warnings: tuple[str, ...]
    prechunked: bool  # own chunks carry vectors from an uploaded export
    source_target_words: int  # chunk size actually used for the sources
    page_texts: dict = dataclasses.field(default_factory=dict)  # (origin, url) -> full text of the page

    @property
    def texts_to_embed(self) -> int:
        """Distinct texts the next analysis sends to the provider at most."""
        texts = {s.text for s in self.answer.statements}
        texts.update(c.text for c in self._chunks_to_embed())
        texts.update(self.page_texts.values())
        return len(texts)

    def _chunks_to_embed(self) -> list[Chunk]:
        """Chunks without a stored vector; uploaded own chunks bring theirs."""
        return self.source_chunks if self.prechunked else self.own_chunks + self.source_chunks


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
            target_override: Optional[int] = None, embed: Optional[Embed] = None,
            merge_short: bool = True, split_long: bool = True) -> Prepared:
    if not keyword.strip():
        raise ValueError("Bitte ein Keyword oder einen Prompt eintragen.")
    if not answer_text.strip():
        raise ValueError("Bitte die AI-Antwort einfügen.")
    answer = segment_answer(keyword, answer_text, target_override=target_override, embed=embed)
    warnings: list[str] = []
    source_target = answer.target_words
    if own_chunks is not None:
        chunks = list(own_chunks)
        own_stats = stats_for(chunks, documents=len({c.url for c in chunks}))
        target = answer.target_words
        if stored_target:
            source_target = stored_target  # sources and reused chunks share one granularity
        if stored_target and abs(stored_target - target) / target > MAX_TARGET_DEVIATION:
            warnings.append(
                f"Die hochgeladenen Chunks wurden mit Zielgröße {stored_target} Wörter geschnitten, "
                f"die aktuelle AI-Antwort ergibt {target}. Die Scores sind dadurch nur eingeschränkt vergleichbar.")
    else:
        chunks, own_stats = chunk_documents(own_docs, answer.target_words, merge_short=merge_short,
                                            split_long=split_long)
    if not chunks:
        raise ValueError("Kein eigener Content mit Text vorhanden. Bitte Export oder URL-Liste prüfen.")
    own_keys = {url_key(d.url) for d in own_docs} if own_chunks is None else {url_key(c.url) for c in chunks}
    benchmarks = []
    warned: set[str] = set()
    for doc in source_docs:
        key = url_key(doc.url)
        if key in own_keys:
            if key not in warned:
                warned.add(key)
                warnings.append(f"{doc.url} gehört zum eigenen Content und wird nicht als Quelle gewertet.")
        else:
            benchmarks.append(doc)
    source_chunks: list[Chunk] = []
    source_stats = None
    if benchmarks:
        source_chunks, source_stats = chunk_documents(benchmarks, source_target, merge_short=merge_short,
                                                      split_long=split_long)
    page_texts = {(SOURCE, d.url): plain_text(d) for d in benchmarks if d.text.strip()}
    if own_chunks is None:
        page_texts.update({(OWN, d.url): plain_text(d) for d in own_docs if d.text.strip()})
    return Prepared(answer, chunks, source_chunks, own_stats, source_stats, tuple(failed), tuple(warnings),
                    own_chunks is not None, source_target, page_texts)


class _StagedEmbedder:
    """Embeds texts in several steps but reports one continuous progress over the whole run."""

    def __init__(self, provider: EmbeddingProvider, cache, progress: Optional[Callable[[int, int], None]],
                 all_texts: Sequence[str]):
        self._provider, self._cache, self._progress = provider, cache, progress
        self._total = len(set(all_texts))
        self._known: dict[str, Optional[np.ndarray]] = {}
        self._done = 0
        self._reported = 0

    def _report(self, done: int) -> None:
        if self._progress and done > self._reported:
            self._reported = done
            self._progress(done, self._total)

    def embed(self, texts: Sequence[str]) -> list[Optional[np.ndarray]]:
        fresh = [t for t in dict.fromkeys(texts) if t not in self._known]
        base = self._done

        def report(done: int, total: int) -> None:  # embed_cached counts only cache misses
            self._report(base + round(done * len(fresh) / total))

        if fresh:
            self._known.update(zip(fresh, embed_cached(self._provider, fresh, self._cache, progress=report)))
        self._done = base + len(fresh)
        self._report(self._done)
        return [self._known[t] for t in texts]


def analyze(prepared: Prepared, provider: EmbeddingProvider, cache, *, covered_delta: float = 0.05,
            weak_delta: float = 0.15, progress: Optional[Callable[[int, int], None]] = None,
            today: Optional[date] = None) -> Result:
    """Embed and match. `prepared` stays untouched, so it can be analysed again with another model."""
    statements = prepared.answer.statements
    statement_texts = [s.text for s in statements]
    to_embed = prepared._chunks_to_embed()
    whole_pages = {key: text for key, text in prepared.page_texts.items() if fits_whole_text(provider.name, text)}
    embedder = _StagedEmbedder(provider, cache, progress,
                               statement_texts + [c.text for c in to_embed] + list(whole_pages.values()))

    def embed_chunks(chunks: list[Chunk]) -> list[Chunk]:
        vectors = embedder.embed([c.text for c in chunks])
        return [dataclasses.replace(c, vector=v) for c, v in zip(chunks, vectors) if v is not None]

    if prepared.prechunked:
        statement_vectors = embedder.embed(statement_texts)
        _require_statement_vectors(statement_vectors)
        dimension = len(statement_vectors[0])
        check_dimension(prepared.own_chunks, dimension)  # before the sources cost any embedding
        own = [dataclasses.replace(c) for c in prepared.own_chunks if c.vector is not None]
        sources = embed_chunks(prepared.source_chunks)
    else:
        n_own = len(prepared.own_chunks)
        vectors = embedder.embed(statement_texts + [c.text for c in to_embed])
        statement_vectors = vectors[:len(statements)]
        _require_statement_vectors(statement_vectors)
        dimension = len(statement_vectors[0])
        chunk_vectors = vectors[len(statements):]
        replaced = [dataclasses.replace(c, vector=v) for c, v in zip(to_embed, chunk_vectors)]
        own = [c for c in replaced[:n_own] if c.vector is not None]
        sources = [c for c in replaced[n_own:] if c.vector is not None]
    warnings = list(prepared.warnings)
    dropped = len(prepared.own_chunks) + len(prepared.source_chunks) - len(own) - len(sources)
    if dropped:
        if dropped == 1:
            warnings.append("1 Chunk konnte nicht eingebettet werden und fehlt in der Auswertung.")
        else:
            warnings.append(f"{dropped} Chunks konnten nicht eingebettet werden und fehlen in der Auswertung.")
    page_vectors = {key: v for key, v in zip(whole_pages, embedder.embed(list(whole_pages.values())))
                    if v is not None}
    pages = {(c.origin, c.url) for c in own + sources}
    matched = match(prepared.answer, statement_vectors, own + sources,
                    covered_delta=covered_delta, weak_delta=weak_delta, page_vectors=page_vectors)
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
        source_target_words=prepared.source_target_words,
        whole_pages=len(pages & set(page_vectors)),
        mean_pages=len(pages - set(page_vectors)),
    )
    return Result(meta, matched, prepared.answer, own, sources, prepared.own_stats, prepared.source_stats,
                  list(statement_vectors), page_vectors)


def _require_statement_vectors(vectors: Sequence[Optional[np.ndarray]]) -> None:
    if any(v is None for v in vectors):
        raise ValueError("Mindestens eine Aussage der AI-Antwort konnte nicht eingebettet werden.")


def rematch(result: Result, *, covered_delta: float, weak_delta: float) -> Result:
    """New thresholds on an existing result; no embedding call."""
    matched = match(result.answer, result.statement_vectors, result.own_chunks + result.source_chunks,
                    covered_delta=covered_delta, weak_delta=weak_delta, page_vectors=result.page_vectors)
    meta = dataclasses.replace(result.meta, covered_delta=covered_delta, weak_delta=weak_delta)
    return dataclasses.replace(result, meta=meta, match=matched)
