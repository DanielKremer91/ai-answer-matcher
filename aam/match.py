"""Cosine matching between a segmented AI answer and chunks of own content and cited sources."""

from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence

import numpy as np

from aam.models import (OWN, SOURCE, Chunk, Hit, MatchResult, PageRow, SegmentedAnswer,
                        StatementMatch, StatementPage, TopicMatch)

STATUS_COVERED = "abgedeckt"
STATUS_WEAK = "schwach"
STATUS_GAP = "Lücke"
DELTA_LABELS = (STATUS_COVERED, STATUS_WEAK, STATUS_GAP)
QUARTILE_LABELS = ("stark", "über Durchschnitt", "unter Durchschnitt", "schwach")
MIN_STATEMENTS_FOR_QUARTILES = 4
TOP_OWN_PAGES_PER_STATEMENT = 10  # rows per statement in the statement-by-page view; sources are all kept
_EPS = 1e-6  # float32 cosines must not flip a status at the threshold


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def weighted_mean(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted mean of the rows, L2-normalised."""
    mean = (matrix * weights[:, None]).sum(axis=0) / max(float(weights.sum()), 1e-12)
    return mean / max(float(np.linalg.norm(mean)), 1e-12)


def delta_status(delta: float, covered_delta: float, weak_delta: float) -> str:
    if delta >= -covered_delta - _EPS:
        return STATUS_COVERED
    if delta >= -weak_delta - _EPS:
        return STATUS_WEAK
    return STATUS_GAP


def quartile_statuses(scores: Sequence[float]) -> list[str]:
    """Relative strength inside the own content; no status below four statements."""
    if len(scores) < MIN_STATEMENTS_FOR_QUARTILES:
        return [""] * len(scores)
    p25, p50, p75 = np.percentile(scores, [25, 50, 75])
    out = []
    for score in scores:
        if score >= p75:
            out.append(QUARTILE_LABELS[0])
        elif score >= p50:
            out.append(QUARTILE_LABELS[1])
        elif score >= p25:
            out.append(QUARTILE_LABELS[2])
        else:
            out.append(QUARTILE_LABELS[3])
    return out


def _matrix(chunks: Sequence[Chunk]) -> np.ndarray:
    return unit_rows(np.vstack([c.vector for c in chunks]).astype(np.float32))


def _hit(chunks: Sequence[Chunk], index: int, score: float) -> Hit:
    chunk = chunks[int(index)]
    return Hit(chunk.url, chunk.text, float(score))


def _words(chunks: Sequence[Chunk], indices: Sequence[int]) -> np.ndarray:
    return np.array([chunks[i].words for i in indices], dtype=np.float32)


def match(answer: SegmentedAnswer, statement_vectors: Sequence[np.ndarray], chunks: Sequence[Chunk], *,
          covered_delta: float = 0.05, weak_delta: float = 0.15,
          page_vectors: Optional[dict] = None) -> MatchResult:
    """`page_vectors` maps (origin, url) to the embedding of the whole page; pages without one use the
    word-weighted average of their chunk vectors."""
    page_vectors = page_vectors or {}
    statements = answer.statements
    s_matrix = unit_rows(np.vstack(statement_vectors).astype(np.float32))
    s_weights = np.array([s.words for s in statements], dtype=np.float32)
    answer_vector = weighted_mean(s_matrix, s_weights)

    own = [c for c in chunks if c.origin == OWN and c.vector is not None]
    sources = [c for c in chunks if c.origin == SOURCE and c.vector is not None]
    if not own:
        raise ValueError("Kein eigener Content mit Embedding vorhanden.")
    has_sources = bool(sources)

    own_matrix = _matrix(own)
    sim_own = s_matrix @ own_matrix.T
    own_best = sim_own.max(axis=1)
    own_arg = sim_own.argmax(axis=1)
    source_matrix = sim_source = source_best = None
    if has_sources:
        source_matrix = _matrix(sources)
        sim_source = s_matrix @ source_matrix.T
        source_best = sim_source.max(axis=1)
        statuses = [delta_status(float(own_best[i] - source_best[i]), covered_delta, weak_delta)
                    for i in range(len(statements))]
    else:
        statuses = quartile_statuses([float(v) for v in own_best])

    labels = {topic.id: topic.label for topic in answer.topics}
    statement_matches = []
    for i, statement in enumerate(statements):
        own_hit = _hit(own, own_arg[i], own_best[i])
        source_hit: Optional[Hit] = None
        delta: Optional[float] = None
        own_vs_source: Optional[float] = None
        if has_sources:
            j = int(sim_source[i].argmax())
            source_hit = _hit(sources, j, sim_source[i, j])
            delta = own_hit.score - source_hit.score
            own_vs_source = float(own_matrix[int(own_arg[i])] @ source_matrix[j])
        statement_matches.append(StatementMatch(statement, labels[statement.topic_id], own_hit, source_hit,
                                                delta, statuses[i], own_vs_source))

    position = {statement.id: i for i, statement in enumerate(statements)}
    topic_matches = []
    for topic in answer.topics:
        rows = [position[s.id] for s in topic.statements]
        topic_vector = weighted_mean(s_matrix[rows], s_weights[rows])
        own_scores = own_matrix @ topic_vector
        own_hit = _hit(own, own_scores.argmax(), own_scores.max())
        source_hit = None
        delta = None
        if has_sources:
            source_scores = source_matrix @ topic_vector
            source_hit = _hit(sources, source_scores.argmax(), source_scores.max())
            delta = own_hit.score - source_hit.score
        counts = dict(Counter(statuses[i] for i in rows if statuses[i]))
        topic_matches.append(TopicMatch(topic, own_hit, source_hit, delta, counts))

    per_statement_pages: list[list[tuple]] = [[] for _ in statements]  # filled by page_rows

    def page_rows(group: Sequence[Chunk], matrix: np.ndarray, sim: np.ndarray, origin: str) -> list[tuple]:
        by_url: dict[str, list[int]] = {}
        for i, chunk in enumerate(group):
            by_url.setdefault(chunk.url, []).append(i)
        rows = []
        for url, indices in by_url.items():
            whole = page_vectors.get((origin, url))
            if whole is not None:
                page_vector = unit_rows(np.asarray(whole, dtype=np.float32)[None, :])[0]
            else:
                page_vector = weighted_mean(matrix[indices], _words(group, indices))
            per_statement = sim[:, indices].max(axis=1)
            if has_sources:
                covered = sum(delta_status(float(per_statement[i] - source_best[i]), covered_delta, weak_delta)
                              == STATUS_COVERED for i in range(len(statements)))
            else:
                mine = set(indices)
                covered = sum(int(own_arg[i]) in mine for i in range(len(statements)))
            rows.append((url, origin, float(page_vector @ answer_vector), float(per_statement.max()), int(covered)))
            best_in_page = sim[:, indices].argmax(axis=1)
            page_fit = s_matrix @ page_vector
            for i in range(len(statements)):
                hit = _hit(group, indices[int(best_in_page[i])], per_statement[i])
                per_statement_pages[i].append((url, origin, hit, float(page_fit[i])))
        return rows

    rows = page_rows(own, own_matrix, sim_own, OWN)
    if has_sources:
        rows += page_rows(sources, source_matrix, sim_source, SOURCE)
    rows.sort(key=lambda row: -row[2])
    pages = [PageRow(rank, url, origin, score, best, covered, len(statements))
             for rank, (url, origin, score, best, covered) in enumerate(rows, start=1)]

    statement_pages = []
    for i, statement in enumerate(statements):
        by_score = sorted(per_statement_pages[i], key=lambda row: -row[2].score)
        own_kept = 0
        rank = 0
        for url, origin, hit, page_fit in by_score:
            if origin == OWN:
                if own_kept == TOP_OWN_PAGES_PER_STATEMENT:
                    continue
                own_kept += 1
            rank += 1
            statement_pages.append(StatementPage(statement, labels[statement.topic_id], rank, url, origin,
                                                 hit, page_fit))

    return MatchResult(pages, topic_matches, statement_matches, has_sources,
                       DELTA_LABELS if has_sources else QUARTILE_LABELS, statement_pages)
