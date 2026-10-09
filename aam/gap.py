"""Content gap between competitor pages and own pages, section by section (spec 2026-10-09, section 6)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Sequence

import numpy as np

from aam.compare import LENGTH_RATIO, Embed, long_windows, text_vectors
from aam.match import unit_rows
from aam.models import Document, SegmentedAnswer, Statement
from aam.sections import Section, plain_text, section_level, split_sections

SINGLE_SECTION_NOTE = "Die Seite hat nur einen Abschnitt, verglichen wird die ganze Seite."


@dataclass(frozen=True)
class GapRow:
    competitor: Section
    own: Section                # best matching own section (over all own pages)
    score: float                # cosine of the two section vectors, or of the best window
    windowed: bool
    best_window: str            # "" when not windowed
    statement: Optional[Statement]
    statement_score: Optional[float]
    rank: int                   # 1 = most similar


@dataclass(frozen=True)
class GapPage:
    url: str
    level: Optional[int]        # section heading level; None when the page has a single section
    sections: int
    words: int
    page_score: float           # whole page against the best own page
    best_own_url: str
    rows: list[GapRow]          # sorted by descending score
    note: str


@dataclass(frozen=True)
class GapResult:
    pages: list[GapPage]
    own_pages: list[tuple[str, int, int]]  # (url, section level or 0, number of sections)
    answer: Optional[SegmentedAnswer]
    provider: str
    model: str
    date: str
    keyword: str = ""


@dataclass
class _Side:
    doc: Document
    sections: list[Section]
    level: Optional[int]
    text: str  # plain text of the whole page


def _prepare(docs: Sequence[Document]) -> list[_Side]:
    sides = []
    for doc in docs:
        sections = split_sections(doc)
        if sections:
            sides.append(_Side(doc, sections, section_level(doc.text), plain_text(doc)))
    return sides


def _unit(vector: np.ndarray) -> np.ndarray:
    return unit_rows(np.asarray(vector, dtype=np.float64)[None, :])[0]


def analyze_gap(competitors: Sequence[Document], own: Sequence[Document], embed: Embed,
                provider_name: str, model: str, *, answer: Optional[SegmentedAnswer] = None,
                keyword: str = "", today: Optional[date] = None) -> GapResult:
    """Compare every section of every competitor page with all own sections and keep the best match.

    Costs at most two `embed` calls: one for all section, page and statement vectors, and one for the
    windows of the longer section of every pair whose length ratio exceeds `LENGTH_RATIO`.
    """
    comp_sides, own_sides = _prepare(competitors), _prepare(own)
    if not comp_sides:
        raise ValueError("Die Wettbewerber-Seiten enthalten keinen Text.")
    if not own_sides:
        raise ValueError("Der eigene Content enthält keinen Text.")

    statements = list(answer.statements) if answer is not None else []
    all_sides = comp_sides + own_sides
    texts = [s.text for side in all_sides for s in side.sections] + [side.text for side in all_sides]
    texts += [st.text for st in statements]
    vectors = {t: v for t, v in text_vectors(texts, embed, provider_name).items() if v is not None}

    def section_vec(section: Section) -> Optional[np.ndarray]:
        return vectors.get(section.text)

    own_sections = [(side, s) for side in own_sides for s in side.sections if section_vec(s) is not None]
    if not own_sections:
        raise ValueError("Für den eigenen Content konnte kein Embedding berechnet werden.")

    # Pairs whose longer section needs windows; all windows are embedded together.
    pairs = [(c, o) for side in comp_sides for c in side.sections if section_vec(c) is not None
             for _, o in own_sections]

    def orientation(a: Section, b: Section) -> tuple[Section, Section]:
        return (a, b) if a.words <= b.words else (b, a)  # (short, long)

    window_texts: dict[tuple[str, int], list[str]] = {}
    for c, o in pairs:
        short, long = orientation(c, o)
        if long.words > LENGTH_RATIO * short.words and (long.text, short.words) not in window_texts:
            windows = long_windows(long.text, short.words)
            window_texts[(long.text, short.words)] = windows if len(windows) > 1 else []
    to_embed = list(dict.fromkeys(w for ws in window_texts.values() for w in ws))
    window_vectors: dict[str, np.ndarray] = {}
    if to_embed:
        window_vectors = {w: _unit(v) for w, v in zip(to_embed, embed(to_embed)) if v is not None}

    def score_pair(c: Section, o: Section) -> tuple[float, bool, str]:
        short, long = orientation(c, o)
        short_vec = section_vec(short)
        windows = [w for w in window_texts.get((long.text, short.words), []) if w in window_vectors]
        if windows:
            best = max(windows, key=lambda w: float(short_vec @ window_vectors[w]))
            return float(short_vec @ window_vectors[best]), True, best
        return float(short_vec @ section_vec(long)), False, ""

    statement_vectors = [(st, vectors[st.text]) for st in statements if st.text in vectors]

    pages: list[GapPage] = []
    for side in comp_sides:
        found: list[tuple[float, Section, Section, bool, str]] = []
        for c in side.sections:
            if section_vec(c) is None:
                continue
            best = None
            for _, o in own_sections:
                score, windowed, window = score_pair(c, o)
                if best is None or score > best[0]:
                    best = (score, c, o, windowed, window)
            found.append(best)
        found.sort(key=lambda item: item[0], reverse=True)
        rows = []
        for rank, (score, c, o, windowed, window) in enumerate(found, start=1):
            statement, statement_score = None, None
            if statement_vectors:
                cv = section_vec(c)
                statement, statement_score = max(((st, float(cv @ v)) for st, v in statement_vectors),
                                                 key=lambda item: item[1])
            rows.append(GapRow(c, o, score, windowed, window, statement, statement_score, rank))

        page_score, best_url = 0.0, ""
        page_vec = vectors.get(side.text)
        if page_vec is not None:
            for other in own_sides:
                other_vec = vectors.get(other.text)
                if other_vec is not None and (not best_url or float(page_vec @ other_vec) > page_score):
                    page_score, best_url = float(page_vec @ other_vec), other.doc.url
        pages.append(GapPage(
            url=side.doc.url, level=side.level, sections=len(side.sections),
            words=sum(s.words for s in side.sections), page_score=page_score, best_own_url=best_url,
            rows=rows, note=SINGLE_SECTION_NOTE if side.level is None else ""))

    return GapResult(
        pages=pages,
        own_pages=[(side.doc.url, side.level or 0, len(side.sections)) for side in own_sides],
        answer=answer, provider=provider_name, model=model,
        date=(today or date.today()).isoformat(), keyword=keyword)
