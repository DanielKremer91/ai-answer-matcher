"""Content gap between competitor pages and own pages, section by section (spec 2026-10-09, section 6)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Sequence

import numpy as np

from aam.compare import LENGTH_RATIO, Embed, long_windows, text_vectors
from aam.gap_statements import (DEFAULT_STATEMENT_TARGET, GapStatement, SectionVerdict, chunk_sections,
                                section_verdicts, statement_level)
from aam.match import unit_rows
from aam.models import Document, SegmentedAnswer, Statement
from aam.segment import MAX_TARGET, MIN_TARGET, derive_target
from aam.sections import Section, plain_text, section_level, split_sections

SINGLE_SECTION_NOTE = "Die Seite hat nur einen Abschnitt."
WHOLE_PAGE_NOTE = SINGLE_SECTION_NOTE + " Verglichen wird die ganze Seite."  # no headings at all
NO_PAGE_VECTOR_NOTE = "Kein Seitenvektor, Ganztext-Wert nicht verfügbar."
# With at most this many own sections every pair above the length ratio gets windows. Above it, per
# competitor section only the `WINDOW_CANDIDATES` best own sections by whole-vector cosine and the
# `WINDOW_CANDIDATES` best among the pairs above the length ratio do (a long own section that contains the
# competitor's passage has a low whole-vector cosine and would otherwise never be windowed).
WINDOW_ALL_UP_TO = 40
WINDOW_CANDIDATES = 5
METHOD_WHOLE = "Ganztext"
METHOD_WINDOW_COMPETITOR = "Fenster aus Wettbewerber-Abschnitt"  # windows cut from the competitor's longer text
METHOD_WINDOW_OWN = "Fenster aus eigenem Abschnitt"


@dataclass(frozen=True)
class GapRow:
    competitor: Section
    own: Section                # best matching own section (over all own pages)
    score: float                # cosine of the two section vectors, or of the best window when that is higher
    windowed: bool              # the score comes from a window
    best_window: str            # "" when not windowed
    statement: Optional[Statement]
    statement_score: Optional[float]
    rank: int                   # 1 = most similar
    method: str = METHOD_WHOLE  # where the score comes from (METHOD_* constants)
    statement_whole_score: Optional[float] = None  # the whole section against that statement, no windows
    verdict: Optional[SectionVerdict] = None       # statement level: how many of the section's statements you cover


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
    dropped: int = 0            # competitor sections without a vector (missing from `rows`)


@dataclass(frozen=True)
class GapResult:
    pages: list[GapPage]
    own_pages: list[tuple[str, int, int]]  # (url, section level or 0, number of sections)
    answer: Optional[SegmentedAnswer]
    provider: str
    model: str
    date: str
    keyword: str = ""
    statements: list = None          # GapStatement per competitor chunk (statement level)
    statement_target: int = DEFAULT_STATEMENT_TARGET
    covered_delta: float = 0.05
    weak_delta: float = 0.15

    def __post_init__(self):
        if self.statements is None:
            object.__setattr__(self, "statements", [])


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
                keyword: str = "", today: Optional[date] = None, content_factor: float = 1.5,
                statement_target: int = DEFAULT_STATEMENT_TARGET, covered_delta: float = 0.05,
                weak_delta: float = 0.15) -> GapResult:
    """Compare every section of every competitor page with all own sections and keep the best match.

    Costs at most two `embed` calls: one for all section, page and statement vectors, and one for the
    windows of the longer section of the pairs above `LENGTH_RATIO` (see `WINDOW_ALL_UP_TO` for which pairs
    qualify). A windowed pair scores the better of its whole-vector and its best-window cosine.
    """
    comp_sides, own_sides = _prepare(competitors), _prepare(own)
    if not comp_sides:
        raise ValueError("Die Wettbewerber-Seiten enthalten keinen Text.")
    if not own_sides:
        raise ValueError("Der eigene Content enthält keinen Text.")

    statements = list(answer.statements) if answer is not None else []
    all_sides = comp_sides + own_sides
    # statement level: both sides cut into statement-sized chunks inside their sections
    comp_chunks = {side.doc.url: chunk_sections(side.doc.url, side.doc.origin, side.sections, statement_target)
                   for side in comp_sides}
    own_chunks = [c for side in own_sides
                  for c in chunk_sections(side.doc.url, side.doc.origin, side.sections, statement_target)]
    texts = [s.embed_text for side in all_sides for s in side.sections] + [side.text for side in all_sides]
    texts += [st.text for st in statements]
    texts += [c.text for chunks in comp_chunks.values() for c in chunks] + [c.text for c in own_chunks]
    vectors = {t: v for t, v in text_vectors(texts, embed, provider_name).items() if v is not None}

    def section_vec(section: Section) -> Optional[np.ndarray]:
        return vectors.get(section.embed_text)

    own_sections = [(side, s) for side in own_sides for s in side.sections if section_vec(s) is not None]
    if not own_sections:
        raise ValueError("Für den eigenen Content konnte kein Embedding berechnet werden.")

    own_matrix = np.vstack([section_vec(o) for _, o in own_sections])

    def orientation(a: Section, b: Section) -> tuple[Section, Section]:
        return (a, b) if a.words <= b.words else (b, a)  # (short, long)

    def exceeds_ratio(a: Section, b: Section) -> bool:
        short, long = orientation(a, b)
        return long.words > LENGTH_RATIO * short.words

    def candidates(c: Section, whole: np.ndarray) -> list[int]:
        if len(own_sections) <= WINDOW_ALL_UP_TO:
            return list(range(len(own_sections)))
        order = [int(i) for i in np.argsort(-whole, kind="stable")]
        top = order[:WINDOW_CANDIDATES]
        long_pairs = [i for i in order if exceeds_ratio(c, own_sections[i][1])][:WINDOW_CANDIDATES]
        return list(dict.fromkeys(top + long_pairs))

    # Per competitor section: whole-vector cosine against every own section, and the pairs that get windows.
    window_texts: dict[tuple[str, int], list[str]] = {}
    plans: dict[int, tuple[np.ndarray, dict[int, tuple[str, int]]]] = {}
    for side in comp_sides:
        for c in side.sections:
            cv = section_vec(c)
            if cv is None:
                continue
            whole = own_matrix @ cv
            windowed: dict[int, tuple[str, int]] = {}
            for index in candidates(c, whole):
                short, long = orientation(c, own_sections[index][1])
                if long.words <= LENGTH_RATIO * short.words:
                    continue
                key = (long.embed_text, max(MIN_TARGET, min(MAX_TARGET, short.words)))
                if key not in window_texts:
                    windows = long_windows(long.embed_text, key[1])
                    window_texts[key] = windows if len(windows) > 1 else []
                windowed[index] = key
            plans[id(c)] = (whole, windowed)
    # Statement windows: a 15-word statement against a 200-word section is diluted, so every competitor
    # section that is more than LENGTH_RATIO times the chunk target also gets windows of that size; the
    # statement score is the better of whole section and best window. The target is the same as in the
    # answer mode: median statement length times the content factor, clamped.
    statement_windows: dict[str, list[str]] = {}
    if statements:
        statement_target, _ = derive_target([st.words for st in statements], content_factor)
        for side in comp_sides:
            for c in side.sections:
                if c.words > LENGTH_RATIO * statement_target:
                    windows = long_windows(c.embed_text, statement_target)
                    statement_windows[c.embed_text] = windows if len(windows) > 1 else []
    to_embed = list(dict.fromkeys([w for ws in window_texts.values() for w in ws]
                                  + [w for ws in statement_windows.values() for w in ws]))
    window_vectors: dict[str, np.ndarray] = {}
    if to_embed:
        window_vectors = {w: _unit(v) for w, v in zip(to_embed, embed(to_embed)) if v is not None}

    def best_own(c: Section) -> tuple[float, Section, bool, str, str]:
        whole, windowed = plans[id(c)]
        best = None
        for index, (_, o) in enumerate(own_sections):
            score, is_windowed, window, method = float(whole[index]), False, "", METHOD_WHOLE
            key = windowed.get(index)
            if key is not None:
                short, long = orientation(c, o)
                usable = [w for w in window_texts[key] if w in window_vectors]
                if usable:
                    short_vec = section_vec(short)
                    best_window = max(usable, key=lambda w: float(short_vec @ window_vectors[w]))
                    window_score = float(short_vec @ window_vectors[best_window])
                    if window_score > score:  # the better of whole and window counts
                        score, is_windowed, window = window_score, True, best_window
                        method = METHOD_WINDOW_COMPETITOR if long is c else METHOD_WINDOW_OWN
            if best is None or score > best[0]:
                best = (score, o, is_windowed, window, method)
        return best

    statement_vectors = [(st, vectors[st.text]) for st in statements if st.text in vectors]

    gap_statements: list[GapStatement] = []
    for side in comp_sides:
        gap_statements += statement_level(side.doc.url, comp_chunks[side.doc.url], own_chunks, vectors,
                                          covered_delta=covered_delta, weak_delta=weak_delta,
                                          statement_vectors=statement_vectors)
    verdicts = section_verdicts(gap_statements)

    pages: list[GapPage] = []
    for side in comp_sides:
        found = []
        for c in side.sections:
            if section_vec(c) is not None:
                found.append((c, *best_own(c)))
        found.sort(key=lambda item: item[1], reverse=True)
        rows = []
        for rank, (c, score, o, windowed, window, method) in enumerate(found, start=1):
            statement, statement_score, statement_whole = None, None, None
            if statement_vectors:
                cv = section_vec(c)
                window_vecs = [window_vectors[w] for w in statement_windows.get(c.embed_text, ()) if w in window_vectors]

                def statement_fit(v: np.ndarray) -> tuple[float, float]:
                    """(best of whole section and windows, whole section alone)."""
                    whole_fit = float(cv @ v)
                    best_window = max((float(wv @ v) for wv in window_vecs), default=whole_fit)
                    return max(whole_fit, best_window), whole_fit

                statement, (statement_score, statement_whole) = max(
                    ((st, statement_fit(v)) for st, v in statement_vectors), key=lambda item: item[1][0])
                statement_score, statement_whole = min(1.0, statement_score), min(1.0, statement_whole)
            rows.append(GapRow(c, o, min(1.0, score), windowed, window, statement, statement_score, rank, method,
                               statement_whole, verdicts.get((side.doc.url, c.index))))

        page_score, best_url = 0.0, ""
        page_vec = vectors.get(side.text)
        if page_vec is not None:
            for other in own_sides:
                other_vec = vectors.get(other.text)
                if other_vec is not None and (not best_url or float(page_vec @ other_vec) > page_score):
                    page_score, best_url = float(page_vec @ other_vec), other.doc.url
        dropped = len(side.sections) - len(rows)
        if side.level is None:
            notes = [WHOLE_PAGE_NOTE]
        else:
            notes = [SINGLE_SECTION_NOTE] if len(side.sections) == 1 else []
        if dropped:
            notes.append(f"{dropped} Abschnitte konnten nicht eingebettet werden und fehlen." if dropped > 1
                         else "1 Abschnitt konnte nicht eingebettet werden und fehlt.")
        if page_vec is None:
            notes.append(NO_PAGE_VECTOR_NOTE)
        pages.append(GapPage(
            url=side.doc.url, level=side.level, sections=len(side.sections),
            words=sum(s.words for s in side.sections), page_score=min(1.0, page_score),
            best_own_url=best_url, rows=rows, note=" ".join(notes), dropped=dropped))

    return GapResult(
        pages=pages,
        own_pages=[(side.doc.url, side.level or 0, len(side.sections)) for side in own_sides],
        answer=answer, provider=provider_name, model=model,
        date=(today or date.today()).isoformat(), keyword=keyword, statements=gap_statements,
        statement_target=statement_target, covered_delta=covered_delta, weak_delta=weak_delta)
