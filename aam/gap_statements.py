"""Statement level of the content-gap mode (spec 2026-10-09, section 6a).

The section level says whether you have a section on the competitor's topic. This level says which of
the competitor's statements you make nowhere. Both pages are cut into statement-sized chunks inside
their sections; every competitor chunk gets its best own chunk and a verdict that needs no fixed,
model-dependent threshold. Two references come from the run itself:

* neighbours: how close the competitor's own non-overlapping chunks of the same section are to the
  statement ("same topic, different statement"). Your best chunk has to beat them to count as the
  same statement.
* specificity: how close your best chunk is to the competitor's OTHER statements. A generic chunk
  ("Welpen brauchen Ruhe") that sits near everything is not a counterpart to anything.

delta = min(score - neighbours, score - specificity), then the stage-1 status rule (covered / weak / gap).
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from aam.chunk import chunk_documents
from aam.match import STATUS_COVERED, STATUS_GAP, STATUS_WEAK, delta_status
from aam.models import Chunk, Document, Statement
from aam.sections import Section
from aam.text import split_sentences

DEFAULT_STATEMENT_TARGET = 30
VERDICT_TOPIC_GAP = "Themenlücke"
VERDICT_DETAIL_GAPS = "Detaillücken"
VERDICT_TOPIC_ONLY = "nur beim Thema"
VERDICT_COVERED = "abgedeckt"


@dataclass(frozen=True)
class SectionChunk:
    section: Section
    chunk: Chunk

    @property
    def text(self) -> str:
        return self.chunk.text


@dataclass(frozen=True)
class GapStatement:
    competitor_url: str
    section: Section                 # competitor section the statement belongs to
    text: str
    words: int
    own_url: str
    own_section: Section
    own_text: str
    score: float                     # cosine to the best own chunk
    ref_neighbours: Optional[float]  # best non-overlapping sibling chunk of the same section; None without siblings
    ref_specificity: float           # the own chunk against the competitor's other chunks
    delta: float
    status: str
    statement: Optional[Statement] = None       # nearest statement of the AI answer, when given
    statement_score: Optional[float] = None
    priority: Optional[float] = None            # statement_score - score


@dataclass(frozen=True)
class SectionVerdict:
    total: int
    covered: int
    weak: int
    gaps: int
    verdict: str


def chunk_sections(url: str, origin: str, sections: Sequence[Section], target_words: int) -> list[SectionChunk]:
    out: list[SectionChunk] = []
    for section in sections:
        chunks, _ = chunk_documents([Document(url, section.text, origin)], target_words)
        out += [SectionChunk(section, c) for c in chunks]
    return out


def _sentences(text: str) -> frozenset[str]:
    return frozenset(s.strip().lower() for s in split_sentences(text) if s.strip())


def verdict(statuses: Sequence[str]) -> SectionVerdict:
    counts = Counter(statuses)
    n, gaps, weak = len(statuses), counts[STATUS_GAP], counts[STATUS_WEAK]
    if n == 0:
        label = ""
    elif gaps * 2 > n:
        label = VERDICT_TOPIC_GAP
    elif gaps:
        label = VERDICT_DETAIL_GAPS
    elif weak * 2 > n:
        label = VERDICT_TOPIC_ONLY
    else:
        label = VERDICT_COVERED
    return SectionVerdict(n, counts[STATUS_COVERED], weak, gaps, label)


def statement_level(competitor_url: str, comp_chunks: Sequence[SectionChunk], own_chunks: Sequence[SectionChunk],
                    vectors: dict, *, covered_delta: float, weak_delta: float,
                    statement_vectors: Sequence[tuple[Statement, np.ndarray]] = ()) -> list[GapStatement]:
    """One GapStatement per competitor chunk that has a vector. `vectors` maps text -> unit vector."""
    comp = [c for c in comp_chunks if c.text in vectors]
    own = [o for o in own_chunks if o.text in vectors]
    if not comp or not own:
        return []
    c_matrix = np.vstack([vectors[c.text] for c in comp]).astype(np.float32)
    o_matrix = np.vstack([vectors[o.text] for o in own]).astype(np.float32)
    sim = c_matrix @ o_matrix.T
    self_sim = c_matrix @ c_matrix.T
    sentences = [_sentences(c.text) for c in comp]
    # chunks that share a sentence (window overlap) are no reference for "a different statement"
    distinct = [[k for k in range(len(comp)) if k != i and not (sentences[i] & sentences[k])] for i in range(len(comp))]

    neighbours: list[Optional[float]] = []
    for i, c in enumerate(comp):
        siblings = [k for k in distinct[i] if comp[k].section.index == c.section.index]
        neighbours.append(max((float(self_sim[i, k]) for k in siblings), default=None))
    known = [r for r in neighbours if r is not None]
    fallback = statistics.median(known) if known else None  # a section with one chunk borrows the page's level

    s_matrix = (np.vstack([v for _, v in statement_vectors]).astype(np.float32) if statement_vectors else None)
    out: list[GapStatement] = []
    for i, c in enumerate(comp):
        j = int(sim[i].argmax())
        score = float(sim[i, j])
        ref_n = neighbours[i] if neighbours[i] is not None else fallback
        others = distinct[i]
        ref_s = max((float(sim[k, j]) for k in others), default=0.0)
        deltas = [score - ref_s] + ([score - ref_n] if ref_n is not None else [])
        delta = min(deltas)
        statement = statement_score = priority = None
        if s_matrix is not None:
            fits = s_matrix @ c_matrix[i]
            k = int(fits.argmax())
            statement, statement_score = statement_vectors[k][0], min(1.0, float(fits[k]))
            priority = round(statement_score - score, 4)
        out.append(GapStatement(
            competitor_url, c.section, c.text, c.chunk.words, own[j].chunk.url, own[j].section, own[j].text,
            min(1.0, score), ref_n, ref_s, delta, delta_status(delta, covered_delta, weak_delta),
            statement, statement_score, priority))
    return out


def section_verdicts(statements: Sequence[GapStatement]) -> dict[tuple[str, int], SectionVerdict]:
    """(competitor url, section index) -> verdict over that section's statements."""
    by_section: dict[tuple[str, int], list[str]] = {}
    for s in statements:
        by_section.setdefault((s.competitor_url, s.section.index), []).append(s.status)
    return {key: verdict(statuses) for key, statuses in by_section.items()}
