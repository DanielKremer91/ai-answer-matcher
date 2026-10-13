"""Data structures shared by all modules. No logic beyond one convenience property."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

OWN = "own"
SOURCE = "source"


@dataclass(frozen=True)
class Document:
    url: str
    text: str
    origin: str  # OWN or SOURCE


@dataclass(frozen=True)
class FailedUrl:
    url: str
    origin: str
    reason: str  # German, shown in the protocol


@dataclass(frozen=True)
class Statement:
    id: int
    topic_id: int
    text: str
    words: int


@dataclass(frozen=True)
class Topic:
    id: int
    label: str
    statements: tuple[Statement, ...]


@dataclass(frozen=True)
class SegmentedAnswer:
    keyword: str
    topics: tuple[Topic, ...]
    target_words: int  # chunk size used for own content and sources
    median_words: float  # median statement length the target was derived from
    method: str  # "structure" or "semantic"
    content_factor: float = 1.0  # content chunk size relative to the statement median

    @property
    def statements(self) -> list[Statement]:
        return [s for topic in self.topics for s in topic.statements]


@dataclass
class Chunk:
    url: str
    origin: str
    index: int
    text: str
    words: int
    vector: Optional[np.ndarray] = None


@dataclass(frozen=True)
class ChunkStats:
    documents: int
    chunks: int
    median_words: float
    p10_words: float
    p90_words: float
    merged: int
    split: int
    empty_urls: tuple[str, ...] = ()
    split_examples: tuple = ()  # (url, words, windows, preview) of split paragraphs, at most MAX_SPLIT_EXAMPLES
    merge_short: bool = True  # settings the chunks were produced with
    split_long: bool = True


@dataclass(frozen=True)
class Hit:
    """Best matching chunk for a statement or topic."""

    url: str
    text: str
    score: float


@dataclass(frozen=True)
class StatementMatch:
    statement: Statement
    topic_label: str
    own: Hit
    source: Optional[Hit]
    delta: Optional[float]  # own score minus source score; None without sources
    status: str  # "" when no status can be given
    own_vs_source: Optional[float] = None  # cosine of the best own and the best source chunk; None without sources


@dataclass(frozen=True)
class TopicMatch:
    topic: Topic
    own: Hit
    source: Optional[Hit]
    delta: Optional[float]
    counts: dict  # status label -> number of statements
    window_words: int = 0  # size of the topic-sized windows the hits come from; 0 = chunk level
    method: str = "Chunks"  # "Fenster" when the hits are topic-sized windows of the pages


@dataclass(frozen=True)
class PageRow:
    rank: int
    url: str
    origin: str
    page_score: float
    best_chunk_score: float
    covered: int
    total: int


@dataclass(frozen=True)
class StatementPage:
    """One statement against one page: the page's best chunk and the page as a whole."""

    statement: Statement
    topic_label: str
    rank: int  # rank of the page for this statement, by best chunk score
    url: str
    origin: str
    best_chunk: Hit  # best chunk of this page for the statement
    page_score: float  # cosine of the statement with the page vector (topical fit of the whole page)


@dataclass(frozen=True)
class MatchResult:
    pages: list
    topics: list
    statements: list
    has_sources: bool
    status_labels: tuple[str, ...]
    statement_pages: list = field(default_factory=list)  # StatementPage rows, grouped by statement


@dataclass(frozen=True)
class RunMeta:
    keyword: str
    date: str  # ISO date of the run
    provider: str
    model: str
    dimension: int
    target_words: int
    median_words: float
    method: str
    covered_delta: float
    weak_delta: float
    failed: tuple[FailedUrl, ...]
    warnings: tuple[str, ...]
    prechunked: bool = False  # own content came from a chunk-embedding export
    source_target_words: int = 0  # chunk size used for the sources; 0 = same as target_words
    whole_pages: int = 0  # pages whose vector is the embedding of the whole text
    mean_pages: int = 0  # pages whose vector is the average of their chunk vectors


@dataclass(frozen=True)
class Result:
    meta: RunMeta
    match: MatchResult
    answer: SegmentedAnswer
    own_chunks: list
    source_chunks: list
    own_stats: ChunkStats
    source_stats: Optional[ChunkStats]
    statement_vectors: list  # kept so thresholds can change without re-embedding
    page_vectors: dict = field(default_factory=dict)  # (origin, url) -> whole-text vector where available
    topic_hits: dict = field(default_factory=dict)  # topic id -> (own Hit, source Hit|None, window_words); kept for rematch
