"""Topic-level comparison with topic-sized windows.

A topic of the AI answer bundles several statements (60 to 100 words). Chunks are cut to statement size,
so no single chunk can cover a whole topic. For the topic view the pages are therefore cut again into
windows of about the topic's size (times the content factor), and the best window counts.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from aam.chunk import _windows
from aam.compare import Embed
from aam.match import unit_rows, weighted_mean
from aam.models import OWN, SOURCE, Hit, SegmentedAnswer, Topic

TOPIC_MIN_WORDS = 15
TOPIC_MAX_WORDS = 300
TOPIC_WINDOW_PAGES = 10  # own pages that get topic windows, chosen by page vector; sources always


def topic_target(topic: Topic, content_factor: float) -> int:
    words = sum(s.words for s in topic.statements)
    return int(max(TOPIC_MIN_WORDS, min(TOPIC_MAX_WORDS, round(words * content_factor))))


def topic_windows(text: str, target_words: int) -> list[str]:
    """Windows of about `target_words` over the whole page, crossing paragraphs, one sentence of overlap."""
    flat = " ".join(line.strip() for line in text.split("\n") if line.strip())
    return _windows(flat, target_words) or ([flat] if flat else [])


def topic_hits(answer: SegmentedAnswer, statement_vectors: Sequence[np.ndarray], page_texts: dict,
               page_vectors: dict, embed: Embed, content_factor: float) -> dict:
    """Per topic id: (best own Hit, best source Hit or None, window words).

    Pages without text (reused chunk export) are skipped; a topic without any own window keeps no entry,
    so the caller falls back to the chunk-level hits.
    """
    statements = answer.statements
    s_matrix = unit_rows(np.vstack(statement_vectors).astype(np.float32))
    weights = np.array([s.words for s in statements], dtype=np.float32)
    position = {s.id: i for i, s in enumerate(statements)}
    own_pages = [(url, text) for (origin, url), text in page_texts.items() if origin == OWN and text.strip()]
    source_pages = [(url, text) for (origin, url), text in page_texts.items() if origin == SOURCE and text.strip()]
    if not own_pages:
        return {}

    plans = []  # (topic, vector, target, own candidates, windows per url)
    for topic in answer.topics:
        rows = [position[s.id] for s in topic.statements]
        vector = weighted_mean(s_matrix[rows], weights[rows])
        target = topic_target(topic, content_factor)
        candidates = own_pages
        if len(own_pages) > TOPIC_WINDOW_PAGES:
            scored = []
            for url, text in own_pages:
                pv = page_vectors.get((OWN, url))
                scored.append((float(unit_rows(np.asarray(pv, dtype=np.float32)[None, :])[0] @ vector)
                               if pv is not None else -1.0, url, text))
            scored.sort(key=lambda item: -item[0])
            candidates = [(url, text) for _, url, text in scored[:TOPIC_WINDOW_PAGES]]
        windows = {url: topic_windows(text, target) for url, text in candidates + source_pages}
        plans.append((topic, vector, target, {u for u, _ in candidates}, windows))

    texts = list(dict.fromkeys(w for _, _, _, _, windows in plans for ws in windows.values() for w in ws))
    vectors = {}
    if texts:
        vectors = {t: unit_rows(np.asarray(v, dtype=np.float32)[None, :])[0]
                   for t, v in zip(texts, embed(texts)) if v is not None}

    def best(vector: np.ndarray, urls: Sequence[str], windows: dict) -> Optional[Hit]:
        found = None
        for url in urls:
            for w in windows.get(url, ()):
                v = vectors.get(w)
                if v is None:
                    continue
                score = min(1.0, float(vector @ v))
                if found is None or score > found.score:
                    found = Hit(url, w, score)
        return found

    hits = {}
    for topic, vector, target, own_urls, windows in plans:
        own_hit = best(vector, [u for u in windows if u in own_urls], windows)
        if own_hit is None:
            continue
        source_hit = best(vector, [u for u, _ in source_pages], windows) if source_pages else None
        hits[topic.id] = (own_hit, source_hit, target)
    return hits
