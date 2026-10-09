"""Compare two texts: whole-text cosine, or best-window cosine when one text is much longer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from aam.chunk import _windows
from aam.embeddings import fits_whole_text
from aam.match import unit_rows, weighted_mean
from aam.segment import MAX_TARGET, MIN_TARGET

Embed = Callable[[list[str]], list[Optional[np.ndarray]]]

# The longer text counts as "much longer" above this many times the words of the shorter one.
LENGTH_RATIO = 4
# Window size used to build the vector of a text that does not fit the model as a whole.
FALLBACK_TARGET = 80


@dataclass(frozen=True)
class Comparison:
    score: float          # whole-text cosine, or the best window cosine when `windowed`
    whole_score: float    # cosine of the short text against the whole long text
    windowed: bool
    best_window: str      # text of the best window ("" when not windowed)
    window_words: int     # word count of the best window (0 when not windowed)
    short_words: int
    long_words: int


def _parts(text: str, provider_name: str) -> tuple[list[str], bool]:
    """Texts to embed for the vector of `text`, and whether that is the whole text."""
    if fits_whole_text(provider_name, text):
        return [text], True
    return _windows(text, FALLBACK_TARGET) or [text], False


def _combine(text: str, whole: bool, parts: list[str], vectors: dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """Vector of `text` from the embedded `parts` (the text itself, or the mean of its windows)."""
    if whole:
        vector = vectors.get(text)
        if vector is None:
            return None
        return unit_rows(vector[None, :])[0]
    kept = [p for p in parts if p in vectors]
    if not kept:
        return None
    matrix = np.vstack([vectors[p] for p in kept]).astype(np.float64)
    weights = np.array([len(p.split()) for p in kept], dtype=np.float64)
    return weighted_mean(unit_rows(matrix), weights)


def _embed_all(texts: list[str], embed: Embed) -> dict[str, np.ndarray]:
    """One `embed` call over the distinct texts; texts without a vector are left out."""
    distinct = list(dict.fromkeys(texts))
    vectors = embed(distinct)
    return {t: np.asarray(v, dtype=np.float64) for t, v in zip(distinct, vectors) if v is not None}


def text_vector(text: str, embed: Embed, provider_name: str) -> Optional[np.ndarray]:
    """Normalised vector of the whole text, or the word-weighted mean of its windows when too long."""
    parts, whole = _parts(text, provider_name)
    return _combine(text, whole, parts, _embed_all(parts, embed))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b)


def compare_texts(short: str, long: str, embed: Embed, provider_name: str) -> Comparison:
    """Compare two texts; the one with fewer words is treated as the short statement."""
    if not short.strip() or not long.strip():
        raise ValueError("Beide Texte müssen ausgefüllt sein.")
    if len(short.split()) > len(long.split()):
        short, long = long, short
    short_words, long_words = len(short.split()), len(long.split())
    windowed = long_words > LENGTH_RATIO * short_words

    short_parts, short_whole = _parts(short, provider_name)
    long_parts, long_whole = _parts(long, provider_name)
    windows: list[str] = []
    if windowed:
        target = max(MIN_TARGET, min(MAX_TARGET, short_words))
        windows = list(dict.fromkeys(_windows(long, target) + _windows(long, 2 * target)))
    vectors = _embed_all(short_parts + long_parts + windows, embed)

    short_vec = _combine(short, short_whole, short_parts, vectors)
    long_vec = _combine(long, long_whole, long_parts, vectors)
    if short_vec is None or long_vec is None:
        raise ValueError("Für einen der Texte konnte kein Embedding berechnet werden.")
    whole_score = _cosine(short_vec, long_vec)
    if not windowed:
        return Comparison(whole_score, whole_score, False, "", 0, short_words, long_words)

    scored = [(_cosine(short_vec, unit_rows(vectors[w][None, :])[0]), w) for w in windows if w in vectors]
    if not scored:
        raise ValueError("Für den längeren Text konnte kein Fenster eingebettet werden.")
    score, best = max(scored, key=lambda item: item[0])
    return Comparison(score, whole_score, True, best, len(best.split()), short_words, long_words)
