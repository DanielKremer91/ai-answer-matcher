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


def line_windows(text: str, target_words: int) -> list[str]:
    """Sentence windows per non-empty line, so a window never crosses a paragraph."""
    windows: list[str] = []
    for line in text.split("\n"):
        if line.strip():
            windows.extend(_windows(line, target_words) or [line.strip()])
    return windows


def long_windows(long_text: str, short_words: int) -> list[str]:
    """Distinct windows of the longer text in both sizes (the target follows the shorter text)."""
    target = max(MIN_TARGET, min(MAX_TARGET, short_words))
    return list(dict.fromkeys(line_windows(long_text, target) + line_windows(long_text, 2 * target)))


def _parts(text: str, provider_name: str) -> tuple[list[str], bool]:
    """Texts to embed for the vector of `text`, and whether that is the whole text."""
    if fits_whole_text(provider_name, text):
        return [text], True
    return line_windows(text, FALLBACK_TARGET) or [text], False


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


def text_vectors(texts: list[str], embed: Embed, provider_name: str) -> dict[str, Optional[np.ndarray]]:
    """Normalised vectors of several texts through ONE `embed` call (see `text_vector`)."""
    distinct = list(dict.fromkeys(texts))
    plans = {t: _parts(t, provider_name) for t in distinct}
    vectors = _embed_all([p for parts, _ in plans.values() for p in parts], embed)
    return {t: _combine(t, whole, parts, vectors) for t, (parts, whole) in plans.items()}


def text_vector(text: str, embed: Embed, provider_name: str) -> Optional[np.ndarray]:
    """Normalised vector of the whole text, or the word-weighted mean of its windows when too long."""
    return text_vectors([text], embed, provider_name)[text]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b)


def compare_texts(short: str, long: str, embed: Embed, provider_name: str) -> Comparison:
    """Compare two texts; the one with fewer words is treated as the short statement."""
    if not short.strip() or not long.strip():
        raise ValueError("Beide Texte müssen ausgefüllt sein.")
    if len(short.split()) > len(long.split()):
        short, long = long, short
    short_words, long_words = len(short.split()), len(long.split())
    # Only a long text that really yields several windows is compared window by window.
    windows = long_windows(long, short_words) if long_words > LENGTH_RATIO * short_words else []
    windowed = len(windows) > 1
    if not windowed:
        windows = []

    short_parts, short_whole = _parts(short, provider_name)
    long_parts, long_whole = _parts(long, provider_name)
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
