"""Two-text comparison with windowing (fake bag-of-words embedder, no network)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import aam.compare as compare
from aam.compare import LENGTH_RATIO, Comparison, compare_texts, text_vector
from aam.embeddings import embed_cached
from aam.text import count_words
from tests.fakes import FakeProvider, bow_vector


def embed(texts):
    return embed_cached(FakeProvider(), texts, None)


def wide_embed(texts):
    """Bag of words with many buckets, so unrelated made-up words rarely collide."""
    return [bow_vector(t, 512) for t in texts]


def sentence(prefix: str, n: int) -> str:
    """A sentence of n made-up words that appear in no other sentence."""
    words = [f"{prefix}{i}" for i in range(n)]
    words[0] = words[0].capitalize()
    return " ".join(words) + "."


def test_length_ratio_is_four():
    assert LENGTH_RATIO == 4


def test_ratio_exactly_four_is_not_windowed():
    short = " ".join(f"kurz{i}" for i in range(10))
    long = " ".join(f"lang{i}" for i in range(40))
    result = compare_texts(short, long, embed, "fake")
    assert isinstance(result, Comparison)
    assert not result.windowed
    assert result.best_window == ""
    assert result.window_words == 0
    assert result.score == result.whole_score
    assert (result.short_words, result.long_words) == (10, 40)


def test_non_windowed_score_is_cosine_of_whole_vectors():
    short = "Das BAFA zahlt Zuschüsse für die Dachdämmung."
    long = "Das BAFA zahlt Zuschüsse von bis zu 20 Prozent für die Dachdämmung."
    a, b = bow_vector(short), bow_vector(long)
    expected = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert compare_texts(short, long, embed, "fake").score == pytest.approx(expected, abs=1e-5)


def test_ratio_above_four_is_windowed():
    short = " ".join(f"kurz{i}" for i in range(10))
    long = " ".join(sentence(f"lang{i}x", 9) for i in range(5))
    result = compare_texts(short, long, embed, "fake")
    assert result.windowed
    assert result.best_window != ""
    assert result.window_words == count_words(result.best_window)


def test_single_window_returns_the_plain_comparison():
    short = " ".join(f"kurz{i}" for i in range(10))
    long = " ".join(f"lang{i}" for i in range(41))  # one unpunctuated line: only one window exists
    result = compare_texts(short, long, embed, "fake")
    assert not result.windowed
    assert result.best_window == ""
    assert result.window_words == 0
    assert result.score == result.whole_score


def test_windows_respect_paragraphs_of_unpunctuated_text():
    lines = [" ".join(f"z{n}w{i}" for i in range(10)) for n in range(30)]
    short = lines[17]
    result = compare_texts(short, "\n".join(lines), wide_embed, "fake")
    assert result.windowed
    assert result.best_window in lines
    assert result.best_window == short
    assert result.score > result.whole_score


def test_fallback_windows_of_text_vector_respect_paragraphs(monkeypatch):
    monkeypatch.setattr(compare, "fits_whole_text", lambda provider_name, text: False)
    lines = [" ".join(f"z{n}w{i}" for i in range(10)) for n in range(30)]
    calls = []

    def recording(texts):
        calls.append(list(texts))
        return wide_embed(texts)

    text_vector("\n".join(lines), recording, "fake")
    assert len(calls[0]) > 1
    assert all(w in lines for w in calls[0])


def test_best_window_contains_the_repeated_sentence():
    short = sentence("treffer", 15)
    fillers = [sentence(f"fuell{i}x", 14) for i in range(6)]
    long = " ".join(fillers[:3] + [short] + fillers[3:])
    result = compare_texts(short, long, wide_embed, "fake")
    assert result.windowed
    assert short in result.best_window
    assert result.score >= result.whole_score
    assert result.score > 0.5


def test_larger_window_wins_when_the_match_spans_two_sentences():
    shared = [f"thema{i}" for i in range(20)]
    short = " ".join(shared)
    # Each matching sentence carries half of the short text plus words of its own, so a
    # window holding only one of them is diluted; the 2x window holds both.
    a = " ".join(shared[:10] + [f"aextra{i}" for i in range(8)]).capitalize() + "."
    b = " ".join(shared[10:] + [f"bextra{i}" for i in range(8)]).capitalize() + "."
    fillers = [sentence(f"fuell{i}x", 18) for i in range(4)]
    long = " ".join(fillers[:3] + [a, b, fillers[3]])
    result = compare_texts(short, long, wide_embed, "fake")
    assert result.windowed
    target = 20  # word count of the short text, inside 15..80
    assert result.window_words > target
    window = result.best_window.lower()
    assert "thema0 " in window and "thema19" in window and "thema10 " in window
    # A window of the plain target size cannot reach this score.
    assert result.score > 0.45


def test_swapped_arguments_give_the_same_result():
    short = sentence("treffer", 15)
    long = " ".join([sentence(f"fuell{i}x", 14) for i in range(3)] + [short])
    straight, swapped = compare_texts(short, long, wide_embed, "fake"), compare_texts(long, short, wide_embed, "fake")
    assert (straight.swapped, swapped.swapped) == (False, True)
    assert replace(swapped, swapped=False) == straight


def test_ratio_is_long_words_over_short_words_and_swapped_tells_the_direction():
    short = " ".join(f"kurz{i}" for i in range(10))
    long = " ".join(f"lang{i}" for i in range(25))
    straight, swapped = compare_texts(short, long, embed, "fake"), compare_texts(long, short, embed, "fake")
    assert (straight.ratio, straight.swapped) == (2.5, False)
    assert (swapped.ratio, swapped.swapped) == (2.5, True)
    assert (swapped.short_words, swapped.long_words) == (10, 25)
    equal = compare_texts(short, short + " x", embed, "fake")
    assert (equal.swapped, round(equal.ratio, 1)) == (False, 1.1)
    assert compare_texts(short, short, embed, "fake").ratio == 1.0


@pytest.mark.parametrize("empty", ["", "   \n "])
def test_empty_text_raises(empty):
    with pytest.raises(ValueError, match="Text"):
        compare_texts(empty, "Irgendein Text.", embed, "fake")
    with pytest.raises(ValueError, match="Text"):
        compare_texts("Irgendein Text.", empty, embed, "fake")


def test_text_vector_embeds_whole_text_when_it_fits():
    calls = []

    def recording(texts):
        calls.append(list(texts))
        return embed(texts)

    text = "Eine kurze Aussage über Dachdämmung."
    vector = text_vector(text, recording, "fake")
    assert calls == [[text]]
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0)


def test_text_vector_falls_back_to_window_mean_above_the_limit(monkeypatch):
    monkeypatch.setattr(compare, "fits_whole_text", lambda provider_name, text: False)
    calls = []

    def recording(texts):
        calls.append(list(texts))
        return embed(texts)

    text = " ".join(sentence(f"teil{i}x", 30) for i in range(8))
    vector = text_vector(text, recording, "fake")
    assert len(calls) == 1
    assert text not in calls[0]
    assert len(calls[0]) >= 2
    assert all(count_words(w) < count_words(text) for w in calls[0])
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0)


def test_text_vector_is_none_when_nothing_could_be_embedded():
    assert text_vector("Ein Text.", lambda texts: [None for _ in texts], "fake") is None


def test_whole_score_of_long_text_uses_window_mean_above_the_limit(monkeypatch):
    short = sentence("treffer", 15)
    long = " ".join([sentence(f"fuell{i}x", 14) for i in range(14)] + [short])
    monkeypatch.setattr(compare, "fits_whole_text", lambda provider_name, text: text == short)
    calls = []

    def recording(texts):
        calls.append(list(texts))
        return wide_embed(texts)

    result = compare_texts(short, long, recording, "fake")
    assert result.windowed
    assert all(long not in call for call in calls)
    assert result.score >= result.whole_score
