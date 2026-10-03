import numpy as np

from aam.embeddings import embed_cached
from aam.cache import MemoryCache
from tests.fakes import FakeProvider


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_fake_provider_is_deterministic_and_rewards_shared_words():
    p = FakeProvider()
    a, b, c = p.embed(["Kosten Dach Euro", "Kosten Dach Preis", "Hund Katze Maus"])
    assert cos(a, b) > cos(a, c)
    assert p.embed(["Kosten Dach Euro"])[0].tolist() == a.tolist()


def test_fake_provider_counts_texts_and_memory_cache_avoids_repeats():
    p = FakeProvider()
    cache = MemoryCache()
    embed_cached(p, ["eins", "zwei"], cache)
    embed_cached(p, ["eins", "drei"], cache)
    assert p.calls == 3
