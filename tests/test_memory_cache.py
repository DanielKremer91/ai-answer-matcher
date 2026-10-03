import numpy as np

from aam.cache import MemoryCache
from aam.embeddings import embed_cached
from tests.fakes import FakeProvider, bow_vector


def test_memory_cache_roundtrip_namespaces_and_clear():
    cache = MemoryCache()
    assert cache.get("embeddings", "k") is None
    cache.set("embeddings", "k", {"v": [1.0]})
    assert cache.get("embeddings", "k")["v"].tolist() == [1.0]
    assert cache.get("other", "k") is None
    assert cache.clear() == 1
    assert cache.get("embeddings", "k") is None


def test_memory_cache_stores_vectors_as_float32_arrays_and_other_values_unchanged():
    cache = MemoryCache()
    cache.set("embeddings", "k", {"v": [0.1, 0.2, 0.3]})
    stored = cache.get("embeddings", "k")["v"]
    assert isinstance(stored, np.ndarray) and stored.dtype == np.float32
    cache.set("other", "k", {"x": [1, 2]})
    assert cache.get("other", "k") == {"x": [1, 2]}


def test_embed_cached_returns_correct_vectors_on_a_cache_hit():
    provider, cache = FakeProvider(), MemoryCache()
    texts = ["Dach decken", "Dach dämmen"]
    first = embed_cached(provider, texts, cache)
    assert provider.calls == 2
    second = embed_cached(provider, texts, cache)
    assert provider.calls == 2  # served from the cache
    for text, a, b in zip(texts, first, second):
        assert b.dtype == np.float32
        assert np.allclose(b, a, atol=1e-6) and np.allclose(b, bow_vector(text), atol=1e-6)
