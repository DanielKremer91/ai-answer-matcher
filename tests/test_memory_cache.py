from aam.cache import MemoryCache


def test_memory_cache_roundtrip_namespaces_and_clear():
    cache = MemoryCache()
    assert cache.get("embeddings", "k") is None
    cache.set("embeddings", "k", {"v": [1.0]})
    assert cache.get("embeddings", "k") == {"v": [1.0]}
    assert cache.get("other", "k") is None
    assert cache.clear() == 1
    assert cache.get("embeddings", "k") is None
