import httpx
import numpy as np
import pytest
import respx

from aam.embeddings import DEFAULT_EMBED_MODELS, PROVIDERS, make_provider
from aam.embeddings.base import EmbeddingError, EmbeddingProvider

OPENAI_URL = "https://api.openai.com/v1/embeddings"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents"
OLLAMA_URL = "http://localhost:11434/api/embed"


def test_constants():
    assert PROVIDERS == ("openai", "gemini", "ollama")
    assert set(DEFAULT_EMBED_MODELS) == set(PROVIDERS)
    assert DEFAULT_EMBED_MODELS["ollama"] == "bge-m3"  # multilingual, needs no task prefix


def test_unknown_provider():
    with pytest.raises(ValueError):
        make_provider("cohere", "x")


@respx.mock
def test_openai_embeds_in_order_and_sends_auth():
    route = respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json={
        "data": [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}]}))
    p = make_provider("openai", "text-embedding-3-small", api_key="sk-test")
    vecs = p.embed(["a", "b"])
    assert vecs[0].tolist() == [1.0, 0.0] and vecs[1].tolist() == [0.0, 1.0]
    assert vecs[0].dtype == np.float32
    req = route.calls.last.request
    assert req.headers["Authorization"] == "Bearer sk-test"
    assert b'"model": "text-embedding-3-small"' in req.content or b'"model":"text-embedding-3-small"' in req.content


@respx.mock
def test_gemini_embeds_and_sends_key_as_header():
    route = respx.post(GEMINI_URL).mock(return_value=httpx.Response(200, json={
        "embeddings": [{"values": [0.1, 0.2, 0.3]}]}))
    p = make_provider("gemini", "gemini-embedding-001", api_key="g-test")
    vecs = p.embed(["hallo"])
    assert vecs[0].shape == (3,)
    req = route.calls.last.request
    assert req.headers["x-goog-api-key"] == "g-test"
    assert "key" not in req.url.params


@respx.mock
def test_ollama_embeds_with_custom_base_url():
    respx.post("http://ollama.local:11434/api/embed").mock(return_value=httpx.Response(200, json={
        "embeddings": [[0.5, 0.5]]}))
    p = make_provider("ollama", "nomic-embed-text", base_url="http://ollama.local:11434")
    assert p.embed(["x"])[0].tolist() == [0.5, 0.5]


@respx.mock
def test_probe_dimension():
    respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.0] * 1536}]}))
    p = make_provider("openai", "text-embedding-3-small", api_key="k")
    assert p.probe_dimension() == 1536


@respx.mock
def test_400_is_splittable_and_single_failure_becomes_none():
    calls = []

    def handler(request):
        import json
        inputs = json.loads(request.content)["input"]
        calls.append(len(inputs))
        if "BAD" in inputs:
            return httpx.Response(400, json={"error": "too long"})
        return httpx.Response(200, json={"data": [{"index": i, "embedding": [float(len(t))]} for i, t in enumerate(inputs)]})

    respx.post(OPENAI_URL).mock(side_effect=handler)
    p = make_provider("openai", "m", api_key="k")
    p.batch_size = 2
    vecs = p.embed(["a", "BAD", "ccc"])
    assert calls == [2, 1, 1, 1]  # batch of two, split into singles, then the last batch
    assert vecs[0].tolist() == [1.0]
    assert vecs[1] is None
    assert vecs[2].tolist() == [3.0]


@respx.mock
def test_401_during_embed_raises_immediately_without_split():
    route = respx.post(OPENAI_URL).mock(return_value=httpx.Response(401, json={"error": "bad key"}))
    p = make_provider("openai", "m", api_key="wrong")
    with pytest.raises(EmbeddingError):
        p.embed(["a", "b"])
    assert route.call_count == 1


@respx.mock
def test_single_text_auth_error_raises_not_none():
    respx.post(OPENAI_URL).mock(return_value=httpx.Response(403, json={"error": "forbidden"}))
    p = make_provider("openai", "m", api_key="k")
    with pytest.raises(EmbeddingError):
        p.embed(["x"])


@respx.mock
def test_404_is_fatal_without_retry():
    route = respx.post(OPENAI_URL).mock(return_value=httpx.Response(404, json={"error": "nope"}))
    p = make_provider("openai", "m", api_key="k")
    slept = []
    p.sleeper = slept.append
    with pytest.raises(EmbeddingError) as info:
        p.embed(["a", "b"])
    assert info.value.fatal and info.value.status == 404
    assert route.call_count == 1
    assert slept == []


@pytest.mark.parametrize("status", [429, 500, 503])
@respx.mock
def test_429_and_5xx_are_fatal_after_three_retries(status):
    route = respx.post(OPENAI_URL).mock(return_value=httpx.Response(status, json={"error": "nope"}))
    p = make_provider("openai", "m", api_key="k")
    slept = []
    p.sleeper = slept.append
    with pytest.raises(EmbeddingError) as info:
        p.embed(["a", "b"])
    assert info.value.fatal and info.value.status == status
    assert route.call_count == 4
    assert slept == [2, 4, 8]


@respx.mock
def test_429_then_200_succeeds_after_one_pause():
    route = respx.post(OPENAI_URL).mock(side_effect=[
        httpx.Response(429, json={"error": "slow down"}),
        httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]}),
    ])
    p = make_provider("openai", "m", api_key="k")
    slept = []
    p.sleeper = slept.append
    assert p.embed(["a"])[0].tolist() == [1.0]
    assert route.call_count == 2
    assert slept == [2]


@pytest.mark.parametrize("header, expected", [("5", 5.0), ("120", 30.0), ("Wed, 21 Oct 2026 07:28:00 GMT", 2)])
@respx.mock
def test_retry_after_header_is_honoured_and_capped(header, expected):
    respx.post(OPENAI_URL).mock(side_effect=[
        httpx.Response(429, headers={"Retry-After": header}, json={"error": "slow down"}),
        httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]}),
    ])
    p = make_provider("openai", "m", api_key="k")
    slept = []
    p.sleeper = slept.append
    p.embed(["a"])
    assert slept == [expected]


@respx.mock
def test_timeout_is_retried_then_succeeds():
    route = respx.post(OPENAI_URL).mock(side_effect=[
        httpx.ReadTimeout("slow"),
        httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]}),
    ])
    p = make_provider("openai", "m", api_key="k")
    slept = []
    p.sleeper = slept.append
    assert p.embed(["a"])[0].tolist() == [1.0]
    assert route.call_count == 2 and slept == [2]


@respx.mock
def test_probe_dimension_retries_on_503():
    respx.post(OPENAI_URL).mock(side_effect=[
        httpx.Response(503),
        httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.0] * 8}]}),
    ])
    p = make_provider("openai", "m", api_key="k")
    p.sleeper = lambda s: None
    assert p.probe_dimension() == 8


@respx.mock
def test_gemini_400_api_key_invalid_is_fatal():
    route = respx.post(GEMINI_URL).mock(return_value=httpx.Response(400, json={"error": {
        "code": 400, "message": "API key not valid. Please pass a valid API key.", "status": "INVALID_ARGUMENT"}}))
    p = make_provider("gemini", "gemini-embedding-001", api_key="bad")
    with pytest.raises(EmbeddingError):
        p.embed(["a", "b"])
    assert route.call_count == 1


@respx.mock
def test_malformed_response_is_fatal():
    route = respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json={"foo": 1}))
    p = make_provider("openai", "m", api_key="k")
    with pytest.raises(EmbeddingError, match="malformed"):
        p.embed(["a", "b"])
    assert route.call_count == 1


@respx.mock
def test_length_mismatch_is_fatal():
    respx.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json={"embeddings": [[0.5, 0.5]]}))
    p = make_provider("ollama", "nomic-embed-text")
    with pytest.raises(EmbeddingError, match="length mismatch"):
        p.embed(["a", "b"])


@respx.mock
def test_transport_error_is_fatal_after_retries():
    route = respx.post(OPENAI_URL).mock(side_effect=httpx.ConnectError("refused"))
    p = make_provider("openai", "m", api_key="k")
    slept = []
    p.sleeper = slept.append
    with pytest.raises(EmbeddingError) as info:
        p.embed(["a", "b"])
    assert info.value.fatal
    assert route.call_count == 4
    assert slept == [2, 4, 8]


@respx.mock
def test_auth_error_raises_embedding_error_immediately():
    respx.post(OPENAI_URL).mock(return_value=httpx.Response(401, json={"error": "bad key"}))
    p = make_provider("openai", "m", api_key="wrong")
    with pytest.raises(EmbeddingError, match="401"):
        p.probe_dimension()


def test_base_is_abstract():
    with pytest.raises(TypeError):
        EmbeddingProvider("m")  # type: ignore[abstract]


from aam.cache import JsonCache
from aam.embeddings import embed_cached


@respx.mock
def test_embed_cached_only_requests_misses_and_writes_back(tmp_path):
    import json

    seen = []

    def handler(request):
        inputs = json.loads(request.content)["input"]
        seen.append(inputs)
        return httpx.Response(200, json={"data": [{"index": i, "embedding": [float(len(t))]} for i, t in enumerate(inputs)]})

    respx.post(OPENAI_URL).mock(side_effect=handler)
    cache = JsonCache(tmp_path)
    p = make_provider("openai", "m", api_key="k")
    first = embed_cached(p, ["aa", "bbb"], cache)
    assert [v.tolist() for v in first] == [[2.0], [3.0]]
    second = embed_cached(p, ["aa", "cccc"], cache)
    assert [v.tolist() for v in second] == [[2.0], [4.0]]
    assert seen == [["aa", "bbb"], ["cccc"]]


@respx.mock
def test_embed_cached_key_includes_model(tmp_path):
    respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]}))
    cache = JsonCache(tmp_path)
    embed_cached(make_provider("openai", "m1", api_key="k"), ["x"], cache)
    assert cache.get("embeddings", "openai|m1|x") == {"v": [1.0]}
    assert cache.get("embeddings", "openai|m2|x") is None


def test_embed_cached_without_cache_calls_provider():
    class Fake(EmbeddingProvider):
        name = "fake"

        def _embed_batch(self, texts):
            return [[1.0] for _ in texts]

    out = embed_cached(Fake("m"), ["a", "b"], None)
    assert len(out) == 2 and out[0].tolist() == [1.0]


@pytest.mark.parametrize("base_url", ["http://[bad]", "http://localhost:abc", "http://[::1", "http://xn--a.com"])
@respx.mock
def test_ollama_malformed_url_raises_fatal_embedding_error(base_url):
    p = make_provider("ollama", "nomic-embed-text", base_url=base_url)
    with pytest.raises(EmbeddingError) as info:
        p.embed(["x"])
    assert info.value.fatal
    with pytest.raises(EmbeddingError):
        p.probe_dimension()


class RecordingProvider(EmbeddingProvider):
    """Fake provider: records every batch, optionally fails fatally on a given call."""

    name = "fake"
    batch_size = 2

    def __init__(self, fail_on_call=None):
        super().__init__("m")
        self.batches = []
        self.fail_on_call = fail_on_call

    def _embed_batch(self, texts):
        self.batches.append(list(texts))
        if self.fail_on_call == len(self.batches):
            raise EmbeddingError("fake: HTTP 401", status=401, fatal=True)
        return [[float(len(t))] for t in texts]


def test_embed_cached_embeds_each_distinct_text_once():
    p = RecordingProvider()
    out = embed_cached(p, ["a", "bb", "a", "bb", "ccc"], None)
    assert p.batches == [["a", "bb"], ["ccc"]]
    assert [v.tolist() for v in out] == [[1.0], [2.0], [1.0], [2.0], [3.0]]


def test_embed_cached_fatal_error_keeps_earlier_batches_cached(tmp_path):
    cache = JsonCache(tmp_path)
    p = RecordingProvider(fail_on_call=2)
    with pytest.raises(EmbeddingError):
        embed_cached(p, ["a", "bb", "ccc", "dddd"], cache)
    assert cache.get("embeddings", "fake|m|a") == {"v": [1.0]}
    assert cache.get("embeddings", "fake|m|bb") is not None
    assert cache.get("embeddings", "fake|m|ccc") is None

    retry = RecordingProvider()
    out = embed_cached(retry, ["a", "bb", "ccc", "dddd"], cache)
    assert retry.batches == [["ccc", "dddd"]]
    assert [v.tolist() for v in out] == [[1.0], [2.0], [3.0], [4.0]]


def test_embed_cached_reports_progress_per_batch():
    seen = []
    embed_cached(RecordingProvider(), ["a", "bb", "ccc", "a"], None, progress=lambda done, total: seen.append((done, total)))
    assert seen == [(2, 3), (3, 3)]

@respx.mock
def test_gemini_requests_semantic_similarity_task_type():
    import json

    route = respx.post(GEMINI_URL).mock(return_value=httpx.Response(200, json={"embeddings": [{"values": [1.0, 0.0]}]}))
    make_provider("gemini", "gemini-embedding-001", api_key="g").embed(["a"])
    body = json.loads(route.calls[0].request.content)
    assert body["requests"][0]["taskType"] == "SEMANTIC_SIMILARITY"


def test_fits_whole_text_uses_provider_word_limits():
    from aam.embeddings import WHOLE_TEXT_WORD_LIMITS, fits_whole_text

    assert fits_whole_text("openai", "ein kurzer Text") is True
    assert fits_whole_text("gemini", " ".join(["wort"] * (WHOLE_TEXT_WORD_LIMITS["gemini"] + 1))) is False
    assert fits_whole_text("unbekannt", " ".join(["wort"] * 100)) is True


@respx.mock
def test_ollama_refuses_silent_truncation():
    import json

    route = respx.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json={"embeddings": [[1.0]]}))
    make_provider("ollama", "bge-m3").embed(["a"])
    assert json.loads(route.calls[0].request.content)["truncate"] is False
