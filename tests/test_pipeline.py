import io
from datetime import date

import pytest
from pytest import approx

from aam.cache import MemoryCache
from aam.ingest.chunks_file import ChunkFileError, chunks_to_csv, read_chunks_file
from aam.match import QUARTILE_LABELS
from aam.models import OWN, SOURCE, Document, FailedUrl
from aam.pipeline import analyze, combine_documents, prepare, rematch
from tests.fakes import ANSWER, OWN_DOCS, SOURCE_DOCS, FakeProvider, sample_result

KEYWORD = "dachsanierung kosten"
COVERED = "Mit Dämmung steigen die Kosten auf 200 bis 300 Euro pro Quadratmeter."
GAP = "Die KfW fördert die Dachdämmung mit zinsgünstigen Krediten."


def by_text(result):
    return {m.statement.text: m for m in result.match.statements}


def test_prepare_segments_and_chunks_with_the_derived_target():
    prepared = prepare(KEYWORD, ANSWER, own_docs=OWN_DOCS, source_docs=SOURCE_DOCS)
    assert prepared.answer.target_words == 15
    assert (prepared.own_stats.documents, prepared.own_stats.chunks, prepared.own_stats.merged) == (2, 3, 2)
    assert prepared.source_stats.chunks == 2
    assert prepared.prechunked is False and prepared.warnings == () and prepared.failed == ()
    assert prepared.texts_to_embed == 7  # 5 statements + 2 chunks with other text; 3 chunks repeat a statement


def test_target_override_changes_the_chunking():
    prepared = prepare(KEYWORD, ANSWER, own_docs=OWN_DOCS, target_override=60)
    assert prepared.answer.target_words == 60
    assert prepared.own_stats.chunks == 2


def test_analyze_finds_coverage_and_gaps_against_sources():
    result = sample_result()
    covered, gap = by_text(result)[COVERED], by_text(result)[GAP]
    assert covered.status == "abgedeckt"
    assert covered.own.url == "https://me.de/kosten" and covered.own.score == approx(1.0, abs=1e-5)
    assert gap.status == "Lücke" and gap.delta < -0.15
    assert gap.source.url == "https://quelle.de/foerderung"
    own_pages = [p for p in result.match.pages if p.origin == OWN]
    assert own_pages[0].url == "https://me.de/kosten"
    meta = result.meta
    assert (meta.keyword, meta.date, meta.provider, meta.model) == (KEYWORD, "2026-10-03", "fake", "fake-model")
    assert (meta.dimension, meta.target_words, meta.method, meta.prechunked) == (64, 15, "structure", False)
    assert (meta.covered_delta, meta.weak_delta) == (0.05, 0.15)
    assert len(result.statement_vectors) == 5


def test_run_without_sources_uses_quartiles_and_own_pages_only():
    result = sample_result(with_sources=False)
    assert result.match.has_sources is False
    assert result.source_chunks == [] and result.source_stats is None
    assert all(m.status in QUARTILE_LABELS for m in result.match.statements)
    assert all(p.origin == OWN for p in result.match.pages)


def test_all_sources_failed_behaves_like_no_sources_and_lists_them():
    failed = [FailedUrl("https://quelle.de/x", SOURCE, "HTTP 403")]
    prepared = prepare(KEYWORD, ANSWER, own_docs=OWN_DOCS, source_docs=[], failed=failed)
    result = analyze(prepared, FakeProvider(), None)
    assert result.match.has_sources is False
    assert result.meta.failed == tuple(failed)


def test_second_analysis_reuses_cached_embeddings():
    provider, cache = FakeProvider(), MemoryCache()
    prepared = prepare(KEYWORD, ANSWER, own_docs=OWN_DOCS, source_docs=SOURCE_DOCS)
    seen = []
    analyze(prepared, provider, cache, progress=lambda done, total: seen.append((done, total)))
    assert provider.calls == 7 and seen[-1] == (7, 7)
    analyze(prepared, provider, cache, weak_delta=0.3)
    assert provider.calls == 7


def test_rematch_applies_new_thresholds_without_embedding():
    result = sample_result()
    loose = rematch(result, covered_delta=2.0, weak_delta=2.0)
    assert {m.status for m in loose.match.statements} == {"abgedeckt"}
    assert (loose.meta.covered_delta, loose.meta.weak_delta) == (2.0, 2.0)
    assert by_text(result)[GAP].status == "Lücke"  # the original result is untouched


def prechunked(first, **overrides):
    data = chunks_to_csv(first.own_chunks, provider="fake", model="fake-model", target_words=15)
    chunks, target = read_chunks_file(io.BytesIO(data), "c.csv", provider="fake", model="fake-model")
    params = {"own_chunks": chunks, "stored_target": target, "source_docs": SOURCE_DOCS, **overrides}
    return prepare(KEYWORD, ANSWER, **params)


def test_reusing_exported_chunks_gives_the_same_results_and_embeds_less():
    first = sample_result()
    prepared = prechunked(first)
    assert prepared.prechunked is True and prepared.warnings == ()
    assert (prepared.own_stats.documents, prepared.own_stats.chunks) == (2, 3)
    provider = FakeProvider()
    second = analyze(prepared, provider, None)
    assert provider.calls == 5  # only the statements; the source chunks repeat two of them
    assert second.meta.prechunked is True
    for a, b in zip(first.match.statements, second.match.statements):
        assert (a.own.url, a.status) == (b.own.url, b.status)
        assert a.own.score == approx(b.own.score, abs=1e-4)
    for a, b in zip(first.match.pages, second.match.pages):
        assert a.url == b.url and a.page_score == approx(b.page_score, abs=1e-4)


def test_exported_chunks_with_another_dimension_are_rejected():
    prepared = prechunked(sample_result())
    with pytest.raises(ChunkFileError, match="Dimension 64.*Dimension 32"):
        analyze(prepared, FakeProvider(dim=32), None)


def test_stored_target_far_from_the_current_one_warns():
    first = sample_result()
    assert prechunked(first, stored_target=20).warnings == ()
    warnings = prechunked(first, stored_target=40).warnings
    assert len(warnings) == 1 and "Zielgröße 40" in warnings[0] and "15" in warnings[0]


def test_combine_documents_prefers_table_rows():
    table = [Document("https://me.de/a", "frog", OWN)]
    fetched = [Document("http://me.de/a/", "fetched", OWN), Document("https://me.de/b", "b", OWN)]
    assert [d.text for d in combine_documents(table, fetched)] == ["frog", "b"]


@pytest.mark.parametrize("keyword,answer,docs,message", [
    ("  ", ANSWER, OWN_DOCS, "Keyword"),
    (KEYWORD, " \n", OWN_DOCS, "AI-Antwort"),
    (KEYWORD, ANSWER, [], "Kein eigener Content"),
    (KEYWORD, ANSWER, [Document("https://me.de/leer", "  ", OWN)], "Kein eigener Content"),
])
def test_prepare_validates_inputs(keyword, answer, docs, message):
    with pytest.raises(ValueError, match=message):
        prepare(keyword, answer, own_docs=docs)


class FlakyProvider(FakeProvider):
    """Returns no vector for texts that mention the team, like a provider rejecting one input."""

    def embed(self, texts):
        return [None if "Team" in text else vector for text, vector in zip(texts, super().embed(texts))]


def test_chunks_without_embedding_are_dropped_with_a_warning():
    prepared = prepare(KEYWORD, ANSWER, own_docs=OWN_DOCS)
    result = analyze(prepared, FlakyProvider(), None)
    assert {c.url for c in result.own_chunks} == {"https://me.de/kosten"}
    assert any(w.startswith("1 Chunk") for w in result.meta.warnings)


def test_statement_without_embedding_aborts():
    class NoStatements(FakeProvider):
        def embed(self, texts):
            return [None if "KfW" in text else vector for text, vector in zip(texts, super().embed(texts))]

    with pytest.raises(ValueError, match="Aussage der AI-Antwort"):
        analyze(prepare(KEYWORD, ANSWER, own_docs=OWN_DOCS), NoStatements(), None)


OWN_AS_SOURCE = Document("https://me.de/kosten/", "Kosten der Dachsanierung\nMit Dämmung steigen die Kosten.", SOURCE)
OWN_WARNING = "https://me.de/kosten/ gehört zum eigenen Content und wird nicht als Quelle gewertet."


def test_source_that_is_an_own_page_is_dropped_with_a_warning():
    prepared = prepare(KEYWORD, ANSWER, own_docs=OWN_DOCS, source_docs=[OWN_AS_SOURCE, *SOURCE_DOCS])
    assert prepared.warnings == (OWN_WARNING,)
    assert {c.url for c in prepared.source_chunks} == {"https://quelle.de/foerderung"}
    assert prepared.source_stats.documents == 1
    result = analyze(prepared, FakeProvider(), None)
    assert result.match.has_sources is True
    assert OWN_WARNING in result.meta.warnings
    urls = [p.url for p in result.match.pages]
    assert len(urls) == len(set(urls))


def test_only_own_pages_as_sources_behave_like_no_sources():
    prepared = prepare(KEYWORD, ANSWER, own_docs=OWN_DOCS, source_docs=[OWN_AS_SOURCE])
    assert prepared.warnings == (OWN_WARNING,)
    assert prepared.source_chunks == [] and prepared.source_stats is None
    result = analyze(prepared, FakeProvider(), None)
    assert result.match.has_sources is False
    assert result.source_chunks == [] and result.source_stats is None
    assert OWN_WARNING in result.meta.warnings


def test_own_pages_among_sources_are_dropped_with_prechunked_own_content():
    prepared = prechunked(sample_result(), source_docs=[OWN_AS_SOURCE, *SOURCE_DOCS])
    assert prepared.prechunked is True and prepared.warnings == (OWN_WARNING,)
    assert {c.url for c in prepared.source_chunks} == {"https://quelle.de/foerderung"}
    result = analyze(prepared, FakeProvider(), None)
    assert result.match.has_sources is True and OWN_WARNING in result.meta.warnings
    urls = [p.url for p in result.match.pages]
    assert len(urls) == len(set(urls))


def test_only_own_pages_as_sources_with_prechunked_own_content():
    prepared = prechunked(sample_result(), source_docs=[OWN_AS_SOURCE])
    assert prepared.warnings == (OWN_WARNING,)
    assert prepared.source_chunks == [] and prepared.source_stats is None
    result = analyze(prepared, FakeProvider(), None)
    assert result.match.has_sources is False and OWN_WARNING in result.meta.warnings
