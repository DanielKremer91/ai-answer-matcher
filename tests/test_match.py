import numpy as np
import pytest
from pytest import approx

from aam.match import QUARTILE_LABELS, delta_status, match, quartile_statuses
from aam.models import OWN, SOURCE, Chunk, SegmentedAnswer, Statement, Topic


def vec(*values):
    return np.array(values, dtype=np.float32)


def chunk(url, origin, index, values, words=10):
    return Chunk(url, origin, index, f"{url}#{index}", words, vec(*values))


def answer(topic_sizes, words=None):
    """Answer with len(topic_sizes) topics; statement i has words[i] words (default 10)."""
    topics, sid = [], 0
    for tid, size in enumerate(topic_sizes):
        statements = []
        for _ in range(size):
            statements.append(Statement(sid, tid, f"s{sid}", (words or {}).get(sid, 10)))
            sid += 1
        topics.append(Topic(tid, f"T{tid}", tuple(statements)))
    return SegmentedAnswer("kw", tuple(topics), 15, 10.0, "structure")


S2 = [vec(1, 0), vec(0, 1)]
CHUNKS = [chunk("/a", OWN, 0, (1, 0)), chunk("/b", OWN, 0, (0.6, 0.8)), chunk("q", SOURCE, 0, (0, 1))]


def test_statement_hits_delta_and_status_with_sources():
    result = match(answer([1, 1]), S2, CHUNKS)
    first, second = result.statements
    assert (first.own.url, first.own.text, first.source.url) == ("/a", "/a#0", "q")
    assert first.own.score == approx(1.0) and first.delta == approx(1.0)
    assert first.status == "abgedeckt"
    assert first.topic_label == "T0"
    assert second.own.url == "/b" and second.own.score == approx(0.8)
    assert second.source.score == approx(1.0) and second.delta == approx(-0.2)
    assert second.status == "Lücke"
    assert result.has_sources is True
    assert result.status_labels == ("abgedeckt", "schwach", "Lücke")


def test_thresholds_are_configurable():
    assert match(answer([1, 1]), S2, CHUNKS, weak_delta=0.25).statements[1].status == "schwach"
    assert match(answer([1, 1]), S2, CHUNKS, covered_delta=0.25, weak_delta=0.3).statements[1].status == "abgedeckt"


@pytest.mark.parametrize("delta,expected", [
    (0.3, "abgedeckt"), (-0.05, "abgedeckt"), (-0.0501, "schwach"),
    (-0.15, "schwach"), (-0.151, "Lücke"),
])
def test_delta_status_boundaries(delta, expected):
    assert delta_status(delta, 0.05, 0.15) == expected


def test_page_ranking_mixes_own_and_source_pages():
    result = match(answer([1, 1]), S2, CHUNKS)
    assert [p.rank for p in result.pages] == [1, 2, 3]
    assert result.pages[0].url == "/b" and result.pages[0].page_score == approx(0.9899, abs=1e-3)
    pages = {p.url: p for p in result.pages}
    assert pages["/a"].page_score == approx(0.7071, abs=1e-3)
    assert pages["/a"].best_chunk_score == approx(1.0)
    assert (pages["/a"].origin, pages["/a"].covered, pages["/a"].total) == (OWN, 1, 2)
    assert pages["/b"].covered == 1
    assert (pages["q"].origin, pages["q"].covered) == (SOURCE, 2)


def test_page_vector_is_weighted_by_chunk_length():
    chunks = [chunk("/a", OWN, 0, (1, 0), words=30), chunk("/a", OWN, 1, (0, 1), words=10)]
    result = match(answer([1]), [vec(1, 0)], chunks)
    assert len(result.pages) == 1
    assert result.pages[0].page_score == approx(0.9487, abs=1e-3)
    assert result.pages[0].best_chunk_score == approx(1.0)


def test_topic_vector_is_weighted_by_statement_length_and_counts_statuses():
    chunks = [chunk("/a", OWN, 0, (1, 0)), chunk("/b", OWN, 0, (0, 1)), chunk("q", SOURCE, 0, (0, 1))]
    result = match(answer([2], words={0: 10, 1: 30}), S2, chunks)
    topic = result.topics[0]
    assert topic.topic.label == "T0"
    assert topic.own.url == "/b" and topic.own.score == approx(0.9487, abs=1e-3)
    assert topic.source.url == "q" and topic.delta == approx(0.0, abs=1e-6)
    assert topic.counts == {"abgedeckt": 2}


def test_without_sources_status_comes_from_quartiles():
    statements = [vec(1, 0), vec(0.9, 0.436), vec(0.6, 0.8), vec(0, 1)]
    result = match(answer([4]), statements, [chunk("/a", OWN, 0, (1, 0)), chunk("/b", OWN, 0, (-1, 0))])
    assert [m.status for m in result.statements] == ["stark", "über Durchschnitt", "unter Durchschnitt", "schwach"]
    assert all(m.source is None and m.delta is None for m in result.statements)
    assert result.has_sources is False and result.status_labels == QUARTILE_LABELS
    assert result.topics[0].source is None and result.topics[0].delta is None
    assert result.topics[0].counts == {"stark": 1, "über Durchschnitt": 1, "unter Durchschnitt": 1, "schwach": 1}
    pages = {p.url: p for p in result.pages}
    assert set(pages) == {"/a", "/b"}
    assert (pages["/a"].covered, pages["/b"].covered, pages["/a"].total) == (4, 0, 4)


def test_without_sources_and_fewer_than_four_statements_there_is_no_status():
    result = match(answer([3]), [vec(1, 0), vec(0, 1), vec(1, 1)], [chunk("/a", OWN, 0, (1, 0))])
    assert [m.status for m in result.statements] == ["", "", ""]
    assert result.topics[0].counts == {}


def test_quartile_statuses():
    assert quartile_statuses([0.9, 0.7, 0.5, 0.3]) == ["stark", "über Durchschnitt", "unter Durchschnitt", "schwach"]
    assert quartile_statuses([0.9, 0.7, 0.5]) == ["", "", ""]


def test_chunks_without_vector_are_ignored_and_missing_own_content_raises():
    no_vector = Chunk("/x", OWN, 0, "x", 5)
    result = match(answer([1]), [vec(1, 0)], [no_vector, chunk("/a", OWN, 0, (1, 0))])
    assert [p.url for p in result.pages] == ["/a"]
    with pytest.raises(ValueError, match="Kein eigener Content"):
        match(answer([1]), [vec(1, 0)], [no_vector, chunk("q", SOURCE, 0, (1, 0))])


def test_vectors_are_normalised_before_comparison():
    result = match(answer([1]), [vec(5, 0)], [chunk("/a", OWN, 0, (3, 0))])
    assert result.statements[0].own.score == approx(1.0)
