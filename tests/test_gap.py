"""Content-gap analysis between competitor pages and own pages (fake embedder, no network)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from aam.gap import GapPage, GapResult, GapRow, analyze_gap
from aam.models import OWN, SOURCE, Document
from aam.segment import segment_answer
from aam.sections import plain_text
from tests.fakes import ANSWER, bow_vector

NOTE = "Die Seite hat nur einen Abschnitt, verglichen wird die ganze Seite."


class Recorder:
    """Embedder over many buckets, so unrelated made-up words rarely collide; counts calls."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return [bow_vector(t, 512) for t in texts]


def words(prefix: str, n: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def sentence(prefix: str, n: int) -> str:
    ws = [f"{prefix}{i}" for i in range(n)]
    ws[0] = ws[0].capitalize()
    return " ".join(ws) + "."


def page(url: str, origin: str, *sections: tuple[str, str]) -> Document:
    text = "\n".join(f"## {title}\n{body}" for title, body in sections)
    return Document(url, "# Titel der Seite\n" + text, origin)


def run(competitors, own, embed=None, **kwargs):
    return analyze_gap(competitors, own, embed or Recorder(), "fake", "fake-model",
                       today=date(2026, 10, 9), **kwargs)


KOSTEN = words("kosten", 12)
FOERDERUNG = words("foerder", 12)
GARANTIE = words("garantie", 12)


def competitor():
    return page("https://wett.de/a", SOURCE, ("Kosten", KOSTEN), ("Förderung", FOERDERUNG), ("Garantie", GARANTIE))


def own_page(url="https://me.de/a"):
    return page(url, OWN, ("Preise", KOSTEN + " " + words("eigen", 3)), ("Zuschüsse", FOERDERUNG + " " + words("eigen", 3)))


def test_rows_are_sorted_descending_and_unmatched_section_ranks_last():
    result = run([competitor()], [own_page()])
    assert isinstance(result, GapResult)
    (gap_page,) = result.pages
    assert isinstance(gap_page, GapPage)
    rows = gap_page.rows
    assert [r.rank for r in rows] == [1, 2, 3]
    assert all(isinstance(r, GapRow) for r in rows)
    scores = [r.score for r in rows]
    assert scores == sorted(scores, reverse=True)
    assert rows[-1].competitor.title == "Garantie"
    assert rows[-1].score < min(scores[:2])
    assert {rows[0].own.title, rows[1].own.title} == {"Preise", "Zuschüsse"}
    assert not any(r.windowed for r in rows)
    assert all(r.best_window == "" for r in rows)


def test_metadata_and_page_numbers():
    result = run([competitor()], [own_page()], keyword="dachsanierung kosten")
    (gap_page,) = result.pages
    assert (result.provider, result.model, result.date) == ("fake", "fake-model", "2026-10-09")
    assert result.keyword == "dachsanierung kosten"
    assert result.answer is None
    assert gap_page.url == "https://wett.de/a"
    assert gap_page.level == 2
    assert gap_page.sections == 3
    assert gap_page.words == 36
    assert gap_page.note == ""
    assert result.own_pages == [("https://me.de/a", 2, 2)]


def test_row_names_the_url_of_the_better_own_page():
    good = own_page("https://me.de/gut")
    poor = page("https://me.de/schwach", OWN, ("Anderes", words("xyz", 12)), ("Mehr", words("abc", 12)))
    result = run([competitor()], [poor, good])
    rows = result.pages[0].rows
    assert rows[0].own.url == "https://me.de/gut"
    assert rows[1].own.url == "https://me.de/gut"
    assert result.pages[0].best_own_url == "https://me.de/gut"
    assert sorted(result.own_pages) == sorted([("https://me.de/gut", 2, 2), ("https://me.de/schwach", 2, 2)])


def test_page_score_is_cosine_of_page_vectors_with_the_best_own_page():
    comp, good = competitor(), own_page("https://me.de/gut")
    poor = page("https://me.de/schwach", OWN, ("Anderes", words("xyz", 12)), ("Mehr", words("abc", 12)))
    result = run([comp], [poor, good])
    a, b = bow_vector(plain_text(comp), 512), bow_vector(plain_text(good), 512)
    expected = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert result.pages[0].page_score == pytest.approx(expected, abs=1e-5)


def test_several_competitor_pages_are_evaluated_one_by_one():
    other = page("https://wett.de/b", SOURCE, ("Eins", words("eins", 12)), ("Zwei", FOERDERUNG))
    result = run([competitor(), other], [own_page()])
    assert [p.url for p in result.pages] == ["https://wett.de/a", "https://wett.de/b"]
    assert [len(p.rows) for p in result.pages] == [3, 2]


def test_windowed_pair_for_much_longer_own_section():
    short = sentence("thema", 20)
    comp = page("https://wett.de/a", SOURCE, ("Thema", short), ("Zweites", words("zwei", 20)))
    fillers = [sentence(f"fuell{i}x", 20) for i in range(9)]
    long_text = "\n".join(fillers[:4] + [short] + fillers[4:])  # 10 paragraphs of 20 words
    own = page("https://me.de/a", OWN, ("Langer Abschnitt", long_text), ("Kurz", words("kurz", 12)))
    result = run([comp], [own])
    row = next(r for r in result.pages[0].rows if r.competitor.title == "Thema")
    assert row.own.title == "Langer Abschnitt"
    assert row.windowed
    assert row.best_window == short
    assert row.score > 0.9


def test_embed_calls_are_few():
    recorder = Recorder()
    run([competitor()], [own_page()], embed=recorder)
    assert len(recorder.calls) == 1  # no windowed pair: section and page vectors in one call

    recorder = Recorder()
    short = sentence("thema", 20)
    comp = page("https://wett.de/a", SOURCE, ("Thema", short), ("Zweites", words("zwei", 20)))
    own = page("https://me.de/a", OWN,
               ("Lang", "\n".join(sentence(f"fuell{i}x", 20) for i in range(10))), ("Kurz", words("kurz", 12)))
    run([comp], [own], embed=recorder, answer=segment_answer("k", ANSWER))
    assert len(recorder.calls) == 2  # vectors, then the windows of the long section


def test_without_answer_no_statement():
    rows = run([competitor()], [own_page()]).pages[0].rows
    assert all(r.statement is None and r.statement_score is None for r in rows)


def test_with_answer_foerderung_section_maps_to_the_kfw_statement():
    answer = segment_answer("dachsanierung kosten", ANSWER)
    comp = page("https://wett.de/a", SOURCE,
                ("Kosten", "Eine Neueindeckung ohne Dämmung liegt bei 100 bis 150 Euro pro Quadratmeter."),
                ("Förderung", "Die KfW fördert die Dachdämmung mit zinsgünstigen Krediten."),
                ("Team", words("team", 12)))
    own = page("https://me.de/a", OWN, ("Preise", words("preis", 12)), ("Mehr", words("mehr", 12)))
    result = run([comp], [own], answer=answer, keyword="dachsanierung kosten")
    assert result.answer is answer
    by_title = {r.competitor.title: r for r in result.pages[0].rows}
    row = by_title["Förderung"]
    assert row.statement is not None
    assert row.statement.text.startswith("Die KfW fördert")
    assert row.statement_score is not None and row.statement_score > 0.8
    assert by_title["Kosten"].statement.text.startswith("Eine Neueindeckung")
    assert by_title["Team"].statement_score < row.statement_score


def test_page_without_headings_gives_one_row_and_a_note():
    comp = Document("https://wett.de/a", "Erster Absatz über Kosten.\nZweiter Absatz über Förderung.", SOURCE)
    result = run([comp], [own_page()])
    (gap_page,) = result.pages
    assert gap_page.level is None
    assert gap_page.sections == 1
    assert gap_page.note == NOTE
    assert len(gap_page.rows) == 1 and gap_page.rows[0].rank == 1


def test_no_own_text_raises():
    with pytest.raises(ValueError, match="eigen"):
        run([competitor()], [Document("https://me.de/leer", "  \n ", OWN)])
    with pytest.raises(ValueError, match="eigen"):
        run([competitor()], [])


def test_no_competitor_text_raises():
    with pytest.raises(ValueError, match="Wettbewerber"):
        run([Document("https://wett.de/leer", "\n", SOURCE)], [own_page()])


def test_pages_without_text_are_skipped():
    empty = Document("https://wett.de/leer", "", SOURCE)
    result = run([empty, competitor()], [own_page()])
    assert [p.url for p in result.pages] == ["https://wett.de/a"]


def test_sections_without_embedding_are_dropped():
    def embed(texts):
        return [None if "garantie0" in t else bow_vector(t, 512) for t in texts]

    result = run([competitor()], [own_page()], embed=embed)
    assert {r.competitor.title for r in result.pages[0].rows} == {"Kosten", "Förderung"}
