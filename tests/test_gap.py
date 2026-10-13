"""Content-gap analysis between competitor pages and own pages (fake embedder, no network)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from aam.embeddings import embed_cached
from aam.gap import (METHOD_WHOLE, METHOD_WINDOW_COMPETITOR, METHOD_WINDOW_OWN, WINDOW_ALL_UP_TO, GapPage, GapResult,
                     GapRow, analyze_gap)
from aam.models import OWN, SOURCE, Document
from aam.segment import segment_answer
from aam.sections import plain_text
from tests.fakes import ANSWER, bow_vector

ONE_SECTION = "Die Seite hat nur einen Abschnitt."
NOTE = ONE_SECTION + " Verglichen wird die ganze Seite."


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


def long_own(topic: str, n_fill: int = 9) -> str:
    return "\n".join(sentence(f"{topic}{i}x", 20) for i in range(n_fill + 1))


def test_windows_are_built_for_at_most_five_own_sections_per_competitor_section():
    short = sentence("thema", 20)
    comp = page("https://wett.de/a", SOURCE, ("Thema", short))
    bodies = [long_own(f"fuell{k}n") for k in range(45)]  # above WINDOW_ALL_UP_TO
    own = page("https://me.de/a", OWN, *[(f"Eigen {k}", b) for k, b in enumerate(bodies)])
    recorder = Recorder()
    result = run([comp], [own], embed=recorder)
    assert len(recorder.calls) == 2
    window_texts = recorder.calls[1]
    touched = {k for k, b in enumerate(bodies) if any(w in b for w in window_texts)}
    assert window_texts and len(touched) <= 5
    assert len(result.pages[0].rows) == 1


def test_best_section_with_matching_content_is_found_when_in_the_top_five():
    short = sentence("thema", 20)
    comp = page("https://wett.de/a", SOURCE, ("Thema", short))
    match = "\n".join([sentence(f"fuell{i}x", 20) for i in range(4)] + [short]
                      + [sentence(f"fuell{i}y", 20) for i in range(5)])
    sections = [(f"Eigen {k}", long_own(f"andere{k}n")) for k in range(19)]
    sections.insert(7, ("Treffer", match))
    result = run([comp], [page("https://me.de/a", OWN, *sections)])
    (row,) = result.pages[0].rows
    assert row.own.title == "Treffer"
    assert row.windowed and row.best_window == short
    assert row.score > 0.9


def test_cap_does_not_change_a_small_case():
    short = sentence("thema", 20)
    comp = page("https://wett.de/a", SOURCE, ("Thema", short))
    match = "\n".join([sentence(f"fuell{i}x", 20) for i in range(4)] + [short]
                      + [sentence(f"fuell{i}y", 20) for i in range(5)])
    own = page("https://me.de/a", OWN, ("Anderes", long_own("anders")), ("Treffer", match),
               ("Kurz", words("kurz", 12)))
    recorder = Recorder()
    result = run([comp], [own], embed=recorder)
    (row,) = result.pages[0].rows
    assert (row.own.title, row.windowed, row.best_window) == ("Treffer", True, short)
    # Both long sections are among the top 5, so both get windows, as without any cap.
    assert any(w in long_own("anders") for w in recorder.calls[1])


def test_identical_sections_score_exactly_one():
    comp = page("https://wett.de/a", SOURCE, ("Kosten", KOSTEN), ("Förderung", FOERDERUNG))
    own = page("https://me.de/a", OWN, ("Kosten", KOSTEN), ("Förderung", FOERDERUNG))
    result = run([comp], [own])
    scores = [r.score for r in result.pages[0].rows]
    assert scores == pytest.approx([1.0, 1.0]) and max(scores) <= 1.0
    assert result.pages[0].page_score <= 1.0


def test_dropped_sections_are_reported():
    def embed(texts):
        return [None if "garantie0" in t else bow_vector(t, 512) for t in texts]

    result = run([competitor()], [own_page()], embed=embed)  # the page text contains the marker too
    (gap_page,) = result.pages
    assert gap_page.dropped == 1
    assert "1 Abschnitt konnte nicht eingebettet werden und fehlt." in gap_page.note
    assert NOTE not in gap_page.note

    def embed_two(texts):
        return [None if "garantie0" in t or "foerder0" in t else bow_vector(t, 512) for t in texts]

    gap_page = run([competitor()], [own_page()], embed=embed_two).pages[0]
    assert gap_page.dropped == 2
    assert gap_page.note.startswith("2 Abschnitte konnten nicht eingebettet werden und fehlen.")

    def embed_sections_only(texts):  # drops the sections, keeps the page vector
        return [None if t.startswith(("Garantie\n", "Förderung\n")) else bow_vector(t, 512) for t in texts]

    gap_page = run([competitor()], [own_page()], embed=embed_sections_only).pages[0]
    assert gap_page.note == "2 Abschnitte konnten nicht eingebettet werden und fehlen."


def test_dropped_note_follows_single_section_note():
    comp = Document("https://wett.de/a", "Erster Absatz über Kosten.\nZweiter Absatz über Förderung.", SOURCE)

    def embed(texts):
        return [None if "Erster Absatz" in t and "Zweiter" not in t else bow_vector(t, 512) for t in texts]

    # The single section (the whole page) still has a vector here, so only the single-section note shows.
    assert run([comp], [own_page()], embed=embed).pages[0].note == NOTE

    def embed_none(texts):
        return [None if "Erster Absatz" in t else bow_vector(t, 512) for t in texts]

    gap_page = run([comp], [own_page()], embed=embed_none).pages[0]
    assert gap_page.rows == [] and gap_page.dropped == 1
    assert gap_page.note == f"{NOTE} 1 Abschnitt konnte nicht eingebettet werden und fehlt. " \
                            "Kein Seitenvektor, Ganztext-Wert nicht verfügbar."
    assert gap_page.page_score == 0.0 and gap_page.best_own_url == ""


def test_missing_page_vector_is_noted():
    comp = competitor()

    def embed(texts):
        return [None if t == plain_text(comp) else bow_vector(t, 512) for t in texts]

    gap_page = run([comp], [own_page()], embed=embed).pages[0]
    assert gap_page.note == "Kein Seitenvektor, Ganztext-Wert nicht verfügbar."
    assert gap_page.best_own_url == "" and gap_page.dropped == 0
    assert len(gap_page.rows) == 3


PASSAGE = " ".join(sentence(f"pass{k}x", 12) for k in range(3))  # 36 words, the competitor's passage


def long_with_passage(n_paragraphs: int = 20) -> str:
    """About 400 words in paragraphs of 20 words with the passage copied verbatim into the middle."""
    paragraphs = [sentence(f"lang{i}x", 20) for i in range(n_paragraphs)]
    paragraphs.insert(n_paragraphs // 2, PASSAGE)
    return "\n".join(paragraphs)


def related_short_sections(n: int) -> list[tuple[str, str]]:
    tokens = PASSAGE.lower().replace(".", "").split()
    return [(f"Verwandt {i}", " ".join(tokens[2 * i:2 * i + 10]) + f" extra{i}") for i in range(n)]


def test_long_own_section_with_the_passage_beats_six_related_short_sections():
    comp = page("https://wett.de/a", SOURCE, ("Thema", PASSAGE))
    own = page("https://me.de/a", OWN, ("Thema", long_with_passage()), *related_short_sections(6))
    (row,) = run([comp], [own]).pages[0].rows
    assert row.own.title == "Thema" and row.own.words > 400
    assert row.windowed and row.best_window == PASSAGE
    assert row.score > 0.95


def test_above_the_all_pairs_limit_the_long_section_is_found_through_the_length_ratio_candidates():
    comp = page("https://wett.de/a", SOURCE, ("Thema", PASSAGE))
    fremd = [(f"Fremd {k}", words(f"fremd{k}x", 12)) for k in range(40)]
    own = page("https://me.de/a", OWN, ("Thema", long_with_passage()), *related_short_sections(6), *fremd)
    recorder = Recorder()
    (row,) = run([comp], [own], embed=recorder).pages[0].rows
    assert WINDOW_ALL_UP_TO == 40
    assert row.own.title == "Thema" and row.windowed and row.best_window == PASSAGE
    assert row.score > 0.95
    assert len(recorder.calls) == 2


def test_up_to_forty_own_sections_every_pair_above_the_ratio_gets_windows():
    comp = page("https://wett.de/a", SOURCE, ("Thema", sentence("thema", 20)))
    bodies = [long_own(f"fuell{k}n") for k in range(WINDOW_ALL_UP_TO)]
    own = page("https://me.de/a", OWN, *[(f"Eigen {k}", b) for k, b in enumerate(bodies)])
    recorder = Recorder()
    run([comp], [own], embed=recorder)
    touched = {k for k, b in enumerate(bodies) if any(w in b for w in recorder.calls[1])}
    assert touched == set(range(WINDOW_ALL_UP_TO))


def test_the_section_title_is_part_of_the_embedded_text():
    question = "Wie lange dauert die Sanierung eines Daches im Altbau"
    comp = page("https://wett.de/a", SOURCE, (question, "Etwa zwei Wochen."), ("Garantie", GARANTIE))
    own = page("https://me.de/a", OWN, (question, "Das hängt vom Gerüst ab."), ("Anderes", words("anders", 12)))
    rows = {r.competitor.title: r for r in run([comp], [own]).pages[0].rows}
    assert rows[question].own.title == question
    assert rows[question].score > 0.6
    assert rows[question].score > rows["Garantie"].score


def test_methods_name_where_the_score_comes_from():
    comp = page("https://wett.de/a", SOURCE, ("Kosten", KOSTEN), ("Thema", PASSAGE))
    own = page("https://me.de/a", OWN, ("Kosten", KOSTEN), ("Lang", long_with_passage()))
    rows = {r.competitor.title: r for r in run([comp], [own]).pages[0].rows}
    assert rows["Kosten"].method == METHOD_WHOLE and not rows["Kosten"].windowed
    assert rows["Thema"].method == METHOD_WINDOW_OWN == "Fenster aus eigenem Abschnitt"
    assert METHOD_WHOLE == "Ganztext"

    long_comp = page("https://wett.de/a", SOURCE, ("Lang", long_with_passage()))
    own = page("https://me.de/a", OWN, ("Thema", PASSAGE), ("Garantie", GARANTIE))
    (row,) = run([long_comp], [own]).pages[0].rows
    assert row.windowed and row.best_window == PASSAGE
    assert row.method == METHOD_WINDOW_COMPETITOR == "Fenster aus Wettbewerber-Abschnitt"


def test_windowed_pair_scores_the_better_of_whole_and_window():
    # The short text's words are spread over five paragraphs of the long one: no window holds them all,
    # so the whole-text cosine wins although the pair is above the length ratio.
    shared = [f"streu{i}" for i in range(10)]
    short = " ".join(shared)
    paragraphs = [" ".join(shared[2 * i:2 * i + 2] + [f"fuell{i}x{j}" for j in range(7)]) + "." for i in range(5)]
    long_text = "\n".join(paragraphs)  # 45 words against 10: ratio 4.5
    comp = page("https://wett.de/a", SOURCE, ("Kurz", short))
    own = page("https://me.de/a", OWN, ("Lang", long_text))
    (row,) = run([comp], [own]).pages[0].rows
    a, b = bow_vector(f"Kurz\n{short}", 512), bow_vector(f"Lang\n{long_text}", 512)
    whole = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert row.score == pytest.approx(whole, abs=1e-5)
    assert row.score > 0.4
    assert not row.windowed and row.best_window == "" and row.method == METHOD_WHOLE


def test_single_section_note_also_fires_for_a_page_with_one_heading_section():
    comp = page("https://wett.de/a", SOURCE, ("Nur eine", KOSTEN))
    (gap_page,) = run([comp], [own_page()]).pages
    assert gap_page.level == 2 and gap_page.sections == 1
    assert gap_page.note == ONE_SECTION


def test_statement_score_uses_statement_sized_windows_of_long_sections():
    answer = segment_answer("kw", ANSWER)
    statement = "Die KfW fördert die Dachdämmung mit zinsgünstigen Krediten."
    filler = " ".join(f"Fuellwort{i} steht hier als Beispieltext ohne Bezug zum Thema Nummer {i} Ende." for i in range(30))
    competitor = Document("https://wett.de/lang", "## Förderung\n" + filler + " " + statement + " " + filler, SOURCE)
    own = Document("https://me.de/a", "## Kosten\nEine Neueindeckung kostet 100 Euro pro Quadratmeter.", OWN)
    result = run([competitor], [own], answer=answer)
    row = result.pages[0].rows[0]
    assert row.statement is not None and row.statement.text == statement
    assert row.statement_score > 0.5  # a window holding the statement; the whole 400-word section scores < 0.2
