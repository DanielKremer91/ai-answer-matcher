"""Tables, protocol, summary and Excel of the content-gap mode (fake embedder, no network)."""

from __future__ import annotations

import io
from dataclasses import replace
from datetime import date

import pandas as pd
from openpyxl import load_workbook

from aam.gap import analyze_gap
from aam.gap_report import gap_protocol_lines, gap_summary, gap_to_excel, sections_frame
from aam.models import OWN, SOURCE, Document
from aam.segment import segment_answer
from tests.fakes import ANSWER, bow_vector

BASE = ["Keyword/Prompt", "Wettbewerber-URL", "Rang", "Abschnitt (Wettbewerber)", "Wörter",
        "Beste eigene URL", "Eigener Abschnitt", "Wörter eigen", "Kosinus eigen", "Vergleichsart", "Bestes Fenster"]
VERDICT = ["Aussagen", "abgedeckt", "schwach", "Lücke", "Urteil"]
AI = ["Nächste AI-Aussage", "Kosinus AI-Aussage", "Kosinus AI-Aussage (ganzer Abschnitt)", "Priorität"]
TEXTS = ["Text Wettbewerber", "Text eigen"]


def words(prefix: str, n: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def page(url, origin, *sections):
    text = "\n".join(f"## {title}\n{body}" for title, body in sections)
    return Document(url, "# Titel\n" + text, origin)


def embed(texts):
    return [bow_vector(t, 512) for t in texts]


KOSTEN, FOERDERUNG, GARANTIE = words("kosten", 12), words("foerder", 12), words("garantie", 12)


def sentence(prefix: str, n: int) -> str:
    ws = [f"{prefix}{i}" for i in range(n)]
    ws[0] = ws[0].capitalize()
    return " ".join(ws) + "."


def competitor(url="https://wett.de/a"):
    return page(url, SOURCE, ("Kosten", KOSTEN), ("Förderung", FOERDERUNG), ("Garantie", GARANTIE))


def own():
    return page("https://me.de/a", OWN, ("Preise", KOSTEN + " " + words("eigen", 3)),
                ("Zuschüsse", FOERDERUNG + " " + words("eigen", 3)))


def run(comps=None, owns=None, **kwargs):
    return analyze_gap(comps or [competitor()], owns or [own()], embed, "fake", "fake-model",
                       today=date(2026, 10, 9), **kwargs)


def test_columns_without_answer():
    frame = sections_frame(run())
    assert list(frame.columns) == BASE + VERDICT + TEXTS
    assert len(frame) == 3
    assert (frame["Keyword/Prompt"] == "").all()


def test_columns_with_answer_and_keyword():
    answer = segment_answer("dachsanierung kosten", ANSWER)
    frame = sections_frame(run(answer=answer, keyword="dachsanierung kosten"))
    assert list(frame.columns) == BASE + VERDICT + AI + TEXTS
    assert (frame["Keyword/Prompt"] == "dachsanierung kosten").all()
    assert frame["Nächste AI-Aussage"].map(bool).all()
    assert frame["Kosinus AI-Aussage"].notna().all()


def test_sorting_is_kept_and_values_are_rounded():
    result = run(comps=[competitor("https://wett.de/a"), competitor("https://wett.de/b")])
    frame = sections_frame(result)
    assert list(frame["Wettbewerber-URL"]) == ["https://wett.de/a"] * 3 + ["https://wett.de/b"] * 3
    assert list(frame["Rang"]) == [1, 2, 3, 1, 2, 3]
    first = frame.iloc[:3]
    assert list(first["Kosinus eigen"]) == sorted(first["Kosinus eigen"], reverse=True)
    assert first.iloc[2]["Abschnitt (Wettbewerber)"] == "Garantie"
    assert all(round(v, 4) == v for v in frame["Kosinus eigen"])
    assert first.iloc[0]["Text Wettbewerber"] in (KOSTEN, FOERDERUNG)
    assert first.iloc[0]["Beste eigene URL"] == "https://me.de/a"


def test_excel_has_two_sheets_with_keyword_first():
    data = gap_to_excel(run(keyword="k"))
    sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
    assert list(sheets) == ["Abschnitte", "Aussagen", "Protokoll", "Erklärung"]
    assert sheets["Abschnitte"].columns[0] == "Keyword/Prompt" and sheets["Aussagen"].columns[0] == "Keyword/Prompt"
    assert sheets["Protokoll"].columns[0] == "Keyword/Prompt"
    assert list(sheets["Protokoll"].columns[1:]) == ["Angabe", "Wert"]
    protocol = sheets["Protokoll"]
    assert list(protocol["Angabe"][:3]) == ["Datum", "Anbieter", "Modell"]
    assert list(protocol["Wert"][:3]) == ["2026-10-09", "fake", "fake-model"]


def test_excel_cells_are_sanitised():
    comp = page("https://wett.de/a", SOURCE, ("Eins", "=SUMME(A1) " + words("eins", 12)), ("Zwei", FOERDERUNG))
    data = gap_to_excel(run(comps=[comp]))
    frame = pd.read_excel(io.BytesIO(data), sheet_name="Abschnitte")
    assert not any(str(v).startswith("=") for v in frame["Text Wettbewerber"])


def test_protocol_lines_wording_for_two_and_one_sections():
    comp = page("https://wett.de/a", SOURCE, ("Eins", words("eins", 12)), ("Zwei", FOERDERUNG))
    result = run(comps=[comp])
    lines = gap_protocol_lines(result)
    score = f"{result.pages[0].page_score:.2f}".replace(".", ",")
    assert lines[0] == (f"https://wett.de/a: Abschnittsebene H2, 2 Abschnitte, 24 Wörter, "
                        f"Ganztext-Wert {score} gegen https://me.de/a")
    assert lines[-2] == "Eigene Seite https://me.de/a: Abschnittsebene H2, 2 Abschnitte"
    assert lines[-1].startswith("Aussagen-Ebene: Chunk-Zielgröße 30 Wörter, ") and lines[-1].endswith(
        "Schwellen abgedeckt 0,05, schwach 0,15")
    assert len(lines) == 3

    single = Document("https://wett.de/b", "Erster Absatz über Kosten.\nZweiter Absatz.", SOURCE)
    one_own = page("https://me.de/b", OWN, ("Nur", words("nur", 12)))
    lines = gap_protocol_lines(run(comps=[single], owns=[one_own]))
    assert lines[0].startswith("https://wett.de/b: Abschnittsebene keine, 1 Abschnitt, 6 Wörter, Ganztext-Wert ")
    assert lines[1] == "Die Seite hat nur einen Abschnitt. Verglichen wird die ganze Seite."
    assert lines[2] == "Eigene Seite https://me.de/b: Abschnittsebene H2, 1 Abschnitt"
    assert lines[3].startswith("Aussagen-Ebene:") and len(lines) == 4


def test_protocol_sheet_lists_the_lines():
    result = run()
    data = gap_to_excel(result)
    protocol = pd.read_excel(io.BytesIO(data), sheet_name="Protokoll")
    values = list(protocol["Wert"])
    assert all(line in values for line in gap_protocol_lines(result))


def with_scores(result, scores, **page_changes):
    """The first page of `result` with its rows cut or reused to carry exactly these scores."""
    gap_page = result.pages[0]
    rows = [replace(gap_page.rows[i % len(gap_page.rows)], score=score, rank=i + 1) for i, score in enumerate(scores)]
    return replace(result, pages=[replace(gap_page, rows=rows, **page_changes)], statements=[])


def test_summary_without_rows_is_empty():
    result = run()
    assert gap_summary(with_scores(result, [])) == ""
    assert gap_summary(replace(result, pages=[])) == ""


def test_summary_without_statements_falls_back_to_the_section_hint():
    result = replace(run(), statements=[])
    scores = sorted(r.score for r in result.pages[0].rows)
    low, median, high = (f"{v:.2f}".replace(".", ",") for v in (scores[0], scores[1], scores[2]))
    assert gap_summary(result) == (
        f"https://wett.de/a: 3 Abschnitte, Kosinus von {low} bis {high}. Die untere Hälfte der Liste "
        f"(Kosinus unter {median}) zuerst prüfen, dort stehen die Kandidaten für Lücken.")


def test_summary_names_verdicts_in_reading_order():
    result = run()  # the competitor's Garantie section has no counterpart in the own page
    text = gap_summary(result)
    first, *rest = text.split("  \n")
    assert first == "https://wett.de/a: 3 Abschnitte, 3 Aussagen."
    assert rest[0].startswith("Themenlücken, zuerst prüfen: **Garantie** (1 von 1 Aussagen ohne Gegenpart)")
    assert rest[-1].startswith("Abgedeckt: ") and "Kosten" in rest[-1] and "Förderung" in rest[-1]


def test_summary_lists_detail_gaps_with_the_missing_statements():
    comp = page("https://wett.de/a", SOURCE, ("Kosten", "\n".join([KOSTEN + ".", sentence("kosten", 12),
                                                                     sentence("garantie", 12)])))
    owns = page("https://me.de/a", OWN, ("Kosten", "\n".join([KOSTEN + ".", sentence("kosten", 12)])))
    result = run(comps=[comp], owns=[owns], statement_target=12)  # 12-word lines stay separate chunks
    v = result.pages[0].rows[0].verdict
    assert (v.total, v.gaps) == (3, 1) and v.verdict == "Detaillücken"
    text = gap_summary(result)
    assert "Detaillücken: **Kosten** (1 von 3): „Garantie0 garantie1" in text


def test_summary_for_two_sections():
    text = gap_summary(with_scores(run(), [0.9, 0.5]))
    assert text == ("https://wett.de/a: 2 Abschnitte, Kosinus von 0,50 bis 0,90. Die untere Hälfte der Liste "
                    "(Kosinus unter 0,70) zuerst prüfen, dort stehen die Kandidaten für Lücken.")


def test_summary_uses_the_median_of_an_odd_and_an_even_number_of_rows():
    text = gap_summary(with_scores(run(), [0.9, 0.8, 0.3, 0.2, 0.1]))
    assert text == ("https://wett.de/a: 5 Abschnitte, Kosinus von 0,10 bis 0,90. Die untere Hälfte der Liste "
                    "(Kosinus unter 0,30) zuerst prüfen, dort stehen die Kandidaten für Lücken.")
    text = gap_summary(with_scores(run(), [0.9, 0.6, 0.4, 0.2]))
    assert "Kosinus von 0,20 bis 0,90" in text and "(Kosinus unter 0,50)" in text


def test_summary_with_equal_scores_still_reads_the_same_way():
    text = gap_summary(with_scores(run(), [0.8, 0.8, 0.8]))
    assert text == ("https://wett.de/a: 3 Abschnitte, Kosinus von 0,80 bis 0,80. Die untere Hälfte der Liste "
                    "(Kosinus unter 0,80) zuerst prüfen, dort stehen die Kandidaten für Lücken.")


def test_summary_for_a_single_section_page_returns_its_note_without_statements():
    single = Document("https://wett.de/b", "Erster Absatz über Kosten.\nZweiter Absatz.", SOURCE)
    result = run(comps=[single])
    assert gap_summary(replace(result, statements=[])) == "Die Seite hat nur einen Abschnitt. Verglichen wird die ganze Seite."
    assert gap_summary(result).startswith("https://wett.de/b: 1 Abschnitt, 1 Aussage.")


def test_summary_for_a_single_section_without_note_names_score_and_own_url():
    result = run()
    one = with_scores(result, [0.5], note="")
    assert gap_summary(one) == f"https://wett.de/a: ein Abschnitt, Kosinus 0,50 gegen {one.pages[0].rows[0].own.url}."


def test_protocol_line_without_whole_text_value():
    result = run()
    result = replace(result, pages=[replace(result.pages[0], best_own_url="", page_score=0.0)])
    line = gap_protocol_lines(result)[0]
    assert line.endswith(", Ganztext-Wert nicht verfügbar")
    assert "gegen" not in line


def test_very_long_texts_are_truncated_in_excel():
    comp = page("https://wett.de/a", SOURCE, ("Lang", "wort " * 10000), ("Kurz", FOERDERUNG))
    owns = page("https://me.de/a", OWN, ("Lang", "text " * 10000), ("Kurz", FOERDERUNG))
    result = run(comps=[comp], owns=[owns])
    assert max(len(r.competitor.text) for r in result.pages[0].rows) > 32767
    workbook = load_workbook(io.BytesIO(gap_to_excel(result)))
    sheet = workbook["Abschnitte"]
    header = [c.value for c in sheet[1]]
    for name in ("Text Wettbewerber", "Text eigen"):
        values = [row[header.index(name)].value for row in sheet.iter_rows(min_row=2)]
        assert max(len(v) for v in values) <= 32767
        longest = max(values, key=len)
        assert longest.endswith("[gekürzt]")


def test_vergleich_column_follows_kosinus_and_names_the_method():
    comp = page("https://wett.de/a", SOURCE, ("Kosten", KOSTEN), ("Lang", "\n".join(
        [" ".join(f"lang{i}x{j}" for j in range(20)) + "." for i in range(10)] + [FOERDERUNG])))
    result = run(comps=[comp])
    frame = sections_frame(result)
    columns = list(frame.columns)
    assert columns.index("Vergleichsart") == columns.index("Kosinus eigen") + 1
    by_title = frame.set_index("Abschnitt (Wettbewerber)")["Vergleichsart"]
    assert by_title["Kosten"] == "Ganztext"
    assert by_title["Lang"] == "Fenster aus Wettbewerber-Abschnitt"
    assert frame.set_index("Abschnitt (Wettbewerber)")["Bestes Fenster"]["Lang"] == FOERDERUNG
    assert frame.set_index("Abschnitt (Wettbewerber)")["Bestes Fenster"]["Kosten"] == ""
    excel = pd.read_excel(io.BytesIO(gap_to_excel(result)), sheet_name="Abschnitte")
    assert "Vergleichsart" in excel.columns


def test_priority_is_ai_closeness_minus_own_coverage():
    answer = segment_answer("dachsanierung kosten", ANSWER)
    frame = sections_frame(run(answer=answer, keyword="dachsanierung kosten"))
    expected = (frame["Kosinus AI-Aussage"] - frame["Kosinus eigen"]).round(4)
    assert (frame["Priorität"] == expected).all()


def test_priority_order_lists_the_relevant_gaps_first():
    from aam.gap_report import priority_frame

    answer = segment_answer("dachsanierung kosten", ANSWER)
    frame = priority_frame(run(answer=answer, keyword="dachsanierung kosten"))
    assert list(frame["Priorität"]) == sorted(frame["Priorität"], reverse=True)
    assert list(frame.columns)[:3] == ["Keyword/Prompt", "Wettbewerber-URL", "Rang"]


def test_gap_excel_has_an_explanation_sheet_covering_every_column():
    from aam.gap_report import gap_explanation_frame

    answer = segment_answer("dachsanierung kosten", ANSWER)
    result = run(answer=answer, keyword="dachsanierung kosten")
    sheets = pd.read_excel(io.BytesIO(gap_to_excel(result)), sheet_name=None)
    assert list(sheets) == ["Abschnitte", "Aussagen", "Protokoll", "Erklärung"]
    legend = gap_explanation_frame(result)
    explained = {(sheet, column) for sheet, column in zip(legend["Blatt"], legend["Spalte"])}
    for column in sections_frame(result).columns:
        assert ("Abschnitte", column) in explained or column == "Keyword/Prompt", column
    from aam.gap_report import statements_frame
    for column in statements_frame(result).columns:
        assert ("Aussagen", column) in explained or column == "Keyword/Prompt", column
    assert list(legend.columns) == ["Keyword/Prompt", "Blatt", "Spalte", "Bedeutung", "So ermittelt"]
    assert ("Protokoll", "Angabe") in explained and ("Protokoll", "Wert") in explained
