"""Tables, protocol, summary and Excel of the content-gap mode (fake embedder, no network)."""

from __future__ import annotations

import io
from datetime import date

import pandas as pd

from aam.gap import analyze_gap
from aam.gap_report import gap_protocol_lines, gap_summary, gap_to_excel, sections_frame
from aam.models import OWN, SOURCE, Document
from aam.segment import segment_answer
from tests.fakes import ANSWER, bow_vector

BASE = ["Keyword/Prompt", "Wettbewerber-URL", "Rang", "Abschnitt (Wettbewerber)", "Wörter",
        "Beste eigene URL", "Eigener Abschnitt", "Wörter eigen", "Kosinus", "Bestes Fenster"]
AI = ["Nächste AI-Aussage", "Kosinus AI-Aussage"]
TEXTS = ["Text Wettbewerber", "Text eigen"]


def words(prefix: str, n: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def page(url, origin, *sections):
    text = "\n".join(f"## {title}\n{body}" for title, body in sections)
    return Document(url, "# Titel\n" + text, origin)


def embed(texts):
    return [bow_vector(t, 512) for t in texts]


KOSTEN, FOERDERUNG, GARANTIE = words("kosten", 12), words("foerder", 12), words("garantie", 12)


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
    assert list(frame.columns) == BASE + TEXTS
    assert len(frame) == 3
    assert (frame["Keyword/Prompt"] == "").all()


def test_columns_with_answer_and_keyword():
    answer = segment_answer("dachsanierung kosten", ANSWER)
    frame = sections_frame(run(answer=answer, keyword="dachsanierung kosten"))
    assert list(frame.columns) == BASE + AI + TEXTS
    assert (frame["Keyword/Prompt"] == "dachsanierung kosten").all()
    assert frame["Nächste AI-Aussage"].map(bool).all()
    assert frame["Kosinus AI-Aussage"].notna().all()


def test_sorting_is_kept_and_values_are_rounded():
    result = run(comps=[competitor("https://wett.de/a"), competitor("https://wett.de/b")])
    frame = sections_frame(result)
    assert list(frame["Wettbewerber-URL"]) == ["https://wett.de/a"] * 3 + ["https://wett.de/b"] * 3
    assert list(frame["Rang"]) == [1, 2, 3, 1, 2, 3]
    first = frame.iloc[:3]
    assert list(first["Kosinus"]) == sorted(first["Kosinus"], reverse=True)
    assert first.iloc[2]["Abschnitt (Wettbewerber)"] == "Garantie"
    assert all(round(v, 4) == v for v in frame["Kosinus"])
    assert first.iloc[0]["Text Wettbewerber"] in (KOSTEN, FOERDERUNG)
    assert first.iloc[0]["Beste eigene URL"] == "https://me.de/a"


def test_excel_has_two_sheets_with_keyword_first():
    data = gap_to_excel(run(keyword="k"))
    sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
    assert list(sheets) == ["Abschnitte", "Protokoll"]
    assert sheets["Abschnitte"].columns[0] == "Keyword/Prompt"
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
    assert lines[-1] == "Eigene Seite https://me.de/a: Abschnittsebene H2, 2 Abschnitte"
    assert len(lines) == 2

    single = Document("https://wett.de/b", "Erster Absatz über Kosten.\nZweiter Absatz.", SOURCE)
    one_own = page("https://me.de/b", OWN, ("Nur", words("nur", 12)))
    lines = gap_protocol_lines(run(comps=[single], owns=[one_own]))
    assert lines[0].startswith("https://wett.de/b: Abschnittsebene keine, 1 Abschnitt, 6 Wörter, Ganztext-Wert ")
    assert lines[1] == "Die Seite hat nur einen Abschnitt, verglichen wird die ganze Seite."
    assert lines[2] == "Eigene Seite https://me.de/b: Abschnittsebene keine, 1 Abschnitt"
    assert len(lines) == 3


def test_protocol_sheet_lists_the_lines():
    result = run()
    data = gap_to_excel(result)
    protocol = pd.read_excel(io.BytesIO(data), sheet_name="Protokoll")
    values = list(protocol["Wert"])
    assert all(line in values for line in gap_protocol_lines(result))


def test_summary_wording():
    result = run()
    scores = sorted(r.score for r in result.pages[0].rows)
    median = scores[1]
    k = sum(s >= median for s in scores)
    text = gap_summary(result)
    shown = f"{median:.2f}".replace(".", ",")
    assert text == (f"https://wett.de/a: {k} von 3 Abschnitten haben bei dir einen Gegenpart mit Kosinus "
                    f"über {shown}; die {3 - k} Abschnitte am Ende der Liste sind die Kandidaten für Lücken.")
    assert k == 2
