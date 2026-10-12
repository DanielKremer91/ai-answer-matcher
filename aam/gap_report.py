"""Tables, protocol lines, summary and Excel for the content-gap mode (spec 2026-10-09, section 6)."""

from __future__ import annotations

import io
import statistics

import pandas as pd

from aam.gap import GapResult
from aam.report import _count, _de, _excel_frame, _frame, _score

SHEETS = ("Abschnitte", "Protokoll", "Erklärung")
_BASE_COLUMNS = ["Wettbewerber-URL", "Rang", "Abschnitt (Wettbewerber)", "Wörter", "Beste eigene URL",
                 "Eigener Abschnitt", "Wörter eigen", "Kosinus", "Vergleich", "Bestes Fenster"]
_AI_COLUMNS = ["Nächste AI-Aussage", "Kosinus AI-Aussage", "Priorität"]
_TEXT_COLUMNS = ["Text Wettbewerber", "Text eigen"]


def sections_frame(result: GapResult) -> pd.DataFrame:
    """One row per competitor section, per page in the given order, best match first."""
    with_answer = result.answer is not None
    rows = []
    for gap_page in result.pages:
        for row in gap_page.rows:
            cells = [gap_page.url, row.rank, row.competitor.title, row.competitor.words, row.own.url,
                     row.own.title, row.own.words, _score(row.score), row.method, row.best_window]
            if with_answer:
                # Priority: close to an AI statement, far from your own content. The higher, the
                # more the competitor's section matters for the answer and the less you cover it.
                priority = (None if row.statement_score is None
                            else round(float(row.statement_score) - float(row.score), 4))
                cells += [row.statement.text if row.statement else "", _score(row.statement_score), priority]
            cells += [row.competitor.text, row.own.text]
            rows.append(cells)
    columns = _BASE_COLUMNS + (_AI_COLUMNS if with_answer else []) + _TEXT_COLUMNS
    return _frame(rows, columns, result.keyword)


def priority_frame(result: GapResult) -> pd.DataFrame:
    """The sections frame ordered by priority (needs an answer); rows without priority last."""
    frame = sections_frame(result)
    if "Priorität" not in frame.columns:
        return frame
    return frame.sort_values("Priorität", ascending=False, na_position="last", kind="stable").reset_index(drop=True)


def _level(level) -> str:
    return f"H{level}" if level else "keine"


def gap_protocol_lines(result: GapResult) -> list[str]:
    lines = []
    for gap_page in result.pages:
        whole = (f"Ganztext-Wert {_de(gap_page.page_score)} gegen {gap_page.best_own_url}"
                 if gap_page.best_own_url else "Ganztext-Wert nicht verfügbar")
        lines.append(f"{gap_page.url}: Abschnittsebene {_level(gap_page.level)}, "
                     f"{_count(gap_page.sections, 'Abschnitt', 'Abschnitte')}, "
                     f"{_count(gap_page.words, 'Wort', 'Wörter')}, {whole}")
        if gap_page.note:
            lines.append(gap_page.note)
    for url, level, sections in result.own_pages:
        lines.append(f"Eigene Seite {url}: Abschnittsebene {_level(level)}, "
                     f"{_count(sections, 'Abschnitt', 'Abschnitte')}")
    return lines


def gap_summary(result: GapResult) -> str:
    """Reading hint for the first competitor page: the range of the scores and where to start reading."""
    if not result.pages or not result.pages[0].rows:
        return ""
    gap_page = result.pages[0]
    rows = gap_page.rows
    n = len(rows)
    if n == 1:
        if gap_page.note:
            return gap_page.note
        return f"{gap_page.url}: ein Abschnitt, Kosinus {_de(rows[0].score)} gegen {rows[0].own.url}."
    scores = [row.score for row in rows]
    return (f"{gap_page.url}: {n} Abschnitte, Kosinus von {_de(min(scores))} bis {_de(max(scores))}. "
            f"Die untere Hälfte der Liste (Kosinus unter {_de(statistics.median(scores))}) zuerst prüfen, "
            "dort stehen die Kandidaten für Lücken.")


def _protocol_frame(result: GapResult) -> pd.DataFrame:
    rows = [("Datum", result.date), ("Anbieter", result.provider), ("Modell", result.model)]
    rows += [("Protokoll", line) for line in gap_protocol_lines(result)]
    return _frame(rows, ["Angabe", "Wert"], result.keyword)


GAP_COLUMN_EXPLANATIONS = {
    ("alle", "Keyword/Prompt"): "Das Keyword oder der Prompt der AI-Antwort, falls eine eingegeben wurde; sonst leer.",
    ("Abschnitte", "Wettbewerber-URL"): "Die Wettbewerber-Seite; bei eingefügtem Text die Bezeichnung, die du eingetragen hast.",
    ("Abschnitte", "Rang"): "Platz des Wettbewerber-Abschnitts in der Liste, nach Kosinus absteigend; die letzten Plätze sind die Kandidaten für Lücken.",
    ("Abschnitte", "Abschnitt (Wettbewerber)"): "Überschrift des Abschnitts auf der Wettbewerber-Seite; ein Abschnitt reicht von einer Überschrift bis zur nächsten derselben Ebene.",
    ("Abschnitte", "Wörter"): "Länge des Wettbewerber-Abschnitts in Wörtern.",
    ("Abschnitte", "Beste eigene URL"): "Deine Seite, deren Abschnitt dem Wettbewerber-Abschnitt am nächsten liegt.",
    ("Abschnitte", "Eigener Abschnitt"): "Überschrift dieses eigenen Abschnitts.",
    ("Abschnitte", "Wörter eigen"): "Länge des eigenen Abschnitts in Wörtern.",
    ("Abschnitte", "Kosinus"): "Ähnlichkeit zwischen beiden Abschnitten, 0 bis 1. Hoch = Thema vorhanden, niedrig = Thema fehlt wahrscheinlich. Absolute Werte hängen vom Modell ab.",
    ("Abschnitte", "Vergleich"): "„Ganztext“ = beide Abschnitte als Ganzes verglichen. „Fenster aus …“ = ein Abschnitt war mehr als viermal so lang wie der andere; der längere wurde in Fenster geteilt, das beste Fenster zählt.",
    ("Abschnitte", "Bestes Fenster"): "Text des besten Fensters, wenn gefenstert wurde; sonst leer.",
    ("Abschnitte", "Nächste AI-Aussage"): "Die Aussage der AI-Antwort, die diesem Wettbewerber-Abschnitt am nächsten liegt.",
    ("Abschnitte", "Kosinus AI-Aussage"): "Ähnlichkeit zwischen Wettbewerber-Abschnitt und dieser AI-Aussage: Zählt der Abschnitt für die AI-Antwort?",
    ("Abschnitte", "Priorität"): "Kosinus AI-Aussage minus Kosinus: hoch = nah an der AI-Antwort und weit weg von deinem Content, also eine Lücke, die sich zuerst lohnt.",
    ("Abschnitte", "Text Wettbewerber"): "Volltext des Wettbewerber-Abschnitts.",
    ("Abschnitte", "Text eigen"): "Volltext des eigenen Abschnitts.",
    ("Protokoll", "Angabe"): "Name des Protokolleintrags.",
    ("Protokoll", "Wert"): "Datum, Anbieter, Modell und pro Seite Abschnittsebene, Anzahl Abschnitte, Wörter und Ganztext-Wert.",
}


def gap_explanation_frame(result: GapResult) -> pd.DataFrame:
    rows = [(sheet, column, meaning) for (sheet, column), meaning in GAP_COLUMN_EXPLANATIONS.items()]
    return _frame(rows, ["Blatt", "Spalte", "Bedeutung"], result.keyword)


def gap_to_excel(result: GapResult) -> bytes:
    buffer = io.BytesIO()
    frames = (sections_frame(result), _protocol_frame(result), gap_explanation_frame(result))
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in zip(SHEETS, frames):
            _excel_frame(frame).to_excel(writer, sheet_name=name, index=False)
    return buffer.getvalue()
