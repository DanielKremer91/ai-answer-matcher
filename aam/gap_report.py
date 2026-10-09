"""Tables, protocol lines, summary and Excel for the content-gap mode (spec 2026-10-09, section 6)."""

from __future__ import annotations

import io
import statistics

import pandas as pd

from aam.gap import GapResult
from aam.report import _count, _de, _excel_frame, _frame, _score

SHEETS = ("Abschnitte", "Protokoll")
_BASE_COLUMNS = ["Wettbewerber-URL", "Rang", "Abschnitt (Wettbewerber)", "Wörter", "Beste eigene URL",
                 "Eigener Abschnitt", "Wörter eigen", "Kosinus", "Bestes Fenster"]
_AI_COLUMNS = ["Nächste AI-Aussage", "Kosinus AI-Aussage"]
_TEXT_COLUMNS = ["Text Wettbewerber", "Text eigen"]


def sections_frame(result: GapResult) -> pd.DataFrame:
    """One row per competitor section, per page in the given order, best match first."""
    with_answer = result.answer is not None
    rows = []
    for gap_page in result.pages:
        for row in gap_page.rows:
            cells = [gap_page.url, row.rank, row.competitor.title, row.competitor.words, row.own.url,
                     row.own.title, row.own.words, _score(row.score), row.best_window]
            if with_answer:
                cells += [row.statement.text if row.statement else "", _score(row.statement_score)]
            cells += [row.competitor.text, row.own.text]
            rows.append(cells)
    columns = _BASE_COLUMNS + (_AI_COLUMNS if with_answer else []) + _TEXT_COLUMNS
    return _frame(rows, columns, result.keyword)


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
    """Reading hint for the first competitor page: how many sections have a counterpart."""
    if not result.pages or not result.pages[0].rows:
        return ""
    gap_page = result.pages[0]
    rows = gap_page.rows
    n = len(rows)
    if n == 1:
        if gap_page.note:
            return gap_page.note
        return f"{gap_page.url}: ein Abschnitt, Kosinus {_de(rows[0].score)} gegen {rows[0].own.url}."
    median = statistics.median(row.score for row in rows)
    covered = sum(row.score >= median for row in rows)
    missing = n - covered
    if missing == 0:
        return (f"{gap_page.url}: alle {n} Abschnitte haben bei dir einen Gegenpart mit Kosinus ab "
                f"{_de(median)}.")
    tail = ("Der Abschnitt am Ende der Liste ist der Kandidat für eine Lücke." if missing == 1
            else f"Die {missing} Abschnitte am Ende der Liste sind die Kandidaten für Lücken.")
    return (f"{gap_page.url}: {covered} von {n} Abschnitten haben bei dir einen Gegenpart mit Kosinus ab "
            f"{_de(median)}. {tail}")


def _protocol_frame(result: GapResult) -> pd.DataFrame:
    rows = [("Datum", result.date), ("Anbieter", result.provider), ("Modell", result.model)]
    rows += [("Protokoll", line) for line in gap_protocol_lines(result)]
    return _frame(rows, ["Angabe", "Wert"], result.keyword)


def gap_to_excel(result: GapResult) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in zip(SHEETS, (sections_frame(result), _protocol_frame(result))):
            _excel_frame(frame).to_excel(writer, sheet_name=name, index=False)
    return buffer.getvalue()
