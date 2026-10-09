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
        lines.append(f"{gap_page.url}: Abschnittsebene {_level(gap_page.level)}, "
                     f"{_count(gap_page.sections, 'Abschnitt', 'Abschnitte')}, "
                     f"{_count(gap_page.words, 'Wort', 'Wörter')}, "
                     f"Ganztext-Wert {_de(gap_page.page_score)} gegen {gap_page.best_own_url}")
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
    scores = [row.score for row in gap_page.rows]
    median = statistics.median(scores)
    total = len(scores)
    covered = sum(score >= median for score in scores)
    return (f"{gap_page.url}: {covered} von {total} Abschnitten haben bei dir einen Gegenpart mit Kosinus "
            f"über {_de(median)}; die {total - covered} Abschnitte am Ende der Liste sind die Kandidaten "
            "für Lücken.")


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
