"""Tables, protocol lines, summary and Excel for the content-gap mode (spec 2026-10-09, section 6)."""

from __future__ import annotations

import io
import statistics

import pandas as pd

from aam.gap import GapResult
from aam.gap_statements import VERDICT_DETAIL_GAPS, VERDICT_TOPIC_GAP, VERDICT_TOPIC_ONLY
from aam.match import STATUS_COVERED, STATUS_GAP, STATUS_WEAK
from aam.report import _count, _de, _excel_frame, _frame, _score

SHEETS = ("Abschnitte", "Aussagen", "Protokoll", "Erklärung")
_BASE_COLUMNS = ["Wettbewerber-URL", "Rang", "Abschnitt (Wettbewerber)", "Wörter", "Beste eigene URL",
                 "Eigener Abschnitt", "Wörter eigen", "Kosinus eigen", "Vergleichsart", "Bestes Fenster"]
_VERDICT_COLUMNS = ["Aussagen", "abgedeckt", "schwach", "Lücke", "Urteil"]
_AI_COLUMNS = ["Nächste AI-Aussage", "Kosinus AI-Aussage", "Kosinus AI-Aussage (ganzer Abschnitt)", "Priorität"]
_TEXT_COLUMNS = ["Text Wettbewerber", "Text eigen"]
_STATEMENT_COLUMNS = ["Wettbewerber-URL", "Abschnitt", "Aussage (Wettbewerber-Chunk)", "Wörter", "Beste eigene URL",
                      "Eigener Abschnitt", "Bester eigener Chunk", "Kosinus eigen", "Referenz Nachbarn",
                      "Referenz Spezifität", "Delta", "Status"]
_STATEMENT_AI_COLUMNS = ["Nächste AI-Aussage", "Kosinus AI-Aussage", "Priorität"]
_STATUS_ORDER = {STATUS_GAP: 0, STATUS_WEAK: 1, STATUS_COVERED: 2}


def sections_frame(result: GapResult) -> pd.DataFrame:
    """One row per competitor section, per page in the given order, best match first."""
    with_answer = result.answer is not None
    rows = []
    for gap_page in result.pages:
        for row in gap_page.rows:
            cells = [gap_page.url, row.rank, row.competitor.title, row.competitor.words, row.own.url,
                     row.own.title, row.own.words, _score(row.score), row.method, row.best_window]
            v = row.verdict
            cells += [v.total, v.covered, v.weak, v.gaps, v.verdict] if v else [0, 0, 0, 0, ""]
            if with_answer:
                # Priority: close to an AI statement, far from your own content. The higher, the
                # more the competitor's section matters for the answer and the less you cover it.
                priority = (None if row.statement_score is None
                            else round(float(row.statement_score) - float(row.score), 4))
                cells += [row.statement.text if row.statement else "", _score(row.statement_score),
                          _score(row.statement_whole_score), priority]
            cells += [row.competitor.text, row.own.text]
            rows.append(cells)
    columns = _BASE_COLUMNS + _VERDICT_COLUMNS + (_AI_COLUMNS if with_answer else []) + _TEXT_COLUMNS
    return _frame(rows, columns, result.keyword)


def statements_frame(result: GapResult) -> pd.DataFrame:
    """One row per competitor chunk: gaps first, then weak, then covered; within a status the largest
    shortfall first."""
    with_answer = result.answer is not None
    ordered = sorted(result.statements, key=lambda s: (_STATUS_ORDER.get(s.status, 9), s.delta))
    rows = []
    for s in ordered:
        cells = [s.competitor_url, s.section.title, s.text, s.words, s.own_url, s.own_section.title, s.own_text,
                 _score(s.score), _score(s.ref_neighbours), _score(s.ref_specificity), _score(s.delta), s.status]
        if with_answer:
            cells += [s.statement.text if s.statement else "", _score(s.statement_score), s.priority]
        rows.append(cells)
    return _frame(rows, _STATEMENT_COLUMNS + (_STATEMENT_AI_COLUMNS if with_answer else []), result.keyword)


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
    lines.append(f"Aussagen-Ebene: Chunk-Zielgröße {result.statement_target} Wörter, "
                 f"{_count(len(result.statements), 'Wettbewerber-Aussage', 'Wettbewerber-Aussagen')}, "
                 f"Schwellen abgedeckt {_de(result.covered_delta)}, schwach {_de(result.weak_delta)}")
    return lines


def gap_summary(result: GapResult) -> str:
    """Verdicts in reading order: topic gaps first, then detail gaps with the missing statements, then the
    rest. Without statements (no chunk got a vector) the old reading hint on the section scores."""
    if not result.pages or not result.pages[0].rows:
        return ""
    if not result.statements:
        return _section_hint(result)
    parts = []
    for gap_page in result.pages:
        topic, detail, weak_only, covered = [], [], [], []
        for row in gap_page.rows:
            v = row.verdict
            if v is None:
                continue
            if v.verdict == VERDICT_TOPIC_GAP:
                topic.append(f"**{row.competitor.title}** ({v.gaps} von {v.total} Aussagen ohne Gegenpart)")
            elif v.verdict == VERDICT_DETAIL_GAPS:
                missing = [s.text for s in result.statements
                           if s.competitor_url == gap_page.url and s.section.index == row.competitor.index
                           and s.status == STATUS_GAP]
                shown = "; ".join(f"„{_shorten(m)}“" for m in missing[:2])
                detail.append(f"**{row.competitor.title}** ({v.gaps} von {v.total}): {shown}")
            elif v.verdict == VERDICT_TOPIC_ONLY:
                weak_only.append(row.competitor.title)
            else:
                covered.append(row.competitor.title)
        lines = [f"{gap_page.url}: {_count(len(gap_page.rows), 'Abschnitt', 'Abschnitte')}, "
                 f"{_count(sum(1 for s in result.statements if s.competitor_url == gap_page.url), 'Aussage', 'Aussagen')}."]
        if topic:
            lines.append("Themenlücken, zuerst prüfen: " + "; ".join(topic) + ".")
        if detail:
            lines.append("Detaillücken: " + "; ".join(detail) + ".")
        if weak_only:
            lines.append("Nur beim Thema, nicht bei den Aussagen: " + ", ".join(weak_only) + ".")
        if not topic and not detail:
            lines.append("Keine Lücke auf Aussagen-Ebene." if not weak_only else "Keine Aussage ganz ohne Gegenpart.")
        if covered:
            lines.append("Abgedeckt: " + ", ".join(covered) + ".")
        parts.append("  \n".join(lines))
    return "\n\n".join(parts)


def _shorten(text: str, words: int = 14) -> str:
    parts = text.split()
    return text if len(parts) <= words else " ".join(parts[:words]) + " …"


def _section_hint(result: GapResult) -> str:
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
    ("alle", "Keyword/Prompt"): "Das Keyword oder der Prompt der AI-Antwort, falls du eine eingegeben hast. Sonst leer.",
    ("Abschnitte", "Wettbewerber-URL"): "Die Wettbewerber-Seite. Bei eingefügtem Text die Bezeichnung, die du eingetragen hast.",
    ("Abschnitte", "Rang"): "Platz des Abschnitts in der Liste, nach Kosinus eigen absteigend. Die letzten Plätze sind die Kandidaten für Lücken.",
    ("Abschnitte", "Abschnitt (Wettbewerber)"): "Überschrift des Abschnitts auf der Wettbewerber-Seite. Ein Abschnitt reicht von einer Überschrift bis zur nächsten derselben Ebene.",
    ("Abschnitte", "Wörter"): "Länge dieses Abschnitts in Wörtern.",
    ("Abschnitte", "Beste eigene URL"): "Deine Seite mit dem Abschnitt, der dem Wettbewerber-Abschnitt am nächsten kommt.",
    ("Abschnitte", "Eigener Abschnitt"): "Überschrift dieses eigenen Abschnitts.",
    ("Abschnitte", "Wörter eigen"): "Länge des eigenen Abschnitts in Wörtern.",
    ("Abschnitte", "Kosinus eigen"): "Wie ähnlich der Wettbewerber-Abschnitt und dein bester Abschnitt sind, 0 bis 1. Hoch = du behandelst das Thema. Niedrig = das Thema fehlt dir wahrscheinlich. Die Zahl hängt vom Modell ab.",
    ("Abschnitte", "Vergleichsart"): "„Ganztext“: Beide Abschnitte wurden als Ganzes verglichen. „Fenster aus …“: Ein Abschnitt war mehr als viermal so lang wie der andere, darum wurde der längere in Stücke geteilt und das beste Stück gezählt.",
    ("Abschnitte", "Bestes Fenster"): "Das beste Stück im Wortlaut, wenn geteilt wurde. Sonst leer.",
    ("Abschnitte", "Aussagen"): "In wie viele Aussagen (Chunks) dieser Abschnitt des Wettbewerbers geschnitten wurde.",
    ("Abschnitte", "abgedeckt"): "Wie viele dieser Aussagen du mit einem eigenen Chunk abdeckst (Blatt Aussagen).",
    ("Abschnitte", "schwach"): "Wie viele dieser Aussagen du nur dem Thema nach triffst, nicht in der Aussage.",
    ("Abschnitte", "Lücke"): "Wie viele dieser Aussagen bei dir keinen Gegenpart haben.",
    ("Abschnitte", "Urteil"): "„Themenlücke“: mehr als die Hälfte der Aussagen fehlt dir, das Thema fehlt praktisch ganz. „Detaillücken“: Du hast das Thema, aber einzelne Aussagen fehlen (Blatt Aussagen zeigt welche). „nur beim Thema“: keine Aussage fehlt ganz, aber die meisten triffst du nur ungefähr. „abgedeckt“: du sagst, was der Wettbewerber sagt.",
    ("Aussagen", "Wettbewerber-URL"): "Die Wettbewerber-Seite, aus der die Aussage stammt.",
    ("Aussagen", "Abschnitt"): "Der Abschnitt der Wettbewerber-Seite, in dem die Aussage steht.",
    ("Aussagen", "Aussage (Wettbewerber-Chunk)"): "Ein Stück des Wettbewerber-Texts von etwa der Chunk-Zielgröße (Seitenleiste), ein bis zwei Sätze. Das ist die Einheit, die hier als eine Aussage gilt.",
    ("Aussagen", "Wörter"): "Länge der Aussage in Wörtern.",
    ("Aussagen", "Beste eigene URL"): "Deine Seite mit dem Chunk, der dieser Aussage am nächsten kommt.",
    ("Aussagen", "Eigener Abschnitt"): "Der Abschnitt deiner Seite, in dem dieser Chunk steht.",
    ("Aussagen", "Bester eigener Chunk"): "Dieser Chunk im Wortlaut.",
    ("Aussagen", "Kosinus eigen"): "Wie nah dein bester Chunk an der Aussage liegt, 0 bis 1. Die Zahl hängt vom Modell ab, darum wird sie mit zwei Referenzen aus demselben Lauf verglichen.",
    ("Aussagen", "Referenz Nachbarn"): "Wie nah die anderen Aussagen desselben Wettbewerber-Abschnitts an dieser Aussage liegen (die nächste davon). Das ist das Niveau „gleiches Thema, andere Aussage“. Liegt dein Chunk darunter, bist du nur beim Thema. Leer, wenn der Abschnitt nur eine Aussage hat und die Seite keine andere Referenz liefert.",
    ("Aussagen", "Referenz Spezifität"): "Wie nah dein bester Chunk an den anderen Aussagen des Wettbewerbers liegt (der nächsten davon). Hoch heißt: Dein Chunk passt zu allem ein bisschen und ist kein echter Gegenpart.",
    ("Aussagen", "Delta"): "Kosinus eigen minus die höhere der beiden Referenzen. Null oder mehr heißt: Dein Chunk ist der Aussage näher als alles, was der Wettbewerber sonst sagt, du sagst dasselbe. Deutlich unter null heißt: Du triffst nur das Thema.",
    ("Aussagen", "Status"): "Aus dem Delta mit den Schwellen der Seitenleiste: „abgedeckt“ (Delta über −Schwelle abgedeckt), „schwach“ (zwischen den Schwellen), „Lücke“ (unter −Schwelle schwach). Die Lücken stehen oben in der Liste.",
    ("Aussagen", "Nächste AI-Aussage"): "Die Aussage der AI-Antwort, die dieser Wettbewerber-Aussage am nächsten kommt.",
    ("Aussagen", "Kosinus AI-Aussage"): "Wie nah die Wettbewerber-Aussage an dieser AI-Aussage liegt. Hoch heißt: Die AI sagt so etwas.",
    ("Aussagen", "Priorität"): "Kosinus AI-Aussage minus Kosinus eigen. Hoch = die AI sagt es, du nicht: zuerst schließen.",
    ("Abschnitte", "Nächste AI-Aussage"): "Die Aussage der AI-Antwort, die diesem Wettbewerber-Abschnitt am nächsten kommt.",
    ("Abschnitte", "Kosinus AI-Aussage"): "Kommt die Stelle vor? Wie nah der Abschnitt an dieser AI-Aussage liegt. Bei langen Abschnitten zählt das beste Stück in Chunk-Größe (Median der Aussagenlänge mal Chunk-Faktor aus der Seitenleiste). Hoch heißt: Irgendwo in diesem Abschnitt steht, was die AI sagt.",
    ("Abschnitte", "Kosinus AI-Aussage (ganzer Abschnitt)"): "Ist der ganze Abschnitt dieser Aussage gewidmet? Derselbe Vergleich ohne Fenster, der Abschnitt als Ganzes. Beide hoch: ein fokussierter Abschnitt zu genau dieser Aussage. Nur die linke Spalte hoch: Die Aussage steht irgendwo in einem breiten Abschnitt. Bei kurzen Abschnitten sind beide Werte gleich.",
    ("Abschnitte", "Priorität"): "Kosinus AI-Aussage minus Kosinus eigen. Hoch = nah an der AI-Antwort und weit weg von deinem Content: eine Lücke, die sich zuerst lohnt.",
    ("Abschnitte", "Text Wettbewerber"): "Der Wettbewerber-Abschnitt im Wortlaut.",
    ("Abschnitte", "Text eigen"): "Der eigene Abschnitt im Wortlaut.",
    ("Protokoll", "Angabe"): "Name des Eintrags.",
    ("Protokoll", "Wert"): "Datum, Anbieter, Modell und pro Seite: Abschnittsebene, Anzahl Abschnitte, Wörter und Ganztext-Wert.",
}


_COSINE = "Kosinus-Ähnlichkeit der beiden Embeddings"
GAP_COLUMN_CALCULATIONS = {
    ("Abschnitte", "Rang"): "Alle Abschnitte einer Wettbewerber-Seite nach Kosinus eigen absteigend sortiert.",
    ("Abschnitte", "Wörter"): "Wörter des Abschnitts (Überschrift und Text), getrennt an Leerzeichen.",
    ("Abschnitte", "Wörter eigen"): "Wörter des eigenen Abschnitts, getrennt an Leerzeichen.",
    ("Abschnitte", "Kosinus eigen"): "Jeder Abschnitt (Überschrift plus Text) wird als ein Text eingebettet. Wettbewerber-Abschnitt gegen jeden eigenen Abschnitt, " + _COSINE + "; der höchste Wert zählt. Bei ungleich langen Paaren (Spalte Vergleichsart) zusätzlich das beste Fenster, der höhere der beiden Werte gewinnt.",
    ("Abschnitte", "Vergleichsart"): "Ist ein Abschnitt mehr als viermal so lang wie sein Gegenpart, wird der längere an Satzgrenzen in Fenster von der Länge des kürzeren geschnitten (ein Satz Überlappung), jedes Fenster eingebettet. Bis 40 eigene Abschnitte für alle solchen Paare, darüber für die fünf besten Ganztext-Paare und die fünf längsten.",
    ("Abschnitte", "Kosinus AI-Aussage"): "Jede Aussage der AI-Antwort eingebettet. Wettbewerber-Abschnitt gegen jede Aussage, " + _COSINE + "; der höchste Wert. Ist der Abschnitt mehr als viermal so lang wie die Chunk-Zielgröße (Median der Aussagenlänge × Chunk-Faktor, 15 bis 80 Wörter), zusätzlich Fenster dieser Größe; der höhere Wert aus Ganztext und bestem Fenster gewinnt.",
    ("Abschnitte", "Kosinus AI-Aussage (ganzer Abschnitt)"): "Embedding des ganzen Wettbewerber-Abschnitts gegen das Embedding der in „Nächste AI-Aussage“ genannten Aussage, " + _COSINE + ", ohne Fenster.",
    ("Abschnitte", "Priorität"): "Kosinus AI-Aussage minus Kosinus eigen, auf vier Stellen gerundet.",
    ("Abschnitte", "Aussagen"): "Jeder Abschnitt wird mit dem Chunker aus Modus 1 in Chunks der Zielgröße „Aussagen-Ebene“ geschnitten (kurze Zeilen zusammengelegt, lange an Satzgrenzen geteilt).",
    ("Abschnitte", "Urteil"): "Themenlücke: Lücken > Hälfte der Aussagen. Detaillücken: 1 bis Hälfte. Nur beim Thema: keine Lücke, schwach > Hälfte. Sonst abgedeckt.",
    ("Aussagen", "Kosinus eigen"): "Embedding des Wettbewerber-Chunks gegen das Embedding jedes eigenen Chunks (alle eigenen Seiten), " + _COSINE + "; der höchste Wert.",
    ("Aussagen", "Referenz Nachbarn"): "Höchster Kosinus des Wettbewerber-Chunks zu den anderen Chunks desselben Abschnitts, die keinen Satz mit ihm teilen. Ohne solchen Nachbarn: Median dieser Werte über die Seite.",
    ("Aussagen", "Referenz Spezifität"): "Höchster Kosinus deines besten Chunks zu den anderen Wettbewerber-Chunks der Seite (ohne Satzüberlappung mit der Aussage).",
    ("Aussagen", "Delta"): "min(Kosinus eigen − Referenz Nachbarn, Kosinus eigen − Referenz Spezifität); ohne Nachbar-Referenz nur der zweite Wert.",
    ("Aussagen", "Status"): "delta_status wie in Modus 1: abgedeckt ab −Schwelle abgedeckt, schwach ab −Schwelle schwach, sonst Lücke.",
    ("Aussagen", "Kosinus AI-Aussage"): "Embedding des Wettbewerber-Chunks gegen jede Aussage der AI-Antwort, " + _COSINE + "; der höchste Wert.",
    ("Aussagen", "Priorität"): "Kosinus AI-Aussage minus Kosinus eigen, vier Stellen.",
}


def gap_explanation_frame(result: GapResult) -> pd.DataFrame:
    rows = [(sheet, column, meaning, GAP_COLUMN_CALCULATIONS.get((sheet, column), ""))
            for (sheet, column), meaning in GAP_COLUMN_EXPLANATIONS.items()]
    return _frame(rows, ["Blatt", "Spalte", "Bedeutung", "So ermittelt"], result.keyword)


def gap_to_excel(result: GapResult) -> bytes:
    buffer = io.BytesIO()
    frames = (sections_frame(result), statements_frame(result), _protocol_frame(result), gap_explanation_frame(result))
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in zip(SHEETS, frames):
            _excel_frame(frame).to_excel(writer, sheet_name=name, index=False)
    return buffer.getvalue()
