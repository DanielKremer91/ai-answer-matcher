"""Turn a Result into tables, a short summary, the chunking protocol and an Excel file."""

from __future__ import annotations

import io
import re
from typing import Optional, Sequence

import pandas as pd

from aam.match import MIN_STATEMENTS_FOR_QUARTILES, STATUS_COVERED, STATUS_GAP
from aam.models import OWN, SOURCE, ChunkStats, FailedUrl, Result, SegmentedAnswer

KEYWORD_COLUMN = "Keyword/Prompt"
SHEETS = ("Seiten", "Themen", "Aussagen", "Aussage x Seite", "Chunks", "Protokoll", "Erklärung")
MATRIX_MAX_PAGES = 15  # more pages than this do not fit a readable matrix
ORIGIN_LABELS = {OWN: "eigen", SOURCE: "Quelle"}
METHOD_LABELS = {"structure": "Struktur", "semantic": "semantisch"}
_SOURCE_COLUMNS = ["Beste Quell-URL", "Bester Quell-Chunk", "Score Quelle", "Delta"]
_ILLEGAL_XML_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_FORMULA_START = re.compile(r"^\s*[=+\-@]")
_EXCEL_CELL_LIMIT = 32767
_EXCEL_CUT_AT = 32750
_TRUNCATION_MARK = " … [gekürzt]"
MAX_EMPTY_URLS_SHOWN = 20  # per origin


def _frame(rows: list, columns: list[str], keyword: str) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=columns)
    df.insert(0, KEYWORD_COLUMN, keyword)
    return df


def _score(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value), 4)


def _de(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _count(n: int, singular: str, plural: str) -> str:
    """'1 Seite' / '2 Seiten'."""
    return f"{n} {singular if n == 1 else plural}"


def _words(value: float) -> str:
    """Rounded word count with its noun: '1 Wort', '12 Wörter'."""
    return _count(int(round(value)), "Wort", "Wörter")


def _excel_safe(value):
    """Make one cell safe for Excel: no illegal characters, no formulas, within the cell limit."""
    if not isinstance(value, str):
        return value
    value = _ILLEGAL_XML_CHARS.sub("", value)
    if len(value) > _EXCEL_CELL_LIMIT:
        value = value[:_EXCEL_CUT_AT] + _TRUNCATION_MARK
    if _FORMULA_START.match(value):
        value = "'" + value
    return value


def _excel_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy of the frame with every text cell made Excel-safe (numbers untouched)."""
    frame = frame.copy()
    for column in frame.columns:
        if frame[column].dtype == object or pd.api.types.is_string_dtype(frame[column]):
            frame[column] = frame[column].map(_excel_safe)
    return frame


WHOLE_VECTOR = "komplett"
AVERAGED_VECTOR = "Durchschnitt der Chunks"


def pages_frame(result: Result) -> pd.DataFrame:
    covered = "Aussagen abgedeckt" if result.match.has_sources else "Aussagen mit bestem Chunk"
    rows = [(p.rank, p.url, ORIGIN_LABELS[p.origin], _score(p.page_score), _score(p.best_chunk_score),
             p.covered, p.total, WHOLE_VECTOR if (p.origin, p.url) in result.page_vectors else AVERAGED_VECTOR)
            for p in result.match.pages]
    columns = ["Rang", "URL", "Herkunft", "Seitenscore", "Bester Chunk Score", covered, "Aussagen gesamt",
               "Seitenvektor"]
    return _frame(rows, columns, result.meta.keyword)


def overlong_pages(page_texts: dict, provider_name: str) -> list[str]:
    """German lines naming the pages whose whole text exceeds the provider's limit (shown before embedding)."""
    from aam.embeddings import DEFAULT_WHOLE_TEXT_WORDS, WHOLE_TEXT_WORD_LIMITS, fits_whole_text

    limit = WHOLE_TEXT_WORD_LIMITS.get(provider_name, DEFAULT_WHOLE_TEXT_WORDS)
    return [f"{url}: {len(text.split())} Wörter, über der Grenze von {limit} Wörtern für {provider_name} "
            f"→ Seitenvektor als {AVERAGED_VECTOR}"
            for (_, url), text in page_texts.items() if not fits_whole_text(provider_name, text)]


def _hit_cells(match_row, has_sources: bool) -> list:
    cells = [match_row.own.url, match_row.own.text, _score(match_row.own.score)]
    if has_sources:
        cells += [match_row.source.url, match_row.source.text, _score(match_row.source.score),
                  _score(match_row.delta)]
    return cells


def topics_frame(result: Result) -> pd.DataFrame:
    has_sources = result.match.has_sources
    labels = list(result.match.status_labels)
    rows = [[t.topic.label, len(t.topic.statements)] + _hit_cells(t, has_sources)
            + [t.counts.get(label, 0) for label in labels] for t in result.match.topics]
    columns = (["Thema", "Anzahl Aussagen", "Beste eigene URL", "Bester eigener Chunk", "Score"]
               + (_SOURCE_COLUMNS if has_sources else []) + labels)
    return _frame(rows, columns, result.meta.keyword)


def statements_frame(result: Result) -> pd.DataFrame:
    has_sources = result.match.has_sources
    rows = [[m.topic_label, m.statement.text, m.statement.words] + _hit_cells(m, has_sources)
            + ([_score(m.own_vs_source)] if has_sources else []) + [m.status]
            for m in result.match.statements]
    columns = (["Thema", "Aussage (AI-Chunk)", "Wortzahl", "Beste eigene URL", "Bester eigener Chunk", "Score"]
               + (_SOURCE_COLUMNS + ["Nähe zum Quell-Chunk"] if has_sources else []) + ["Status"])
    return _frame(rows, columns, result.meta.keyword)


def statement_pages_frame(result: Result) -> pd.DataFrame:
    """Every statement against the pages kept for it: best chunk of the page and fit of the whole page."""
    rows = [(r.topic_label, r.statement.text, r.rank, r.url, ORIGIN_LABELS[r.origin],
             _score(r.best_chunk.score), _score(r.page_score), r.best_chunk.text)
            for r in result.match.statement_pages]
    columns = ["Thema", "Aussage (AI-Chunk)", "Rang", "URL", "Herkunft", "Bester Chunk Score",
               "Seiten-Relevanz", "Bester Chunk"]
    return _frame(rows, columns, result.meta.keyword)


def statement_pages_matrix(result: Result, max_pages: int = MATRIX_MAX_PAGES) -> Optional[pd.DataFrame]:
    """Statements as rows, pages as columns, best chunk score as value; None when there are too many pages.

    Pages are ordered as in the page ranking. A page that is not among a statement's kept pages stays empty.
    """
    kept = {(r.url, r.origin) for r in result.match.statement_pages}
    pages = [(p.url, p.origin) for p in result.match.pages if (p.url, p.origin) in kept]
    if not pages or len(pages) > max_pages:
        return None
    header = {page: page[0] + (" (Quelle)" if page[1] == SOURCE else "") for page in pages}
    cells: dict[int, dict] = {}
    for r in result.match.statement_pages:
        row = cells.setdefault(r.statement.id, {"Thema": r.topic_label, "Aussage": r.statement.text})
        row[header[(r.url, r.origin)]] = _score(r.best_chunk.score)
    return pd.DataFrame(list(cells.values()), columns=["Thema", "Aussage"] + [header[page] for page in pages])


def chunks_frame(result: Result) -> pd.DataFrame:
    rows = [("AI-Antwort", "", s.id, s.text, s.words) for s in result.answer.statements]
    rows += [(ORIGIN_LABELS[c.origin], c.url, c.index, c.text, c.words)
             for c in list(result.own_chunks) + list(result.source_chunks)]
    return _frame(rows, ["Herkunft", "URL", "Chunk-Index", "Text", "Wortzahl"], result.meta.keyword)


def answer_frame(answer: SegmentedAnswer) -> pd.DataFrame:
    """Every AI chunk with its topic, shown before any embedding happens."""
    rows = [(topic.label, s.text, s.words) for topic in answer.topics for s in topic.statements]
    return pd.DataFrame(rows, columns=["Thema", "Aussage", "Wortzahl"])


def _stats_line(label: str, stats: ChunkStats) -> str:
    line = (f"{label}: {_count(stats.documents, 'Seite', 'Seiten')} → {_count(stats.chunks, 'Chunk', 'Chunks')}, "
            f"Median {_words(stats.median_words)} "
            f"(p10: {stats.p10_words:.0f}, p90: {stats.p90_words:.0f}), "
            f"{_count(stats.merged, 'Absatz', 'Absätze')} zusammengelegt, "
            f"{_count(stats.split, 'Absatz', 'Absätze')} geteilt")
    if stats.empty_urls:
        line += f", {_count(len(stats.empty_urls), 'Seite', 'Seiten')} ohne Text"
    if not stats.merge_short:
        line += ", kurze Absätze nicht zusammengelegt"
    if not stats.split_long:
        line += ", lange Absätze nicht geteilt"
    return line


def _prechunked_line(stats: ChunkStats) -> str:
    """Merge and split counts are unknown for an uploaded export, so they are left out."""
    return (f"Eigener Content (aus Chunk-Embedding-Export): {_count(stats.documents, 'Seite', 'Seiten')} → "
            f"{_count(stats.chunks, 'Chunk', 'Chunks')}, Median {_words(stats.median_words)} "
            f"(p10: {stats.p10_words:.0f}, p90: {stats.p90_words:.0f})")


def _empty_lines(origin: str, urls: Sequence[str]) -> list[str]:
    lines = [f"Ohne Text ({ORIGIN_LABELS[origin]}): {url}" for url in urls[:MAX_EMPTY_URLS_SHOWN]]
    if len(urls) > MAX_EMPTY_URLS_SHOWN:
        lines.append(f"… und {len(urls) - MAX_EMPTY_URLS_SHOWN} weitere")
    return lines


def protocol_lines(answer: SegmentedAnswer, own_stats: ChunkStats, source_stats: Optional[ChunkStats],
                   failed: Sequence[FailedUrl], *, prechunked: bool = False,
                   source_target_words: Optional[int] = None) -> list[str]:
    lines = [
        f"AI-Antwort: {_count(len(answer.topics), 'Thema', 'Themen')}, "
        f"{_count(len(answer.statements), 'Aussage', 'Aussagen')}, "
        f"Median {_words(answer.median_words)}"
        + (f" × {_de(answer.content_factor, 1)}" if answer.content_factor != 1.0 else "")
        + f" → Zielgröße {_words(answer.target_words)} "
        f"(Segmentierung: {METHOD_LABELS[answer.method]})",
        _prechunked_line(own_stats) if prechunked else _stats_line("Eigener Content", own_stats),
    ]
    if source_stats is None:
        lines.append("Quellen: keine. Der Status zeigt die relative Stärke innerhalb des eigenen Contents.")
    else:
        line = _stats_line("Quellen", source_stats)
        if source_target_words and source_target_words != answer.target_words:
            line += f", geschnitten mit Zielgröße {_words(source_target_words)} wie der hochgeladene Export"
        lines.append(line)
    lines += _empty_lines(OWN, own_stats.empty_urls)
    if source_stats is not None:
        lines += _empty_lines(SOURCE, source_stats.empty_urls)
    lines += [f"Nicht geladen ({ORIGIN_LABELS[f.origin]}): {f.url} ({f.reason})" for f in failed]
    return lines


def protocol_frame(result: Result) -> pd.DataFrame:
    meta = result.meta
    rows = [
        ("Datum", meta.date),
        ("Embedding-Anbieter", meta.provider),
        ("Modell", meta.model),
        ("Dimension", meta.dimension),
        ("Zielgröße (Wörter)", meta.target_words),
        ("Median Aussagenlänge (Wörter)", meta.median_words),
        ("Segmentierung", METHOD_LABELS[meta.method]),
        ("Schwelle abgedeckt", round(meta.covered_delta, 4)),
        ("Schwelle schwach", round(meta.weak_delta, 4)),
        ("Seitenvektoren", f"{_count(meta.whole_pages, 'Seite', 'Seiten')} komplett eingebettet, "
                           f"{meta.mean_pages} als Durchschnitt ihrer Chunks"),
    ]
    rows += [("Chunking", line) for line in protocol_lines(
        result.answer, result.own_stats, result.source_stats, meta.failed, prechunked=meta.prechunked,
        source_target_words=meta.source_target_words or None)]
    rows += [("Hinweis", warning) for warning in meta.warnings]
    return _frame(rows, ["Angabe", "Wert"], meta.keyword)


def summary(result: Result) -> str:
    matched = result.match
    top = next(p for p in matched.pages if p.origin == OWN)
    text = (f"Relevanteste eigene Seite: {top.url} (Seitenscore {_de(top.page_score)}, "
            f"Rang {top.rank} von {len(matched.pages)}).")
    if not matched.has_sources:
        return (f"{text} Sie stellt bei {top.covered} von {top.total} Aussagen den besten Chunk der Domain. "
                "Ohne Quellen gibt es keinen Benchmark: der Status zeigt nur die relative Stärke "
                "innerhalb des eigenen Contents."
                + (f" Für einen Status sind mindestens {MIN_STATEMENTS_FOR_QUARTILES} Aussagen oder Quell-URLs nötig."
                   if len(matched.statements) < MIN_STATEMENTS_FOR_QUARTILES else ""))
    covered = sum(m.status == STATUS_COVERED for m in matched.statements)
    text += (f" Über die ganze Domain sind {covered} von {top.total} Aussagen abgedeckt, "
             f"diese Seite allein deckt {top.covered} ab.")
    gaps = [(t.topic.label, t.counts[STATUS_GAP]) for t in matched.topics if t.counts.get(STATUS_GAP)]
    if not gaps:
        return f"{text} Keine Lücken."
    parts = [f"{count} {'Aussage' if count == 1 else 'Aussagen'} im Thema „{label}“" for label, count in gaps]
    return f"{text} Lücken: {', '.join(parts)}."


# Column explanations for the sheet "Erklärung". "alle" marks columns that appear on every sheet.
_SCORE_NOTE = ("Kosinus-Ähnlichkeit der Embeddings, 0 bis 1. Absolute Werte hängen vom Modell ab; "
               "vergleiche nur Werte aus demselben Lauf.")
COLUMN_EXPLANATIONS = {
    ("alle", "Keyword/Prompt"): "Das Keyword oder der Prompt dieses Laufs, damit sich Exporte mehrerer Läufe zuordnen lassen.",
    ("Seiten", "Rang"): "Platz der Seite im Ranking nach Seitenscore; eigene Seiten und Quellen gemischt.",
    ("Seiten", "URL"): "Adresse der Seite; bei eingefügtem Text die Bezeichnung, die du eingetragen hast.",
    ("Seiten", "Herkunft"): "„eigen“ = dein Content, „Quelle“ = zitierte Quelle oder Wettbewerber.",
    ("Seiten", "Seitenscore"): "Ganze AI-Antwort gegen ganze Seite: " + _SCORE_NOTE + " Beantwortet „Passt die Seite insgesamt zum Thema?“.",
    ("Seiten", "Bester Chunk Score"): "Höchster Kosinus irgendeines Chunks dieser Seite zu irgendeiner Aussage der AI-Antwort: die beste einzelne Passage.",
    ("Seiten", "Aussagen abgedeckt"): "Anzahl Aussagen, bei denen der beste Chunk dieser Seite höchstens die Schwelle „abgedeckt“ unter dem besten Quell-Chunk liegt.",
    ("Seiten", "Aussagen mit bestem Chunk"): "Ohne Quellen: Anzahl Aussagen, bei denen diese Seite den besten Chunk der ganzen Domain stellt.",
    ("Seiten", "Aussagen gesamt"): "Anzahl Aussagen, in die die AI-Antwort zerlegt wurde.",
    ("Seiten", "Seitenvektor"): "„komplett“ = der ganze Seitentext wurde als ein Embedding berechnet; „Durchschnitt der Chunks“ = Text zu lang für das Modell oder aus einem Chunk-Export, dann wortgewichteter Mittelwert der Chunk-Vektoren.",
    ("Themen", "Thema"): "Themengebiet der AI-Antwort, erkannt an Überschriften oder Einleitungswörtern; „Einleitung“ = Text vor der ersten Überschrift.",
    ("Themen", "Anzahl Aussagen"): "Wie viele Aussagen der AI-Antwort zu diesem Thema gehören.",
    ("Themen", "Beste eigene URL"): "Deine Seite mit dem Chunk, der dem Themen-Vektor (Mittelwert seiner Aussagen) am nächsten liegt.",
    ("Themen", "Bester eigener Chunk"): "Text dieses Chunks.",
    ("Themen", "Score"): "Kosinus zwischen Themen-Vektor und deinem besten Chunk. " + _SCORE_NOTE,
    ("Themen", "Beste Quell-URL"): "Quelle mit dem Chunk, der dem Themen-Vektor am nächsten liegt.",
    ("Themen", "Bester Quell-Chunk"): "Text dieses Quell-Chunks.",
    ("Themen", "Score Quelle"): "Kosinus zwischen Themen-Vektor und bestem Quell-Chunk.",
    ("Themen", "Delta"): "Score minus Score Quelle: negativ heißt, die Quelle liegt näher am Thema als du.",
    ("Themen", "abgedeckt"): "Anzahl Aussagen dieses Themas mit Status „abgedeckt“ (Delta ≥ −Schwelle abgedeckt).",
    ("Themen", "schwach"): "Anzahl Aussagen dieses Themas mit Status „schwach“ (Delta zwischen den beiden Schwellen); ohne Quellen: unterstes Viertel der eigenen Scores.",
    ("Themen", "Lücke"): "Anzahl Aussagen dieses Themas mit Status „Lücke“ (Delta unter −Schwelle schwach).",
    ("Themen", "stark"): "Ohne Quellen: Anzahl Aussagen im obersten Viertel der eigenen Scores.",
    ("Themen", "über Durchschnitt"): "Ohne Quellen: Anzahl Aussagen zwischen Median und oberstem Viertel.",
    ("Themen", "unter Durchschnitt"): "Ohne Quellen: Anzahl Aussagen zwischen unterstem Viertel und Median.",
    ("Aussagen", "Thema"): "Thema, zu dem die Aussage gehört.",
    ("Aussagen", "Aussage (AI-Chunk)"): "Eine Aussage der AI-Antwort: eine Zeile, ein Bullet oder ein Satz; die Einheit, die Google aus einer Quelle zieht.",
    ("Aussagen", "Wortzahl"): "Länge der Aussage in Wörtern; der Median aller Aussagen bestimmt die Chunk-Größe.",
    ("Aussagen", "Beste eigene URL"): "Deine Seite mit dem Chunk, der dieser Aussage am nächsten liegt.",
    ("Aussagen", "Bester eigener Chunk"): "Text dieses Chunks.",
    ("Aussagen", "Score"): "Kosinus zwischen Aussage und deinem besten Chunk. " + _SCORE_NOTE,
    ("Aussagen", "Beste Quell-URL"): "Quelle mit dem Chunk, der dieser Aussage am nächsten liegt; vermutlich die Stelle, aus der die AI die Aussage gezogen hat.",
    ("Aussagen", "Bester Quell-Chunk"): "Text dieses Quell-Chunks.",
    ("Aussagen", "Score Quelle"): "Kosinus zwischen Aussage und bestem Quell-Chunk.",
    ("Aussagen", "Delta"): "Score minus Score Quelle: wie weit dein bester Absatz hinter dem besten Absatz der Quellen liegt.",
    ("Aussagen", "Nähe zum Quell-Chunk"): "Kosinus zwischen deinem besten Chunk und dem besten Quell-Chunk derselben Aussage: sagen beide dasselbe?",
    ("Aussagen", "Status"): "Mit Quellen: „abgedeckt“, „schwach“ oder „Lücke“ nach den Schwellen aus der Seitenleiste. Ohne Quellen: „stark“, „über Durchschnitt“, „unter Durchschnitt“, „schwach“ nach Vierteln der eigenen Scores; leer bei weniger als 4 Aussagen.",
    ("Aussage x Seite", "Thema"): "Thema, zu dem die Aussage gehört.",
    ("Aussage x Seite", "Aussage (AI-Chunk)"): "Die Aussage der AI-Antwort.",
    ("Aussage x Seite", "Rang"): "Platz der Seite für diese Aussage, nach Bester Chunk Score; pro Aussage die zehn besten eigenen Seiten und alle Quellen.",
    ("Aussage x Seite", "URL"): "Adresse der Seite, die mit dieser Aussage verglichen wurde.",
    ("Aussage x Seite", "Herkunft"): "„eigen“ oder „Quelle“.",
    ("Aussage x Seite", "Bester Chunk Score"): "Höchster Kosinus eines Chunks dieser Seite zur Aussage: Gibt es auf der Seite eine passende Passage?",
    ("Aussage x Seite", "Seiten-Relevanz"): "Kosinus zwischen Aussage und Seitenvektor: Passt die Seite als Ganzes zum Thema der Aussage? Relatives Signal, Seiten mit vielen Themen schneiden schlechter ab.",
    ("Aussage x Seite", "Bester Chunk"): "Text des besten Chunks dieser Seite für die Aussage.",
    ("Chunks", "Herkunft"): "„AI-Antwort“ = Aussage der Antwort, „eigen“ = Chunk deines Contents, „Quelle“ = Chunk einer Quelle.",
    ("Chunks", "URL"): "Seite, aus der der Chunk stammt; leer bei Aussagen der AI-Antwort.",
    ("Chunks", "Chunk-Index"): "Laufende Nummer des Chunks innerhalb der Seite, in Lesereihenfolge.",
    ("Chunks", "Text"): "Der Text, der eingebettet wurde.",
    ("Chunks", "Wortzahl"): "Länge des Chunks in Wörtern.",
    ("Protokoll", "Angabe"): "Name der Einstellung oder des Protokolleintrags.",
    ("Protokoll", "Wert"): "Datum, Anbieter, Modell, Dimension, Zielgröße, Schwellen, Chunk-Statistik, nicht geladene URLs und Hinweise dieses Laufs.",
}


def explanation_frame(result: Result) -> pd.DataFrame:
    """One row per column of the other sheets: which sheet, which column, what it means."""
    rows = [(sheet, column, meaning) for (sheet, column), meaning in COLUMN_EXPLANATIONS.items()]
    return _frame(rows, ["Blatt", "Spalte", "Bedeutung"], result.meta.keyword)


def to_excel(result: Result) -> bytes:
    frames = (pages_frame(result), topics_frame(result), statements_frame(result),
              statement_pages_frame(result), chunks_frame(result), protocol_frame(result),
              explanation_frame(result))
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in zip(SHEETS, frames):
            _excel_frame(frame).to_excel(writer, sheet_name=name, index=False)
    return buffer.getvalue()
