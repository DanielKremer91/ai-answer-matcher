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
_TOPIC_SOURCE_COLUMNS = ["Beste Quell-URL", "Beste Quell-Passage", "Score Quelle", "Delta"]
TOPIC_PASSAGE_FROM_EXPORT = "Chunks aus Export"
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
            + [f"Fenster ({t.window_words} Wörter)" if t.method == "Fenster" else TOPIC_PASSAGE_FROM_EXPORT]
            + [t.counts.get(label, 0) for label in labels] for t in result.match.topics]
    columns = (["Thema", "Anzahl Aussagen", "Beste eigene URL", "Beste eigene Passage", "Score eigen"]
               + (_TOPIC_SOURCE_COLUMNS if has_sources else []) + ["Passagengröße"] + labels)
    return _frame(rows, columns, result.meta.keyword)


def statements_frame(result: Result) -> pd.DataFrame:
    has_sources = result.match.has_sources
    rows = [[m.topic_label, m.statement.text, m.statement.words] + _hit_cells(m, has_sources)
            + ([_score(m.own_vs_source)] if has_sources else []) + [m.status]
            for m in result.match.statements]
    columns = (["Thema", "Aussage (AI-Chunk)", "Wortzahl", "Beste eigene URL", "Bester eigener Chunk", "Score eigen"]
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
        ("Antwortvektor", "AI-Antwort komplett eingebettet" if meta.whole_answer
                          else "Durchschnitt der Aussagen-Vektoren"),
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
_SCORE_NOTE = ("Ähnlichkeit von 0 bis 1 (Kosinus der Embeddings). Die Zahl hängt vom Modell ab; "
               "vergleiche nur Werte aus demselben Lauf miteinander.")
COLUMN_EXPLANATIONS = {
    ("alle", "Keyword/Prompt"): "Das Keyword oder der Prompt dieses Laufs. So kannst du Exporte mehrerer Läufe auseinanderhalten.",
    # ---- Seiten
    ("Seiten", "Rang"): "Platz der Seite nach Seitenscore. Eigene Seiten und Quellen stehen in einer Liste.",
    ("Seiten", "URL"): "Adresse der Seite. Bei eingefügtem Text steht hier die Bezeichnung, die du eingetragen hast.",
    ("Seiten", "Herkunft"): "„eigen“ = deine Seite. „Quelle“ = zitierte Quelle oder Wettbewerber.",
    ("Seiten", "Seitenscore"): "Wie gut passt die ganze Seite zur ganzen AI-Antwort? " + _SCORE_NOTE,
    ("Seiten", "Bester Chunk Score"): "Die beste einzelne Passage der Seite: der höchste Wert, den irgendein Absatz dieser Seite zu irgendeiner Aussage der AI-Antwort erreicht.",
    ("Seiten", "Aussagen abgedeckt"): "Wie viele Aussagen der AI-Antwort diese Seite abdeckt. Abgedeckt heißt: Ihr bester Absatz liegt höchstens um die Schwelle „abgedeckt“ unter dem besten Absatz der Quellen.",
    ("Seiten", "Aussagen mit bestem Chunk"): "Nur ohne Quellen: Bei wie vielen Aussagen diese Seite den besten Absatz deiner ganzen Domain hat.",
    ("Seiten", "Aussagen gesamt"): "In wie viele Aussagen die AI-Antwort zerlegt wurde.",
    ("Seiten", "Seitenvektor"): "„komplett“: Der ganze Seitentext wurde als ein Stück eingebettet. „Durchschnitt der Chunks“: Der Text war zu lang für das Modell (oder kam aus einem Chunk-Export), dann wurde aus den Absätzen ein Mittelwert gebildet.",
    # ---- Themen
    ("Themen", "Thema"): "Ein Abschnitt der AI-Antwort, erkannt an einer Überschrift oder einem fetten Einleitungswort. „Einleitung“ = Text vor der ersten Überschrift.",
    ("Themen", "Anzahl Aussagen"): "Wie viele Aussagen der AI-Antwort zu diesem Thema gehören.",
    ("Themen", "Beste eigene URL"): "Deine Seite mit der Passage, die dem Thema am nächsten kommt.",
    ("Themen", "Beste eigene Passage"): "Diese Passage im Wortlaut: ein Fenster in Themengröße aus deiner Seite, nicht ein Chunk aus Schritt 4.",
    ("Themen", "Score eigen"): "Wie nah deine beste Passage am Thema als Ganzem liegt. " + _SCORE_NOTE,
    ("Themen", "Beste Quell-URL"): "Die Quelle mit der Passage, die dem Thema am nächsten kommt.",
    ("Themen", "Beste Quell-Passage"): "Diese Passage im Wortlaut, ein Fenster in Themengröße aus der Quelle.",
    ("Themen", "Score Quelle"): "Wie nah die beste Passage der Quellen am Thema liegt.",
    ("Themen", "Delta"): "Score eigen minus Score Quelle. Unter null heißt: Die Quelle hat die bessere Passage zum Thema.",
    ("Themen", "Passagengröße"): "„Fenster (n Wörter)“: Für dieses Thema wurden deine Seiten und die Quellen in Passagen von etwa n Wörtern geschnitten (Wortzahl des Themas mal Chunk-Faktor); die beste Passage zählt. „Chunks aus Export“: nur bei wiederverwendeten Chunk-Exporten ohne Seitentext, dort zählt der beste einzelne Chunk.",
    ("Themen", "abgedeckt"): "Wie viele Aussagen dieses Themas den Status „abgedeckt“ haben.",
    ("Themen", "schwach"): "Wie viele Aussagen dieses Themas den Status „schwach“ haben.",
    ("Themen", "Lücke"): "Wie viele Aussagen dieses Themas den Status „Lücke“ haben.",
    ("Themen", "stark"): "Nur ohne Quellen: Aussagen im besten Viertel deiner eigenen Werte.",
    ("Themen", "über Durchschnitt"): "Nur ohne Quellen: Aussagen zwischen Mitte und bestem Viertel.",
    ("Themen", "unter Durchschnitt"): "Nur ohne Quellen: Aussagen zwischen schlechtestem Viertel und Mitte.",
    # ---- Aussagen
    ("Aussagen", "Thema"): "Das Thema, zu dem die Aussage gehört.",
    ("Aussagen", "Aussage (AI-Chunk)"): "Eine einzelne Aussage der AI-Antwort: eine Zeile, ein Bullet oder ein Satz. Ein Bullet mit mehreren Sätzen wird in Sätze geteilt, sein Label („Die Fahrt:“) bleibt an jedem Satz; Sätze, die nur auf den Satz davor verweisen („Das spendet Trost.“), bleiben an ihm hängen. Das ist die Einheit, die Google aus einer Quelle zieht.",
    ("Aussagen", "Wortzahl"): "Länge der Aussage in Wörtern. Der Median aller Aussagen bestimmt, wie groß deine Chunks geschnitten werden.",
    ("Aussagen", "Beste eigene URL"): "Deine Seite mit dem Absatz, der dieser Aussage am nächsten kommt.",
    ("Aussagen", "Bester eigener Chunk"): "Dieser Absatz im Wortlaut.",
    ("Aussagen", "Score eigen"): "Wie nah dein bester Absatz an der Aussage liegt. " + _SCORE_NOTE,
    ("Aussagen", "Beste Quell-URL"): "Die Quelle mit dem Absatz, der der Aussage am nächsten kommt. Vermutlich die Stelle, aus der die AI die Aussage gezogen hat.",
    ("Aussagen", "Bester Quell-Chunk"): "Dieser Absatz im Wortlaut.",
    ("Aussagen", "Score Quelle"): "Wie nah der beste Absatz der Quellen an der Aussage liegt.",
    ("Aussagen", "Delta"): "Score eigen minus Score Quelle: Wie weit dein bester Absatz hinter dem besten Absatz der Quellen liegt. Unter null heißt: Die Quelle ist näher dran.",
    ("Aussagen", "Nähe zum Quell-Chunk"): "Wie ähnlich dein bester Absatz und der beste Quell-Absatz einander sind. Hoch heißt: Beide sagen dasselbe.",
    ("Aussagen", "Status"): "Mit Quellen: „abgedeckt“ (Delta über −Schwelle abgedeckt), „schwach“ (zwischen den Schwellen), „Lücke“ (unter −Schwelle schwach). Ohne Quellen: „stark“, „über Durchschnitt“, „unter Durchschnitt“, „schwach“ je nach Viertel deiner eigenen Werte; leer bei weniger als 4 Aussagen.",
    # ---- Aussage x Seite
    ("Aussage x Seite", "Thema"): "Das Thema, zu dem die Aussage gehört.",
    ("Aussage x Seite", "Aussage (AI-Chunk)"): "Die Aussage der AI-Antwort.",
    ("Aussage x Seite", "Rang"): "Platz der Seite für diese Aussage, nach Bester Chunk Score. Pro Aussage stehen die zehn besten eigenen Seiten und alle Quellen in der Liste.",
    ("Aussage x Seite", "URL"): "Die Seite, die hier mit der Aussage verglichen wird.",
    ("Aussage x Seite", "Herkunft"): "„eigen“ = deine Seite, „Quelle“ = Quelle oder Wettbewerber.",
    ("Aussage x Seite", "Bester Chunk Score"): "Gibt es auf dieser Seite einen Absatz, der zur Aussage passt? Der Wert des besten Absatzes.",
    ("Aussage x Seite", "Seiten-Relevanz"): "Passt die Seite als Ganzes zum Thema der Aussage? Nur zum Sortieren geeignet: Seiten mit vielen Themen bekommen hier niedrigere Werte.",
    ("Aussage x Seite", "Bester Chunk"): "Der beste Absatz dieser Seite im Wortlaut.",
    # ---- Chunks
    ("Chunks", "Herkunft"): "„AI-Antwort“ = eine Aussage der Antwort. „eigen“ = ein Absatz deines Contents. „Quelle“ = ein Absatz einer Quelle.",
    ("Chunks", "URL"): "Die Seite, aus der der Absatz stammt. Leer bei Aussagen der AI-Antwort.",
    ("Chunks", "Chunk-Index"): "Laufende Nummer des Absatzes innerhalb der Seite, in Lesereihenfolge.",
    ("Chunks", "Text"): "Der Text, der eingebettet wurde.",
    ("Chunks", "Wortzahl"): "Länge in Wörtern.",
    # ---- Protokoll
    ("Protokoll", "Angabe"): "Name der Einstellung oder des Eintrags.",
    ("Protokoll", "Wert"): "Datum, Anbieter, Modell, Dimension, Zielgröße, Schwellen, Chunk-Statistik, nicht geladene URLs und Hinweise dieses Laufs.",
}

# How each measured value is computed. "Embedding" = the number vector the model makes from a text;
# "Kosinus" = the similarity of two such vectors, 1 = identical direction. Columns that are not
# measured (names, texts, counts of rows) have no entry and get an empty cell in the legend.
_COSINE = "Kosinus-Ähnlichkeit der beiden Embeddings"
COLUMN_CALCULATIONS = {
    ("Seiten", "Rang"): "Alle Seiten nach Seitenscore absteigend sortiert, Platz 1 = höchster Wert.",
    ("Seiten", "Seitenscore"): "Embedding der ganzen AI-Antwort (als ein Text) gegen das Embedding der ganzen Seite (Spalte Seitenvektor sagt, wie der entstand). " + _COSINE + ".",
    ("Seiten", "Bester Chunk Score"): "Jede Aussage der AI-Antwort einzeln eingebettet, jeder Chunk der Seite einzeln eingebettet. Kosinus für jedes Paar Aussage × Chunk, der höchste Wert zählt.",
    ("Seiten", "Aussagen abgedeckt"): "Pro Aussage: bester Chunk dieser Seite minus bester Chunk aller Quellen. Liegt die Differenz über −Schwelle „abgedeckt“, zählt die Aussage.",
    ("Seiten", "Aussagen mit bestem Chunk"): "Pro Aussage wird der Chunk mit dem höchsten Kosinus über alle eigenen Seiten gesucht; gezählt wird, wie oft er auf dieser Seite liegt.",
    ("Seiten", "Seitenvektor"): "Passt der Seitentext in die Grenze des Anbieters, wird er als ein Text eingebettet. Sonst: Mittelwert der Chunk-Embeddings, jeder Chunk mit seiner Wortzahl gewichtet, dann auf Länge 1 normiert.",
    ("Themen", "Score eigen"): "Themenvektor = Mittelwert der Embeddings aller Aussagen des Themas, mit Wortzahl gewichtet. Deine Seiten werden in Fenster von Themenlänge × Chunk-Faktor geschnitten (Satzgrenzen, ein Satz Überlappung), jedes Fenster eingebettet. Höchster Kosinus zwischen Themenvektor und einem Fenster.",
    ("Themen", "Score Quelle"): "Wie Score eigen, mit den Fenstern der Quellen.",
    ("Themen", "Delta"): "Score eigen minus Score Quelle.",
    ("Themen", "Passagengröße"): "Fenstergröße = Wortzahl des Themas × Chunk-Faktor, begrenzt auf 15 bis 300 Wörter. Fenster werden über die zehn eigenen Seiten mit dem höchsten Seitenvektor-Kosinus zum Thema und über alle Quellen gelegt.",
    ("Themen", "abgedeckt"): "Anzahl Aussagen des Themas mit diesem Status (Blatt Aussagen, Spalte Status).",
    ("Themen", "schwach"): "Anzahl Aussagen des Themas mit diesem Status.",
    ("Themen", "Lücke"): "Anzahl Aussagen des Themas mit diesem Status.",
    ("Themen", "stark"): "Anzahl Aussagen des Themas mit diesem Status.",
    ("Themen", "über Durchschnitt"): "Anzahl Aussagen des Themas mit diesem Status.",
    ("Themen", "unter Durchschnitt"): "Anzahl Aussagen des Themas mit diesem Status.",
    ("Aussagen", "Wortzahl"): "Wörter der Aussage, getrennt an Leerzeichen.",
    ("Aussagen", "Score eigen"): "Embedding der Aussage gegen das Embedding jedes eigenen Chunks, " + _COSINE + "; der höchste Wert zählt.",
    ("Aussagen", "Score Quelle"): "Wie Score eigen, mit den Chunks aller Quellen.",
    ("Aussagen", "Delta"): "Score eigen minus Score Quelle.",
    ("Aussagen", "Nähe zum Quell-Chunk"): "Embedding deines besten Chunks gegen das Embedding des besten Quell-Chunks, " + _COSINE + ".",
    ("Aussagen", "Status"): "Mit Quellen aus dem Delta und den beiden Schwellen in der Seitenleiste. Ohne Quellen: Alle Score-Werte sortiert und in Viertel geteilt; oberstes Viertel „stark“, unterstes „schwach“.",
    ("Aussage x Seite", "Rang"): "Pro Aussage alle Seiten nach Bester Chunk Score absteigend sortiert.",
    ("Aussage x Seite", "Bester Chunk Score"): "Embedding der Aussage gegen jeden Chunk dieser Seite, " + _COSINE + "; der höchste Wert.",
    ("Aussage x Seite", "Seiten-Relevanz"): "Embedding der Aussage gegen den Seitenvektor dieser Seite (ganze Seite), " + _COSINE + ".",
    ("Chunks", "Chunk-Index"): "Zählt ab 0 je Seite in der Reihenfolge, in der die Chunks im Text stehen.",
    ("Chunks", "Wortzahl"): "Wörter des Chunks, getrennt an Leerzeichen.",
}


def explanation_frame(result: Result) -> pd.DataFrame:
    """One row per column of the other sheets: sheet, column, what it means, how a measured value is computed."""
    rows = [(sheet, column, meaning, COLUMN_CALCULATIONS.get((sheet, column), ""))
            for (sheet, column), meaning in COLUMN_EXPLANATIONS.items()]
    return _frame(rows, ["Blatt", "Spalte", "Bedeutung", "So ermittelt"], result.meta.keyword)


def to_excel(result: Result) -> bytes:
    frames = (pages_frame(result), topics_frame(result), statements_frame(result),
              statement_pages_frame(result), chunks_frame(result), protocol_frame(result),
              explanation_frame(result))
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in zip(SHEETS, frames):
            _excel_frame(frame).to_excel(writer, sheet_name=name, index=False)
    return buffer.getvalue()
