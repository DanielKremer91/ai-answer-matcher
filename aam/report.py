"""Turn a Result into tables, a short summary, the chunking protocol and an Excel file."""

from __future__ import annotations

import io
from typing import Optional, Sequence

import pandas as pd

from aam.match import STATUS_COVERED, STATUS_GAP
from aam.models import OWN, SOURCE, ChunkStats, FailedUrl, Result, SegmentedAnswer

KEYWORD_COLUMN = "Keyword/Prompt"
SHEETS = ("Seiten", "Themen", "Aussagen", "Chunks", "Protokoll")
ORIGIN_LABELS = {OWN: "eigen", SOURCE: "Quelle"}
METHOD_LABELS = {"structure": "Struktur", "semantic": "semantisch"}
_SOURCE_COLUMNS = ["Beste Quell-URL", "Bester Quell-Chunk", "Score Quelle", "Delta"]


def _frame(rows: list, columns: list[str], keyword: str) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=columns)
    df.insert(0, KEYWORD_COLUMN, keyword)
    return df


def _score(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value), 4)


def _de(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def pages_frame(result: Result) -> pd.DataFrame:
    covered = "Aussagen abgedeckt" if result.match.has_sources else "Aussagen mit bestem Chunk"
    rows = [(p.rank, p.url, ORIGIN_LABELS[p.origin], _score(p.page_score), _score(p.best_chunk_score),
             p.covered, p.total) for p in result.match.pages]
    columns = ["Rang", "URL", "Herkunft", "Seitenscore", "Bester Chunk Score", covered, "Aussagen gesamt"]
    return _frame(rows, columns, result.meta.keyword)


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
    rows = [[m.topic_label, m.statement.text, m.statement.words] + _hit_cells(m, has_sources) + [m.status]
            for m in result.match.statements]
    columns = (["Thema", "Aussage (AI-Chunk)", "Wortzahl", "Beste eigene URL", "Bester eigener Chunk", "Score"]
               + (_SOURCE_COLUMNS if has_sources else []) + ["Status"])
    return _frame(rows, columns, result.meta.keyword)


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
    line = (f"{label}: {stats.documents} Seiten → {stats.chunks} Chunks, Median {stats.median_words:.0f} Wörter "
            f"(p10: {stats.p10_words:.0f}, p90: {stats.p90_words:.0f}), "
            f"{stats.merged} Absätze zusammengelegt, {stats.split} Absätze geteilt")
    if stats.empty_urls:
        line += f", {len(stats.empty_urls)} Seiten ohne Text"
    return line


def protocol_lines(answer: SegmentedAnswer, own_stats: ChunkStats, source_stats: Optional[ChunkStats],
                   failed: Sequence[FailedUrl], *, prechunked: bool = False) -> list[str]:
    lines = [
        f"AI-Antwort: {len(answer.topics)} Themen, {len(answer.statements)} Aussagen, "
        f"Median {answer.median_words:.0f} Wörter → Zielgröße {answer.target_words} Wörter "
        f"(Segmentierung: {METHOD_LABELS[answer.method]})",
        _stats_line("Eigener Content (aus Chunk-Embedding-Export)" if prechunked else "Eigener Content", own_stats),
    ]
    if source_stats is None:
        lines.append("Quellen: keine. Der Status zeigt die relative Stärke innerhalb des eigenen Contents.")
    else:
        lines.append(_stats_line("Quellen", source_stats))
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
        ("Schwelle abgedeckt", meta.covered_delta),
        ("Schwelle schwach", meta.weak_delta),
    ]
    rows += [("Chunking", line) for line in protocol_lines(
        result.answer, result.own_stats, result.source_stats, meta.failed, prechunked=meta.prechunked)]
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
                "innerhalb des eigenen Contents.")
    covered = sum(m.status == STATUS_COVERED for m in matched.statements)
    text += (f" Über die ganze Domain sind {covered} von {top.total} Aussagen abgedeckt, "
             f"diese Seite allein deckt {top.covered} ab.")
    gaps = [(t.topic.label, t.counts[STATUS_GAP]) for t in matched.topics if t.counts.get(STATUS_GAP)]
    if not gaps:
        return f"{text} Keine Lücken."
    parts = [f"{count} {'Aussage' if count == 1 else 'Aussagen'} im Thema „{label}“" for label, count in gaps]
    return f"{text} Lücken: {', '.join(parts)}."


def to_excel(result: Result) -> bytes:
    frames = (pages_frame(result), topics_frame(result), statements_frame(result), chunks_frame(result),
              protocol_frame(result))
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in zip(SHEETS, frames):
            frame.to_excel(writer, sheet_name=name, index=False)
    return buffer.getvalue()
