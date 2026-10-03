import io
from dataclasses import replace

import openpyxl
import pandas as pd

from aam.match import QUARTILE_LABELS
from aam.models import SOURCE, Chunk, ChunkStats, FailedUrl, SegmentedAnswer, Statement, Topic
from aam.report import (KEYWORD_COLUMN, SHEETS, answer_frame, chunks_frame, pages_frame, protocol_frame,
                        protocol_lines, statements_frame, summary, to_excel, topics_frame)
from tests.fakes import sample_result

KEYWORD = "dachsanierung kosten"


def test_excel_has_five_sheets_and_each_starts_with_the_keyword():
    sheets = pd.read_excel(io.BytesIO(to_excel(sample_result())), sheet_name=None)
    assert tuple(sheets) == SHEETS == ("Seiten", "Themen", "Aussagen", "Chunks", "Protokoll")
    for name, df in sheets.items():
        assert df.columns[0] == KEYWORD_COLUMN == "Keyword/Prompt", name
        assert set(df[KEYWORD_COLUMN]) == {KEYWORD}, name


def test_pages_frame_with_sources():
    df = pages_frame(sample_result())
    assert list(df.columns) == ["Keyword/Prompt", "Rang", "URL", "Herkunft", "Seitenscore", "Bester Chunk Score",
                                "Aussagen abgedeckt", "Aussagen gesamt"]
    assert list(df["Rang"]) == [1, 2, 3]
    assert set(df["Herkunft"]) == {"eigen", "Quelle"}
    assert set(df["Aussagen gesamt"]) == {5}
    assert all(round(v, 4) == v for v in df["Seitenscore"])


def test_topics_frame_with_sources():
    df = topics_frame(sample_result())
    assert list(df.columns) == ["Keyword/Prompt", "Thema", "Anzahl Aussagen", "Beste eigene URL",
                                "Bester eigener Chunk", "Score", "Beste Quell-URL", "Bester Quell-Chunk",
                                "Score Quelle", "Delta", "abgedeckt", "schwach", "Lücke"]
    assert list(df["Thema"]) == ["Einleitung", "Kosten im Überblick", "Förderung"]
    row = df[df["Thema"] == "Förderung"].iloc[0]
    assert (row["Anzahl Aussagen"], row["Lücke"], row["abgedeckt"]) == (2, 2, 0)
    assert row["Beste Quell-URL"] == "https://quelle.de/foerderung"


def test_statements_frame_with_sources():
    df = statements_frame(sample_result())
    assert list(df.columns) == ["Keyword/Prompt", "Thema", "Aussage (AI-Chunk)", "Wortzahl", "Beste eigene URL",
                                "Bester eigener Chunk", "Score", "Beste Quell-URL", "Bester Quell-Chunk",
                                "Score Quelle", "Delta", "Status"]
    assert len(df) == 5
    row = df[df["Aussage (AI-Chunk)"] == "Das BAFA zahlt Zuschüsse von bis zu 20 Prozent."].iloc[0]
    assert (row["Thema"], row["Wortzahl"], row["Status"]) == ("Förderung", 9, "Lücke")
    assert abs(row["Delta"] - (row["Score"] - row["Score Quelle"])) < 2e-4


def test_frames_without_sources_drop_the_source_columns():
    result = sample_result(with_sources=False)
    pages, topics, statements = pages_frame(result), topics_frame(result), statements_frame(result)
    assert "Aussagen mit bestem Chunk" in pages.columns and "Aussagen abgedeckt" not in pages.columns
    assert set(pages["Herkunft"]) == {"eigen"}
    for df in (topics, statements):
        assert not {"Beste Quell-URL", "Bester Quell-Chunk", "Score Quelle", "Delta"} & set(df.columns)
    assert list(topics.columns[-4:]) == list(QUARTILE_LABELS)
    assert statements.columns[-1] == "Status"


def test_chunks_frame_lists_answer_statements_own_and_source_chunks():
    df = chunks_frame(sample_result())
    assert list(df.columns) == ["Keyword/Prompt", "Herkunft", "URL", "Chunk-Index", "Text", "Wortzahl"]
    assert df["Herkunft"].value_counts().to_dict() == {"AI-Antwort": 5, "eigen": 3, "Quelle": 2}
    assert "embedding_0" not in df.columns


def test_protocol_lines_describe_answer_chunking_and_failures():
    result = sample_result()
    failed = [FailedUrl("https://quelle.de/x", SOURCE, "HTTP 403")]
    lines = protocol_lines(result.answer, result.own_stats, result.source_stats, failed)
    assert lines[0] == ("AI-Antwort: 3 Themen, 5 Aussagen, Median 12 Wörter → Zielgröße 15 Wörter "
                        "(Segmentierung: Struktur)")
    assert lines[1].startswith("Eigener Content: 2 Seiten → 3 Chunks, Median ")
    assert "2 Absätze zusammengelegt, 0 Absätze geteilt" in lines[1]
    assert lines[2].startswith("Quellen: 1 Seite → 2 Chunks")
    assert lines[3] == "Nicht geladen (Quelle): https://quelle.de/x (HTTP 403)"


def test_protocol_lines_without_sources_and_with_prechunked_content():
    result = sample_result(with_sources=False)
    lines = protocol_lines(result.answer, result.own_stats, None, [], prechunked=True)
    assert lines[1].startswith("Eigener Content (aus Chunk-Embedding-Export): ")
    assert lines[2].startswith("Quellen: keine")


def test_protocol_frame_contains_the_run_settings():
    df = protocol_frame(sample_result())
    assert list(df.columns) == ["Keyword/Prompt", "Angabe", "Wert"]
    values = dict(zip(df["Angabe"], df["Wert"]))
    assert values["Datum"] == "2026-10-03"
    assert values["Embedding-Anbieter"] == "fake" and values["Modell"] == "fake-model"
    assert values["Dimension"] == 64 and values["Zielgröße (Wörter)"] == 15
    assert values["Segmentierung"] == "Struktur"
    assert values["Schwelle abgedeckt"] == 0.05 and values["Schwelle schwach"] == 0.15
    assert any(a == "Chunking" for a in df["Angabe"])


def test_summary_with_sources_names_top_page_and_gap_topics():
    text = summary(sample_result())
    assert text.startswith("Relevanteste eigene Seite: https://me.de/kosten (Seitenscore 0,")
    assert "von 3)" in text
    assert "Lücken: 2 Aussagen im Thema „Förderung“" in text


def test_summary_without_sources_explains_the_missing_benchmark():
    text = summary(sample_result(with_sources=False))
    assert text.startswith("Relevanteste eigene Seite: https://me.de/kosten")
    assert "keinen Benchmark" in text


def test_answer_frame_shows_every_ai_chunk():
    df = answer_frame(sample_result().answer)
    assert list(df.columns) == ["Thema", "Aussage", "Wortzahl"]
    assert len(df) == 5 and df.iloc[3]["Thema"] == "Förderung" and df.iloc[3]["Wortzahl"] == 8


NASTY_TEXTS = ["abc\x00def\x07", "=SUM(A1)", "+49 221 123", "@handle", "-5 Grad", "x" * 40000]


def _result_with_nasty_chunks():
    result = sample_result()
    own = [Chunk(url="https://me.de/nasty", origin="own", index=i, text=text, words=3)
           for i, text in enumerate(NASTY_TEXTS)]
    return replace(result, own_chunks=own)


def test_excel_export_survives_control_characters_formulas_and_long_texts():
    data = to_excel(_result_with_nasty_chunks())
    sheet = openpyxl.load_workbook(io.BytesIO(data))["Chunks"]
    header = [c.value for c in sheet[1]]
    text_col = header.index("Text") + 1
    cells = {row[text_col - 1].value: row[text_col - 1] for row in sheet.iter_rows(min_row=2)}
    values = list(cells)
    assert "abcdef" in values and not any("\x00" in v or "\x07" in v for v in values)
    for original in ("=SUM(A1)", "+49 221 123", "@handle", "-5 Grad"):
        cell = cells["'" + original]
        assert cell.data_type == "s" and cell.value == "'" + original
    long_values = [v for v in values if v.startswith("xxx")]
    assert len(long_values) == 1
    assert len(long_values[0]) <= 32767 and long_values[0].endswith("[gekürzt]")


def test_excel_cleaning_does_not_alter_the_frames_shown_in_the_ui():
    result = _result_with_nasty_chunks()
    to_excel(result)
    texts = list(chunks_frame(result)["Text"])
    for original in NASTY_TEXTS:
        assert original in texts


def _stats(documents, chunks, merged, split, empty_urls=(), median=1.0):
    return ChunkStats(documents=documents, chunks=chunks, median_words=median, p10_words=1.0, p90_words=1.0,
                      merged=merged, split=split, empty_urls=empty_urls)


def test_protocol_lines_use_singular_forms_for_counts_of_one():
    answer = SegmentedAnswer("k", (Topic(1, "T", (Statement(1, 1, "Ein Satz.", 2),)),), 1, 1.0, "structure")
    lines = protocol_lines(answer, _stats(1, 1, 1, 1, ("https://x.de",)), _stats(2, 2, 2, 2, ("a", "b")), [])
    assert lines[0] == "AI-Antwort: 1 Thema, 1 Aussage, Median 1 Wort → Zielgröße 1 Wort (Segmentierung: Struktur)"
    assert lines[1] == ("Eigener Content: 1 Seite → 1 Chunk, Median 1 Wort (p10: 1, p90: 1), "
                        "1 Absatz zusammengelegt, 1 Absatz geteilt, 1 Seite ohne Text")
    assert lines[2] == ("Quellen: 2 Seiten → 2 Chunks, Median 1 Wort (p10: 1, p90: 1), "
                        "2 Absätze zusammengelegt, 2 Absätze geteilt, 2 Seiten ohne Text")


def test_summary_keeps_aussagen_plural_for_totals_and_singular_for_single_gaps():
    text = summary(sample_result())
    assert " von 5 Aussagen abgedeckt" in text
    result = sample_result()
    topics = [replace(t, counts={**t.counts, "Lücke": 1}) if t.counts.get("Lücke") else t
              for t in result.match.topics]
    text = summary(replace(result, match=replace(result.match, topics=topics)))
    assert "1 Aussage im Thema „Förderung“" in text


def test_protocol_thresholds_are_rounded():
    result = sample_result()
    result = replace(result, meta=replace(result.meta, covered_delta=0.05000000000000001,
                                          weak_delta=0.15000000000000002))
    values = dict(zip(protocol_frame(result)["Angabe"], protocol_frame(result)["Wert"]))
    assert values["Schwelle abgedeckt"] == 0.05 and values["Schwelle schwach"] == 0.15
    assert repr(values["Schwelle schwach"]) == "0.15"
