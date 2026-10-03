import io

import pandas as pd

from aam.match import QUARTILE_LABELS
from aam.models import SOURCE, FailedUrl
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
    assert lines[2].startswith("Quellen: 1 Seiten → 2 Chunks")
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
