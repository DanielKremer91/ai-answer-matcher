import io

import pandas as pd
import pytest

from aam.ingest.content import documents_from_frame, read_content_table
from aam.models import OWN, SOURCE

FROG_CSV = (
    "Address,Extract Main Content 1,Title\n"
    'https://me.de/a,"Absatz eins.\nAbsatz zwei.",A\n'
    "https://me.de/b,,B\n"
    "https://me.de/a/,Doppelt,C\n"
)


def test_frog_export_is_read_with_paragraphs_empty_rows_and_duplicates():
    docs, info = read_content_table(io.BytesIO(FROG_CSV.encode("utf-8")), "frog.csv")
    assert [d.url for d in docs] == ["https://me.de/a", "https://me.de/b"]
    assert docs[0].text == "Absatz eins.\nAbsatz zwei."
    assert docs[0].origin == OWN
    assert (info.url_column, info.content_column) == ("Address", "Extract Main Content 1")
    assert (info.rows, info.documents, info.empty, info.duplicates) == (3, 2, 1, 1)


def test_utf16_tab_export_with_german_headers():
    data = "Adresse\tSeiteninhalt\nhttps://me.de/ä\tGrüße aus Köln\n".encode("utf-16")
    docs, info = read_content_table(io.BytesIO(data), "export.csv", origin=SOURCE)
    assert docs[0].text == "Grüße aus Köln"
    assert docs[0].origin == SOURCE
    assert (info.url_column, info.content_column) == ("Adresse", "Seiteninhalt")


def test_content_column_is_found_by_substring_when_no_exact_name_matches():
    df = pd.DataFrame({"URL": ["https://me.de/a"], "Custom Main Content": ["Text"]})
    docs, info = documents_from_frame(df)
    assert info.content_column == "Custom Main Content"
    assert docs[0].text == "Text"


def test_exact_column_name_wins_over_substring():
    df = pd.DataFrame({"url": ["https://me.de/a"], "Content Type": ["text/html"], "Main Content": ["Text"]})
    assert documents_from_frame(df)[1].content_column == "Main Content"


def test_rows_without_url_are_skipped():
    df = pd.DataFrame({"url": ["", "https://me.de/a"], "text": ["x", "y"]})
    docs, info = documents_from_frame(df)
    assert [d.text for d in docs] == ["y"]
    assert info.documents == 1


@pytest.mark.parametrize("columns", [["Foo", "Bar"], ["url", "Bar"], ["Foo", "text"]])
def test_missing_columns_raise_with_the_found_names(columns):
    df = pd.DataFrame({c: ["x"] for c in columns})
    with pytest.raises(ValueError, match=f"Gefunden: {columns[0]}, {columns[1]}"):
        documents_from_frame(df)
