import io

import pandas as pd
import pytest

from aam.ingest.tables import TableInfo, describe_table_info, read_table, read_table_info


def frog_text(delimiter: str, rows: int = 3, dims: int = 50) -> str:
    """Screaming Frog UI export layout: the vector is one quoted, comma-separated field."""
    header = delimiter.join(f'"{c}"' for c in ("Address", "Content Type", "Status Code", "Status", "Embeddings X 1"))
    lines = [header]
    for r in range(rows):
        vector = ",".join(f"{(r + 1) * 0.001 * (i + 1):.6f}" for i in range(dims))
        cells = [f"https://me.de/{r}", "text/html;charset=utf-8", "200", "", vector]
        lines.append(delimiter.join(f'"{c}"' for c in cells))
    return "\r\n".join(lines) + "\r\n"


@pytest.mark.parametrize("delimiter", [",", ";"])
def test_sniffer_handles_quoted_vector_with_many_commas(tmp_path, delimiter):
    p = tmp_path / "frog.csv"
    p.write_text(frog_text(delimiter), encoding="utf-8")
    df, info = read_table_info(p)
    assert list(df.columns) == ["Address", "Content Type", "Status Code", "Status", "Embeddings X 1"]
    assert len(df) == 3
    assert df.loc[0, "Embeddings X 1"].count(",") == 49
    assert info.delimiter == delimiter


def test_unquoted_semicolon_header_with_comma_vectors(tmp_path):
    p = tmp_path / "semi.csv"
    rows = ["url;embedding"] + [f'https://me.de/{i};"0.1,0.2,0.3,0.4"' for i in range(5)]
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    df, info = read_table_info(p)
    assert info.delimiter == ";"
    assert list(df.columns) == ["url", "embedding"]
    assert df.loc[0, "embedding"] == "0.1,0.2,0.3,0.4"


def test_header_count_fallback_when_sniffer_guesses_a_delimiter_missing_from_header(monkeypatch, tmp_path):
    from aam.ingest import tables

    class Wrong:
        delimiter = ","

    monkeypatch.setattr(tables.csv.Sniffer, "sniff", lambda self, sample, delimiters=None: Wrong())
    p = tmp_path / "semi.csv"
    p.write_text('url;embedding\nhttps://me.de/a;"0.1,0.2"\n', encoding="utf-8")
    df, info = read_table_info(p)
    assert info.delimiter == ";"
    assert list(df.columns) == ["url", "embedding"]


def test_header_count_fallback_when_sniffer_raises(monkeypatch, tmp_path):
    import csv

    from aam.ingest import tables

    def boom(self, sample, delimiters=None):
        raise csv.Error("could not determine delimiter")

    monkeypatch.setattr(tables.csv.Sniffer, "sniff", boom)
    p = tmp_path / "pipe.txt"
    p.write_text("a|b|c\n1|2|3\n", encoding="utf-8")
    df, info = read_table_info(p)
    assert info.delimiter == "|"
    assert list(df.columns) == ["a", "b", "c"]


def test_read_tsv(tmp_path):
    p = tmp_path / "x.tsv"
    p.write_text("url_from\turl_to\nhttps://a.de\thttps://b.de/x\n", encoding="utf-8")
    df, info = read_table_info(p)
    assert list(df.columns) == ["url_from", "url_to"]
    assert info == TableInfo(delimiter="\t", encoding="utf-8", kind="csv", rows=1, columns=["url_from", "url_to"])


def test_read_xlsx_returns_strings_and_blank_for_missing(tmp_path):
    p = tmp_path / "x.xlsx"
    pd.DataFrame({"url": ["https://a.de", "https://b.de"], "dr": [72.5, None], "n": [3, 4]}).to_excel(p, index=False)
    df, info = read_table_info(p)
    assert df.loc[0, "dr"] == "72.5"
    assert df.loc[1, "dr"] == ""
    assert df.loc[0, "n"] == "3"
    assert all(isinstance(v, str) for v in df.to_numpy().ravel())
    assert info == TableInfo(delimiter=None, encoding="xlsx", kind="xlsx", rows=2, columns=["url", "dr", "n"])


def test_read_xlsx_reads_only_first_sheet(tmp_path):
    p = tmp_path / "two.xlsx"
    with pd.ExcelWriter(p) as writer:
        pd.DataFrame({"first": ["a"]}).to_excel(writer, sheet_name="Eins", index=False)
        pd.DataFrame({"second": ["b"]}).to_excel(writer, sheet_name="Zwei", index=False)
    assert list(read_table(p).columns) == ["first"]


def test_read_xlsm_extension_and_stream(tmp_path):
    p = tmp_path / "x.xlsx"
    pd.DataFrame({"url": ["https://a.de"]}).to_excel(p, index=False)
    stream = io.BytesIO(p.read_bytes())
    df, info = read_table_info(stream, filename="upload.xlsm")
    assert info.kind == "xlsx" and list(df.columns) == ["url"]


def test_xls_without_xlrd_raises_german_hint(monkeypatch, tmp_path):
    import builtins

    real_import = builtins.__import__

    def no_xlrd(name, *args, **kwargs):
        if name == "xlrd":
            raise ImportError("no xlrd")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_xlrd)
    p = tmp_path / "alt.xls"
    p.write_bytes(b"\xd0\xcf\x11\xe0 not really xls")
    with pytest.raises(ValueError, match=r"\.xlsx"):
        read_table(p)


def test_broken_xlsx_raises_german_value_error(tmp_path):
    p = tmp_path / "kaputt.xlsx"
    p.write_bytes(b"this is not a zip file")
    with pytest.raises(ValueError, match="Datei konnte nicht gelesen werden"):
        read_table(p)


def test_stream_is_rewound_so_it_can_be_read_twice():
    stream = io.BytesIO(b"a,b\n1,2\n")
    read_table(stream, filename="x.csv")
    df = read_table(stream, filename="x.csv")
    assert list(df.columns) == ["a", "b"]


@pytest.mark.parametrize(
    "data,label",
    [
        ("a;b\nÄ;1\n".encode("utf-8"), "utf-8"),
        ("a;b\nÄ;1\n".encode("utf-8-sig"), "utf-8"),
        ("a\tb\nÄ\t1\n".encode("utf-16"), "utf-16"),
        ("a;b\nÄ;1\n".encode("cp1252"), "windows-1252"),
        ("a;b\nKüche Größe Übung;".encode("utf-8") + b"M\x8arz\n", "utf-8 (mit Ersatzzeichen)"),
    ],
)
def test_table_info_encoding_label(data, label):
    _, info = read_table_info(io.BytesIO(data), filename="x.csv")
    assert info.encoding == label


@pytest.mark.parametrize(
    "info,expected",
    [
        (TableInfo(";", "utf-8 (mit Ersatzzeichen)", "csv", 100, ["a"]),
         "Gelesen: CSV · Trennzeichen ; · UTF-8 mit Ersatzzeichen · 100 Zeilen"),
        (TableInfo(",", "utf-8", "csv", 1336, ["a"]), "Gelesen: CSV · Trennzeichen , · UTF-8 · 1336 Zeilen"),
        (TableInfo("\t", "utf-16", "csv", 2, ["a"]), "Gelesen: CSV · Trennzeichen Tab · UTF-16 · 2 Zeilen"),
        (TableInfo(None, "xlsx", "xlsx", 1, ["a"]), "Gelesen: XLSX · 1 Zeile"),
    ],
)
def test_describe_table_info(info, expected):
    assert describe_table_info(info) == expected


def test_ndjson_is_read_as_table(tmp_path):
    from aam.ingest.tables import describe_table_info, read_table_info
    p = tmp_path / "frog.ndjson"
    p.write_text('{"Address":"https://a.de/x","Ext 1":"0.1,0.2"}\n\n{"Address":"https://a.de/y","Ext 1":null}\n',
                 encoding="utf-8")
    df, info = read_table_info(p)
    assert list(df.columns) == ["Address", "Ext 1"] and len(df) == 2
    assert df.loc[1, "Ext 1"] == ""
    assert info.kind == "ndjson" and describe_table_info(info) == "Gelesen: NDJSON · 2 Zeilen"


def test_ndjson_keeps_unicode_line_separators_inside_strings(tmp_path):
    from aam.ingest.tables import read_table
    p = tmp_path / "x.ndjson"
    p.write_bytes('{"Address":"https://a.de","Title":"a\u2028b"}\r\n'.encode("utf-8"))
    df = read_table(p)
    assert len(df) == 1 and df.loc[0, "Title"] == "a\u2028b"


def test_broken_ndjson_raises_value_error(tmp_path):
    import pytest
    from aam.ingest.tables import read_table
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n{kaputt\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Datei konnte nicht gelesen werden"):
        read_table(p)
