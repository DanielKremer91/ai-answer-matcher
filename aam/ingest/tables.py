"""Read an uploaded table (CSV/TSV/TXT in any common encoding and delimiter, or Excel).

Shared by the content import and the chunk-embedding import. Every cell comes back as a string (blank cells as ""), so callers parse numbers themselves."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

_ENCODINGS = ("utf-8-sig", "utf-16", "cp1252")
_SNIFF_BYTES = 64 * 1024
_DELIMITERS = ",;\t|"
_EXCEL_EXTENSIONS = (".xlsx", ".xlsm", ".xls")
_NDJSON_EXTENSIONS = (".ndjson", ".jsonl")
_ERROR_PREFIX = "Datei konnte nicht gelesen werden"

ENC_UTF8 = "utf-8"
ENC_UTF8_REPLACED = "utf-8 (mit Ersatzzeichen)"
ENC_UTF16 = "utf-16"
ENC_CP1252 = "windows-1252"
ENC_XLSX = "xlsx"

_ENCODING_LABELS = {"utf-8-sig": ENC_UTF8, "utf-16": ENC_UTF16, "cp1252": ENC_CP1252}


@dataclass(frozen=True)
class TableInfo:
    delimiter: Optional[str]
    encoding: str  # one of the ENC_* labels
    kind: str  # "csv", "xlsx" or "ndjson"
    rows: int
    columns: list[str] = field(default_factory=list)


def _mostly_utf8(data: bytes) -> Optional[str]:
    """Decode text that is UTF-8 apart from a few stray bytes.

    Excel on macOS keeps the original UTF-8 text but writes its own generated strings
    (e.g. "03. März") in MacRoman. cp1252 would "succeed" on such a file and turn every
    umlaut into mojibake, so prefer UTF-8 with replacement when the damage is small.
    """
    text = data.decode("utf-8", errors="replace")
    bad = text.count("\ufffd")
    good = sum(1 for ch in text if ord(ch) >= 0x80 and ch != "\ufffd")
    if good == 0 or bad > max(2, good // 10):
        return None  # not UTF-8 at all (e.g. plain cp1252), let the other codecs try
    return text


def _decode_with_label(data: bytes) -> tuple[str, str]:
    try:
        return data.decode("utf-8-sig"), ENC_UTF8
    except UnicodeDecodeError:
        pass
    mostly = _mostly_utf8(data)
    if mostly is not None:
        return mostly.lstrip("﻿"), ENC_UTF8_REPLACED
    for encoding in _ENCODINGS:
        # Without a BOM the utf-16 codec "succeeds" on almost any even-length input, so
        # only try it when a BOM is present.
        if encoding == "utf-16" and not data.startswith((b"\xff\xfe", b"\xfe\xff")):
            continue
        try:
            return data.decode(encoding), _ENCODING_LABELS[encoding]
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace"), ENC_UTF8_REPLACED


def _decode(data: bytes) -> str:
    return _decode_with_label(data)[0]


def _header_delimiter(text: str) -> str:
    """Most frequent delimiter candidate in the header line; ',' if none occurs."""
    header = text.split("\n", 1)[0]
    counts = {d: header.count(d) for d in _DELIMITERS}
    best = max(_DELIMITERS, key=lambda d: counts[d])  # ties: earlier candidate wins
    return best if counts[best] else ","


def _sniff_delimiter(text: str) -> str:
    """csv.Sniffer first; the header line decides when the sniffer fails or guesses a
    delimiter that does not even occur in the header (vectors full of commas can do that).
    """
    try:
        sniffed = csv.Sniffer().sniff(text[:_SNIFF_BYTES], delimiters=_DELIMITERS).delimiter
    except csv.Error:
        return _header_delimiter(text)
    header = text.split("\n", 1)[0]
    if sniffed not in header and any(d in header for d in _DELIMITERS):
        return _header_delimiter(text)
    return sniffed


def _source_name(source, filename: Optional[str]) -> str:
    return (filename or getattr(source, "name", None) or str(source)).lower()


def _rewind(source) -> None:
    """Uploaded files may be read several times (e.g. a retry with a manual mapping)."""
    seek = getattr(source, "seek", None)
    if callable(seek):
        try:
            seek(0)
        except (OSError, ValueError):
            pass


def _read_excel(source, name: str) -> pd.DataFrame:
    if name.endswith(".xls"):
        try:
            import xlrd  # noqa: F401
        except ImportError:
            raise ValueError(
                f"{_ERROR_PREFIX}: Das alte Excel-Format .xls wird nicht unterstützt. "
                "Bitte in Excel als .xlsx speichern und erneut hochladen."
            ) from None
    try:
        df = pd.read_excel(source, sheet_name=0, dtype=str)
    except Exception as exc:  # openpyxl/zipfile raise many types for damaged files
        raise ValueError(f"{_ERROR_PREFIX}: {exc}") from exc
    df.columns = [str(c) for c in df.columns]  # numeric headers would break name lookups
    return df.fillna("")


def _read_ndjson(text: str) -> pd.DataFrame:
    """One JSON object per line (Screaming Frog MCP exports); null and blanks become ""."""
    records = []
    for no, line in enumerate(text.split("\n"), start=1):  # not splitlines: U+2028 may sit inside strings
        line = line.rstrip("\r")
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{_ERROR_PREFIX}: Zeile {no} ist kein gültiges JSON ({exc})") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"{_ERROR_PREFIX}: Zeile {no} ist kein JSON-Objekt")
        records.append({str(k): "" if v is None else (v if isinstance(v, str) else json.dumps(v)) for k, v in obj.items()})
    if not records:
        raise ValueError(f"{_ERROR_PREFIX}: Datei ist leer")
    return pd.DataFrame.from_records(records).fillna("")


def read_table_info(source, filename: Optional[str] = None) -> tuple[pd.DataFrame, TableInfo]:
    """Read CSV/TSV/TXT or XLSX/XLSM/XLS and describe how it was read.

    `filename` is used for type detection when `source` is a stream. Only the first
    sheet of a workbook is read. Raises ValueError with a German message.
    """
    name = _source_name(source, filename)
    _rewind(source)
    if name.endswith(_EXCEL_EXTENSIONS):
        df = _read_excel(source, name)
        columns = [str(c) for c in df.columns]
        return df, TableInfo(delimiter=None, encoding=ENC_XLSX, kind="xlsx", rows=len(df), columns=columns)
    try:
        if hasattr(source, "read"):
            data = source.read()
        else:
            with open(source, "rb") as fh:
                data = fh.read()
        if isinstance(data, str):
            data = data.encode("utf-8")
        if not data.strip():
            raise ValueError("Datei ist leer")
        text, encoding = _decode_with_label(data)
        if name.endswith(_NDJSON_EXTENSIONS):
            df = _read_ndjson(text)
            return df, TableInfo(delimiter=None, encoding=encoding, kind="ndjson", rows=len(df),
                                 columns=[str(c) for c in df.columns])
        delimiter = _sniff_delimiter(text)
        df = pd.read_csv(io.StringIO(text), sep=delimiter, dtype=str, keep_default_na=False)
    except ValueError as exc:
        if str(exc).startswith(_ERROR_PREFIX):
            raise
        raise ValueError(f"{_ERROR_PREFIX}: {exc}") from exc
    except (OSError, csv.Error, pd.errors.ParserError) as exc:
        raise ValueError(f"{_ERROR_PREFIX}: {exc}") from exc
    columns = [str(c) for c in df.columns]
    return df, TableInfo(delimiter=delimiter, encoding=encoding, kind="csv", rows=len(df), columns=columns)


def read_table(source, filename: Optional[str] = None) -> pd.DataFrame:
    """Like read_table_info, without the description."""
    return read_table_info(source, filename)[0]


_DELIMITER_NAMES = {"\t": "Tab"}
_ENCODING_NAMES = {
    ENC_UTF8: "UTF-8",
    ENC_UTF8_REPLACED: "UTF-8 mit Ersatzzeichen",
    ENC_UTF16: "UTF-16",
    ENC_CP1252: "Windows-1252",
}


def describe_table_info(info: TableInfo) -> str:
    """German one-liner for the UI, e.g. 'Gelesen: CSV · Trennzeichen ; · UTF-8 · 100 Zeilen'."""
    rows = f"{info.rows} {'Zeile' if info.rows == 1 else 'Zeilen'}"
    if info.kind == "xlsx":
        return f"Gelesen: XLSX · {rows}"
    if info.kind == "ndjson":
        return f"Gelesen: NDJSON · {rows}"
    delimiter = _DELIMITER_NAMES.get(info.delimiter or "", info.delimiter or "?")
    encoding = _ENCODING_NAMES.get(info.encoding, info.encoding)
    return f"Gelesen: CSV · Trennzeichen {delimiter} · {encoding} · {rows}"
