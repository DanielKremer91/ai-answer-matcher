"""Chunk-embedding export: embed a domain once, reuse it for further keywords."""

from __future__ import annotations

import re
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from aam.ingest.tables import read_table
from aam.models import OWN, Chunk
from aam.text import count_words

META_COLUMNS = ("url", "chunk_index", "text", "words", "provider", "model", "target_words")
_EMBEDDING_COLUMN = re.compile(r"embedding_(\d+)")


class ChunkFileError(ValueError):
    """The uploaded chunk-embedding export cannot be used with the current settings."""


def is_chunks_table(columns: Sequence[str]) -> bool:
    names = {str(c) for c in columns}
    return "chunk_index" in names and "embedding_0" in names


def chunks_to_csv(chunks: Sequence[Chunk], *, provider: str, model: str, target_words: int) -> bytes:
    rows = [c for c in chunks if c.vector is not None]
    if not rows:
        raise ChunkFileError("Keine Chunks mit Embedding vorhanden.")
    meta = pd.DataFrame({
        "url": [c.url for c in rows],
        "chunk_index": [c.index for c in rows],
        "text": [c.text for c in rows],
        "words": [c.words for c in rows],
        "provider": provider,
        "model": model,
        "target_words": target_words,
    })
    matrix = np.round(np.vstack([c.vector for c in rows]).astype(np.float64), 6)
    vectors = pd.DataFrame(matrix, columns=[f"embedding_{i}" for i in range(matrix.shape[1])])
    return pd.concat([meta, vectors], axis=1).to_csv(index=False).encode("utf-8")


def chunks_from_frame(df: pd.DataFrame, *, provider: str, model: str) -> tuple[list[Chunk], int]:
    columns = [str(c) for c in df.columns]
    missing = [c for c in META_COLUMNS if c not in columns]
    embedding_columns = sorted((c for c in columns if _EMBEDDING_COLUMN.fullmatch(c)),
                               key=lambda c: int(c.split("_")[1]))
    if missing or not embedding_columns:
        missing_text = ", ".join(missing) if missing else "embedding_0"
        raise ChunkFileError(
            f"Die Datei ist kein Chunk-Embedding-Export dieses Tools. Fehlende Spalten: {missing_text}")
    if df.empty:
        raise ChunkFileError("Der Chunk-Embedding-Export enthält keine Zeilen.")
    found = {(str(p).strip(), str(m).strip()) for p, m in zip(df["provider"], df["model"])}
    if found != {(provider, model)}:
        described = ", ".join(f"{p} / {m}" for p, m in sorted(found))
        raise ChunkFileError(
            f"Der Export wurde mit {described} erzeugt, eingestellt ist {provider} / {model}. "
            "Bitte Anbieter und Modell in der Seitenleiste anpassen oder den Content neu einbetten.")
    try:
        matrix = df[embedding_columns].to_numpy(dtype=str).astype(np.float32)
        target = int(float(df["target_words"].iloc[0]))
        indices = [int(float(v)) for v in df["chunk_index"]]
    except ValueError as exc:
        raise ChunkFileError("Der Chunk-Embedding-Export enthält ungültige Zahlen.") from exc
    chunks = []
    for row, (url, text) in enumerate(zip(df["url"], df["text"])):
        text = str(text)
        chunks.append(Chunk(str(url).strip(), OWN, indices[row], text, count_words(text), matrix[row]))
    return chunks, target


def read_chunks_file(source, filename: Optional[str] = None, *, provider: str, model: str) -> tuple[list[Chunk], int]:
    try:
        df = read_table(source, filename)
    except ValueError as exc:
        if "leer" in str(exc):
            raise ChunkFileError("Der Chunk-Embedding-Export enthält keine Zeilen.") from exc
        raise
    return chunks_from_frame(df, provider=provider, model=model)


def check_dimension(chunks: Sequence[Chunk], dimension: int) -> None:
    """The uploaded vectors must have the dimension the current model produces."""
    if chunks and len(chunks[0].vector) != dimension:
        raise ChunkFileError(
            f"Die hochgeladenen Chunk-Embeddings haben Dimension {len(chunks[0].vector)}, "
            f"das eingestellte Modell liefert Dimension {dimension}.")
