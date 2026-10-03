import io

import numpy as np
import pandas as pd
import pytest
from pytest import approx

from aam.ingest.chunks_file import (ChunkFileError, check_dimension, chunks_from_frame, chunks_to_csv,
                                    is_chunks_table, read_chunks_file)
from aam.models import OWN, Chunk


def sample_chunks():
    return [
        Chunk("https://me.de/a", OWN, 0, "Erster Chunk, mit Komma; und Semikolon", 6,
              np.array([0.1234567, -0.5, 0.25], dtype=np.float32)),
        Chunk("https://me.de/a", OWN, 1, 'Zweiter "Chunk" mit Anführungszeichen', 4,
              np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        Chunk("https://me.de/b", OWN, 0, "Ohne Vektor", 2),
    ]


def exported(**overrides):
    params = {"provider": "openai", "model": "text-embedding-3-small", "target_words": 27, **overrides}
    return chunks_to_csv(sample_chunks(), **params)


def test_roundtrip_keeps_text_order_vectors_and_target():
    chunks, target = read_chunks_file(io.BytesIO(exported()), "chunks_embeddings.csv",
                                      provider="openai", model="text-embedding-3-small")
    assert target == 27
    assert [(c.url, c.index, c.origin) for c in chunks] == [("https://me.de/a", 0, OWN), ("https://me.de/a", 1, OWN)]
    assert chunks[0].text == "Erster Chunk, mit Komma; und Semikolon"
    assert chunks[1].text == 'Zweiter "Chunk" mit Anführungszeichen'
    assert [c.words for c in chunks] == [6, 4]
    assert chunks[0].vector.tolist() == approx([0.123457, -0.5, 0.25], abs=1e-6)
    assert chunks[0].vector.dtype == np.float32


def test_export_columns():
    header = exported().decode("utf-8").split("\n", 1)[0].strip()
    assert header == "url,chunk_index,text,words,provider,model,target_words,embedding_0,embedding_1,embedding_2"


def test_is_chunks_table():
    assert is_chunks_table(["url", "chunk_index", "text", "embedding_0"]) is True
    assert is_chunks_table(["url", "text"]) is False
    assert is_chunks_table(["Address", "embedding_0"]) is False


def test_other_model_is_rejected_with_both_names():
    with pytest.raises(ChunkFileError, match="openai / text-embedding-3-small erzeugt, eingestellt ist openai / other"):
        read_chunks_file(io.BytesIO(exported()), "c.csv", provider="openai", model="other")


def test_other_provider_is_rejected():
    with pytest.raises(ChunkFileError, match="eingestellt ist gemini"):
        read_chunks_file(io.BytesIO(exported()), "c.csv", provider="gemini", model="text-embedding-3-small")


def test_table_that_is_not_an_export_is_rejected():
    df = pd.DataFrame({"url": ["u"], "chunk_index": ["0"], "embedding_0": ["0.1"]})
    with pytest.raises(ChunkFileError, match="kein Chunk-Embedding-Export.*text, words, provider, model, target_words"):
        chunks_from_frame(df, provider="openai", model="m")


def test_invalid_numbers_are_rejected():
    text = exported().decode("utf-8").replace("-0.5", "kaputt")
    with pytest.raises(ChunkFileError, match="ungültige Zahlen"):
        read_chunks_file(io.BytesIO(text.encode("utf-8")), "c.csv", provider="openai", model="text-embedding-3-small")


def test_empty_export_is_rejected():
    header = exported().decode("utf-8").split("\n", 1)[0] + "\n"
    with pytest.raises(ChunkFileError, match="keine Zeilen"):
        read_chunks_file(io.BytesIO(header.encode("utf-8")), "c.csv", provider="openai", model="text-embedding-3-small")


def test_export_without_any_vector_raises():
    with pytest.raises(ChunkFileError, match="Keine Chunks mit Embedding"):
        chunks_to_csv([Chunk("u", OWN, 0, "x", 1)], provider="p", model="m", target_words=20)


def test_check_dimension():
    chunks = sample_chunks()[:2]
    check_dimension(chunks, 3)
    with pytest.raises(ChunkFileError, match="Dimension 3.*Dimension 1536"):
        check_dimension(chunks, 1536)
