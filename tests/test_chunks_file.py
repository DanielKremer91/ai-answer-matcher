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


def read_exported(data: bytes):
    return read_chunks_file(io.BytesIO(data), "c.csv", provider="openai", model="text-embedding-3-small")


def test_excel_resaved_export_with_semicolons_and_decimal_commas_is_accepted():
    frame = pd.read_csv(io.BytesIO(exported()), dtype=str, keep_default_na=False)
    for column in ("embedding_0", "embedding_1", "embedding_2"):
        frame[column] = frame[column].str.replace(".", ",", regex=False)
    frame["target_words"] = "27,0"
    resaved = frame.to_csv(sep=";", index=False).encode("utf-8")
    assert b"0,123457" in resaved
    chunks, target = read_exported(resaved)
    assert target == 27
    assert chunks[0].vector.tolist() == approx([0.123457, -0.5, 0.25], abs=1e-6)
    assert chunks[1].vector.tolist() == approx([1.0, 0.0, 0.0], abs=1e-6)
    assert chunks[0].text == "Erster Chunk, mit Komma; und Semikolon"


def test_unreadable_numbers_mention_the_excel_resave():
    text = exported().decode("utf-8").replace("-0.5", "kaputt")
    with pytest.raises(ChunkFileError, match="ohne ihn in Excel neu zu speichern"):
        read_exported(text.encode("utf-8"))


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
def test_non_finite_embedding_values_are_rejected(bad):
    text = exported().decode("utf-8").replace("-0.5", bad)
    with pytest.raises(ChunkFileError, match="ungültige Zahlen"):
        read_exported(text.encode("utf-8"))


def test_empty_embedding_cell_is_rejected():
    text = exported().decode("utf-8").replace("-0.5", "")
    with pytest.raises(ChunkFileError, match="ungültige Zahlen"):
        read_exported(text.encode("utf-8"))


def test_infinite_chunk_index_is_rejected():
    frame = pd.read_csv(io.BytesIO(exported()), dtype=str, keep_default_na=False)
    frame.loc[0, "chunk_index"] = "inf"
    with pytest.raises(ChunkFileError, match="ungültige Zahlen"):
        read_exported(frame.to_csv(index=False).encode("utf-8"))


def test_target_words_below_one_is_rejected():
    with pytest.raises(ChunkFileError, match="ungültige Zahlen"):
        read_exported(exported(target_words=0))


def test_roundtrip_keeps_line_break_tab_and_pipe_in_the_text():
    text = "Zeile eins\nZeile zwei\tmit Tab | und Pipe"
    data = chunks_to_csv([Chunk("https://me.de/a", OWN, 0, text, 8, np.array([1.0, 2.0], dtype=np.float32))],
                         provider="openai", model="text-embedding-3-small", target_words=20)
    chunks, _ = read_exported(data)
    assert chunks[0].text == text
