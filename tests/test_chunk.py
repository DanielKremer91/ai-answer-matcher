from aam.chunk import chunk_document, chunk_documents, stats_for
from aam.models import OWN, Chunk, Document


def sentence(i: int, n: int = 12) -> str:
    """A sentence with exactly n words that the sentence splitter recognises."""
    return f"Satz{i} " + " ".join(f"wort{i}{j}" for j in range(n - 2)) + f" schluss{i}."


def doc(text: str, url: str = "https://me.de/a") -> Document:
    return Document(url, text, OWN)


def test_short_paragraphs_are_merged_with_their_neighbours():
    text = "\n".join(["Kosten der Dachsanierung im Überblick", sentence(1, 22), sentence(2, 3)])
    chunks, merged, split = chunk_document(doc(text), 20)
    assert len(chunks) == 1
    assert chunks[0].words == 30
    assert chunks[0].text.startswith("Kosten der Dachsanierung im Überblick Satz1")
    assert (merged, split) == (2, 0)
    assert (chunks[0].url, chunks[0].origin, chunks[0].index, chunks[0].vector) == ("https://me.de/a", OWN, 0, None)


def test_no_merge_when_the_result_would_exceed_twice_the_target():
    text = "\n".join(["Kosten der Dachsanierung im Überblick", sentence(1, 38)])
    chunks, merged, split = chunk_document(doc(text), 20)
    assert [c.words for c in chunks] == [5, 38]
    assert (merged, split) == (0, 0)


def test_long_paragraph_is_cut_into_windows_with_one_sentence_overlap():
    paragraph = " ".join(sentence(i) for i in range(1, 6))  # 60 words, target 20
    chunks, merged, split = chunk_document(doc(paragraph), 20)
    assert [c.text for c in chunks] == [
        f"{sentence(1)} {sentence(2)}",
        f"{sentence(2)} {sentence(3)}",
        f"{sentence(3)} {sentence(4)}",
        f"{sentence(4)} {sentence(5)}",
    ]
    assert [c.index for c in chunks] == [0, 1, 2, 3]
    assert (merged, split) == (0, 1)


def test_window_remainder_is_kept():
    paragraph = " ".join(sentence(i) for i in range(1, 7))  # 72 words, target 30
    chunks, _, _ = chunk_document(doc(paragraph), 30)
    assert [c.words for c in chunks] == [36, 36, 24]
    assert chunks[2].text == f"{sentence(5)} {sentence(6)}"


def test_single_oversized_sentence_stays_one_chunk():
    chunks, merged, split = chunk_document(doc(sentence(1, 50)), 20)
    assert [c.words for c in chunks] == [50]
    assert (merged, split) == (0, 0)


def test_whitespace_is_normalised_and_blank_lines_are_ignored():
    chunks, _, _ = chunk_document(doc(f"\n\n  {sentence(1, 20)}  \n\n\t\n{sentence(2, 20)}\n"), 20)
    assert [c.text for c in chunks] == [sentence(1, 20), sentence(2, 20)]


def test_empty_document_yields_no_chunks_and_is_reported():
    chunks, stats = chunk_documents([doc(" \n ", "https://me.de/leer"), doc(sentence(1, 20))], 20)
    assert len(chunks) == 1
    assert stats.documents == 2
    assert stats.empty_urls == ("https://me.de/leer",)


def test_stats_describe_the_word_distribution():
    docs = [doc(sentence(1, n), f"https://me.de/{n}") for n in (10, 20, 30)]
    chunks, stats = chunk_documents(docs, 20)
    assert [c.words for c in chunks] == [10, 20, 30]
    assert (stats.documents, stats.chunks, stats.merged, stats.split) == (3, 3, 0, 0)
    assert (stats.median_words, stats.p10_words, stats.p90_words) == (20.0, 12.0, 28.0)


def test_stats_for_without_chunks():
    stats = stats_for([], documents=1, empty_urls=["u"])
    assert (stats.chunks, stats.median_words, stats.empty_urls) == (0, 0.0, ("u",))


def test_stats_for_existing_chunks():
    chunks = [Chunk("u", OWN, 0, "a b", 2), Chunk("u", OWN, 1, "a b c d", 4)]
    assert stats_for(chunks, documents=1).median_words == 3.0
