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
        f"{sentence(2)} {sentence(3)} {sentence(4)}",
        f"{sentence(4)} {sentence(5)}",
    ]
    assert [c.index for c in chunks] == [0, 1, 2]
    assert (merged, split) == (0, 1)


def test_window_overlaps_the_previous_sentence_within_the_limit():
    paragraph = " ".join(sentence(i) for i in range(1, 7))  # 72 words, target 30
    chunks, _, _ = chunk_document(doc(paragraph), 30)
    assert [c.words for c in chunks] == [36, 48]
    assert chunks[0].text == " ".join(sentence(i) for i in (1, 2, 3))
    assert chunks[1].text == " ".join(sentence(i) for i in (3, 4, 5, 6))


def test_windows_stay_within_twice_the_target_and_lose_no_text():
    sents = [sentence(i, 30) for i in range(1, 4)]
    chunks, _, split = chunk_document(doc(" ".join(sents)), 20)
    assert all(c.words <= 40 for c in chunks)
    assert all(any(s in c.text for c in chunks) for s in sents)
    assert all(c.text.count(s) <= 1 for c in chunks for s in sents)
    assert [c.text for c in chunks] == sents  # 30 + 30 > 40: no overlap fits
    assert split == 1


def test_sentences_above_the_overlap_budget_become_exactly_their_own_chunks():
    sents = [sentence(i, 39) for i in range(1, 5)]
    chunks, _, _ = chunk_document(doc(" ".join(sents)), 20)
    assert [c.text for c in chunks] == sents
    assert [c.words for c in chunks] == [39, 39, 39, 39]


def test_short_tail_group_is_folded_without_duplicating_a_sentence():
    s1, s2, s3, s4 = sentence(1, 25), sentence(2, 25), sentence(3, 4), sentence(4, 4)
    chunks, merged, split = chunk_document(doc(" ".join([s1, s2, s3, s4])), 20)
    # groups [s1] [s2] [s3 s4] -> last group (8 < 10) folds into [s2]; overlap s1 would give 58 > 40
    assert [c.text for c in chunks] == [s1, f"{s2} {s3} {s4}"]
    assert [c.words for c in chunks] == [25, 33]
    assert all(c.text.count(s) <= 1 for c in chunks for s in (s1, s2, s3, s4))
    assert (merged, split) == (0, 1)


def test_short_last_group_is_folded_into_the_previous_group():
    sents = [sentence(1), sentence(2), sentence(3), sentence(4), sentence(5, 4)]
    chunks, merged, split = chunk_document(doc(" ".join(sents)), 20)
    # groups [s1 s2] [s3 s4] [s5] -> [s1 s2] [s3 s4 s5]; window 2 = s2 + group = 40 words
    assert [c.text for c in chunks] == [
        f"{sents[0]} {sents[1]}",
        f"{sents[1]} {sents[2]} {sents[3]} {sents[4]}",
    ]
    assert [c.words for c in chunks] == [24, 40]
    assert (merged, split) == (0, 1)


def test_oversized_sentence_in_a_paragraph_is_its_own_chunk():
    a, b, big, c, d = sentence(1), sentence(2), sentence(3, 50), sentence(4), sentence(5)
    chunks, merged, split = chunk_document(doc(" ".join([a, b, big, c, d])), 20)
    # groups [a b] [big] [c d]; the overlap sentence would push either neighbour over 40
    assert [x.text for x in chunks] == [f"{a} {b}", big, f"{c} {d}"]
    assert [x.words for x in chunks] == [24, 50, 24]
    assert (merged, split) == (0, 1)


def test_heading_merges_into_the_first_window_when_it_fits():
    heading = "Kosten der Dachsanierung im Überblick"
    sents = [sentence(i) for i in range(1, 5)]  # 48 words -> windows of 24 and 36
    chunks, merged, split = chunk_document(doc("\n".join([heading, " ".join(sents)])), 20)
    assert [c.text for c in chunks] == [
        f"{heading} {sents[0]} {sents[1]}",
        f"{sents[1]} {sents[2]} {sents[3]}",
    ]
    assert (merged, split) == (1, 1)


def test_heading_stays_separate_when_the_first_window_is_too_long():
    heading = "Kosten der Dachsanierung im Überblick"
    sents = [sentence(1, 39), sentence(2, 39)]
    chunks, merged, split = chunk_document(doc("\n".join([heading, " ".join(sents)])), 20)
    assert [c.text for c in chunks] == [heading, sents[0], sents[1]]
    assert (merged, split) == (0, 1)


def test_short_paragraph_after_a_split_paragraph_merges_into_its_last_window():
    sents = [sentence(i) for i in range(1, 5)]  # windows of 24 and 36 words
    tail = "Ende des Textes"
    chunks, merged, split = chunk_document(doc("\n".join([" ".join(sents), tail])), 20)
    assert [c.text for c in chunks] == [
        f"{sents[0]} {sents[1]}",
        f"{sents[1]} {sents[2]} {sents[3]} {tail}",
    ]
    assert [c.words for c in chunks] == [24, 39]
    assert (merged, split) == (1, 1)


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


def test_splitting_can_be_switched_off():
    paragraph = " ".join(sentence(i) for i in range(1, 6))  # 60 words, target 20
    chunks, merged, split = chunk_document(doc(paragraph), 20, split_long=False)
    assert [c.words for c in chunks] == [60] and (merged, split) == (0, 0)


def test_merging_can_be_switched_off():
    text = "\n".join(["Kosten der Dachsanierung im Überblick", sentence(1, 22), sentence(2, 3)])
    chunks, merged, split = chunk_document(doc(text), 20, merge_short=False)
    assert [c.words for c in chunks] == [5, 22, 3] and (merged, split) == (0, 0)


def test_stats_list_the_split_paragraphs_and_the_settings():
    long_paragraph = " ".join(sentence(i) for i in range(1, 6))
    docs = [doc(long_paragraph, "https://me.de/lang"), doc(sentence(9, 20), "https://me.de/kurz")]
    _, stats = chunk_documents(docs, 20)
    assert stats.split_examples == (("https://me.de/lang", 60, 3),)
    assert (stats.merge_short, stats.split_long) == (True, True)
    _, stats = chunk_documents(docs, 20, split_long=False, merge_short=False)
    assert stats.split_examples == () and stats.split == 0
    assert (stats.merge_short, stats.split_long) == (False, False)


def test_heading_markers_are_stripped_and_the_heading_merges_with_the_next_paragraph():
    text = "## Kosten\nDie Sanierung kostet 100 Euro pro Quadratmeter und hängt vom Material ab."
    chunks, merged, split = chunk_document(doc(text), 20)
    assert [c.text for c in chunks] == [
        "Kosten Die Sanierung kostet 100 Euro pro Quadratmeter und hängt vom Material ab."]
    assert (merged, split) == (1, 0)
    assert "#" not in chunks[0].text
