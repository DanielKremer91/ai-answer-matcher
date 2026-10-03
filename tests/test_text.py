from aam.text import count_words, normalize_space, short_label, split_sentences


def test_count_words_and_normalize_space():
    assert count_words("  Eine  Dachsanierung\tkostet Geld ") == 4
    assert normalize_space("  a \n b\t c ") == "a b c"


def test_split_sentences_basic():
    assert split_sentences("Das Dach ist alt. Es muss saniert werden! Wirklich? Ja.") == [
        "Das Dach ist alt.", "Es muss saniert werden!", "Wirklich?", "Ja."]


def test_split_sentences_keeps_abbreviations_together():
    text = "Das kostet z. B. 500 Euro. Die Dämmung kostet ca. 100 Euro bzw. etwas mehr. Siehe Nr. 3 der Liste."
    assert split_sentences(text) == [
        "Das kostet z. B. 500 Euro.",
        "Die Dämmung kostet ca. 100 Euro bzw. etwas mehr.",
        "Siehe Nr. 3 der Liste.",
    ]


def test_split_sentences_keeps_ordinals_together():
    assert split_sentences("Am 1. Januar steigt der Preis. Das gilt bundesweit.") == [
        "Am 1. Januar steigt der Preis.", "Das gilt bundesweit."]


def test_split_sentences_does_not_mistake_word_endings_for_abbreviations():
    assert split_sentences("Die Dämmung ist Pflicht und es bleibt ein Rest. Danach folgt der Estrich.") == [
        "Die Dämmung ist Pflicht und es bleibt ein Rest.", "Danach folgt der Estrich."]


def test_split_sentences_empty():
    assert split_sentences("   ") == []


def test_short_label():
    assert short_label("Kosten im Überblick:") == "Kosten im Überblick"
    assert short_label("eins zwei drei vier fünf sechs sieben acht neun zehn") == "eins zwei drei vier fünf sechs sieben acht …"
