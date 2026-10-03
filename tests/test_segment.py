import pytest

from aam.embeddings import embed_cached
from aam.segment import INTRO_LABEL, derive_target, segment_answer
from tests.fakes import ANSWER, FakeProvider

LEAD_INS = """Die Kosten einer Dachsanierung hängen von mehreren Faktoren ab:

* **Kosten:** Eine Neueindeckung kostet 100 bis 150 Euro pro Quadratmeter.
* **Förderung:** Die KfW und das BAFA fördern energetische Sanierungen.
* **Dauer:** Die Arbeiten dauern meist ein bis zwei Wochen.
"""

BULLETS = """Wichtige Punkte bei der Dachsanierung
- Die Statik des Dachstuhls muss vorab geprüft werden.
- Eine Dämmung ist nach dem Gebäudeenergiegesetz oft Pflicht.
1. Ein Gerüst ist für die Arbeiten zwingend erforderlich.
"""

PROSE = (
    "Die Kosten der Dachsanierung liegen bei 200 Euro. "
    "Die Kosten der Dachsanierung steigen mit Dämmung. "
    "Die Kosten der Dachsanierung sinken bei Eigenleistung. "
    "Eine Förderung zahlt das BAFA als Zuschuss. "
    "Eine Förderung zahlt das BAFA nach Antrag. "
    "Eine Förderung zahlt das BAFA für Wärmepumpen."
)


def fake_embed():
    provider = FakeProvider()
    return lambda texts: embed_cached(provider, texts, None)


def labels(answer):
    return [t.label for t in answer.topics]


def test_headings_define_topics_and_blank_lines_do_not_split_them():
    answer = segment_answer("dachsanierung kosten", ANSWER)
    assert labels(answer) == [INTRO_LABEL, "Kosten im Überblick", "Förderung"]
    assert [len(t.statements) for t in answer.topics] == [1, 2, 2]
    assert answer.method == "structure"
    assert answer.keyword == "dachsanierung kosten"
    assert answer.topics[2].statements[0].text == "Die KfW fördert die Dachdämmung mit zinsgünstigen Krediten."


def test_statement_ids_are_consecutive_and_carry_topic_and_word_count():
    answer = segment_answer("kw", ANSWER)
    statements = answer.statements
    assert [s.id for s in statements] == [0, 1, 2, 3, 4]
    assert [s.topic_id for s in statements] == [0, 1, 1, 2, 2]
    assert statements[3].words == 8


def test_lead_ins_become_topic_labels_and_markdown_is_removed():
    answer = segment_answer("kw", LEAD_INS)
    assert labels(answer) == [INTRO_LABEL, "Kosten", "Förderung", "Dauer"]
    assert answer.topics[0].statements[0].text == "Die Kosten einer Dachsanierung hängen von mehreren Faktoren ab"
    assert answer.topics[1].statements[0].text == "Eine Neueindeckung kostet 100 bis 150 Euro pro Quadratmeter."


def test_bullets_and_numbering_are_stripped_and_stay_in_one_topic():
    answer = segment_answer("kw", BULLETS)
    assert labels(answer) == ["Wichtige Punkte bei der Dachsanierung"]
    assert [s.text for s in answer.statements] == [
        "Die Statik des Dachstuhls muss vorab geprüft werden.",
        "Eine Dämmung ist nach dem Gebäudeenergiegesetz oft Pflicht.",
        "Ein Gerüst ist für die Arbeiten zwingend erforderlich.",
    ]


def test_markdown_heading_is_recognised():
    answer = segment_answer("kw", "## Kosten\nDie Sanierung kostet viel Geld.\n## Dauer\nSie dauert zwei Wochen.")
    assert labels(answer) == ["Kosten", "Dauer"]


def test_blank_lines_split_topics_only_without_explicit_markers():
    text = "Der erste Absatz hat einen vollständigen Satz mit Punkt.\n\nDer zweite Absatz hat ebenfalls einen vollständigen Satz mit Punkt."
    answer = segment_answer("kw", text)
    assert len(answer.topics) == 2
    assert answer.topics[0].label == INTRO_LABEL
    assert answer.topics[1].label == "Der zweite Absatz hat ebenfalls einen vollständigen Satz …"


def test_short_lines_without_statements_are_kept_as_statements():
    answer = segment_answer("kw", "Dachdämmung\nFassadendämmung\nKellerdämmung")
    assert sorted(s.text for s in answer.statements) == ["Dachdämmung", "Kellerdämmung"]
    assert "Fassadendämmung" in labels(answer)


def test_long_lines_are_split_into_sentences():
    sentence = "Dieser Satz hat genau zehn Wörter und endet mit Punkt."
    line = " ".join([sentence] * 7)  # 70 words on one line
    answer = segment_answer("kw", "Kosten\n" + line)
    assert len(answer.statements) == 7
    assert all(s.text == sentence for s in answer.statements)


def test_semantic_fallback_splits_prose_at_the_topic_change():
    answer = segment_answer("kw", PROSE, embed=fake_embed())
    assert answer.method == "semantic"
    assert [len(t.statements) for t in answer.topics] == [3, 3]
    assert answer.topics[1].label.startswith("Eine Förderung zahlt das BAFA")


def test_no_fallback_without_embed_function_or_with_few_sentences():
    assert segment_answer("kw", PROSE).method == "structure"
    short = "Die Kosten liegen bei 200 Euro. Eine Förderung zahlt das BAFA."
    answer = segment_answer("kw", short, embed=fake_embed())
    assert answer.method == "structure"
    assert len(answer.topics) == 1


def test_no_fallback_when_structure_found_several_topics():
    calls = []

    def embed(texts):
        calls.append(texts)
        return fake_embed()(texts)

    segment_answer("kw", ANSWER, embed=embed)
    assert calls == []


def test_derive_target_uses_median_and_clamps():
    assert derive_target([20, 30]) == (25, 25.0)
    assert derive_target([5, 5, 6]) == (15, 5.0)
    assert derive_target([100, 100]) == (80, 100.0)


def test_target_comes_from_statements_and_can_be_overridden():
    answer = segment_answer("kw", ANSWER)
    assert answer.median_words == 12.0
    assert answer.target_words == 15
    assert segment_answer("kw", ANSWER, target_override=40).target_words == 40


@pytest.mark.parametrize("text", ["", "   \n \n", "## \n**  **"])
def test_empty_answer_raises_german_error(text):
    with pytest.raises(ValueError, match="keine erkennbare Aussage"):
        segment_answer("kw", text)
