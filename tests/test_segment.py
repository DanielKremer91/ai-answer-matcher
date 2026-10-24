import numpy as np
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
    assert labels(answer) == [INTRO_LABEL]
    assert [s.text for s in answer.statements] == ["Dachdämmung", "Fassadendämmung", "Kellerdämmung"]


UNPUNCTUATED = [
    "Eine Dachsanierung kostet je nach Umfang zwischen 100 und 300 Euro.",
    "Die Kosten hängen von mehreren Faktoren ab",
    "Dachfläche und Material",
    "Die Fläche bestimmt den Preis pro Quadratmeter",
    "Dämmung und Gerüst",
    "Beides erhöht den Aufwand erheblich",
]


def test_unpunctuated_short_lines_are_not_headings_when_followed_by_heading_like_lines():
    answer = segment_answer("kw", "\n".join(UNPUNCTUATED))
    assert [(t.label, [s.text for s in t.statements]) for t in answer.topics] == [
        (INTRO_LABEL, UNPUNCTUATED)]
    assert answer.method == "structure"


def test_real_heading_followed_by_bullet_or_punctuated_sentence_still_works():
    text = "Kosten im Überblick\n- Die Neueindeckung kostet 100 Euro.\nFörderung\nDas BAFA zahlt Zuschüsse."
    answer = segment_answer("kw", text)
    assert [(t.label, [s.text for s in t.statements]) for t in answer.topics] == [
        ("Kosten im Überblick", ["Die Neueindeckung kostet 100 Euro."]),
        ("Förderung", ["Das BAFA zahlt Zuschüsse."]),
    ]


def test_forced_heading_stays_a_heading_before_another_heading_like_line():
    answer = segment_answer("kw", "## Kosten\nDachfläche und Material\nBeides erhöht den Aufwand erheblich")
    assert [(t.label, [s.text for s in t.statements]) for t in answer.topics] == [
        ("Kosten", ["Dachfläche und Material", "Beides erhöht den Aufwand erheblich"])]


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


def test_identical_sentences_do_not_cause_a_semantic_split():
    text = " ".join(["Die Kosten der Dachsanierung liegen bei 200 Euro."] * 6)
    answer = segment_answer("kw", text, embed=fake_embed())
    assert answer.method == "structure"
    assert len(answer.topics) == 1


def test_nearly_identical_embeddings_do_not_cause_a_semantic_split():
    rng = np.random.default_rng(0)
    base = rng.normal(size=16)
    vectors = [base + rng.normal(scale=1e-4, size=16) for _ in range(8)]
    text = " ".join(f"Die Kosten der Dachsanierung liegen bei {100 + i} Euro." for i in range(8))
    answer = segment_answer("kw", text, embed=lambda texts: vectors)
    assert answer.method == "structure"
    assert len(answer.topics) == 1


KELLER = """Eine Kellersanierung kostet im Durchschnitt 15.000 bis 30.000 Euro, je nach Zustand des Gebäudes und Umfang der Arbeiten. Die Maßnahmen reichen von einfacher Trockenlegung und Putzarbeiten bis hin zur aufwendigen Außendämmung und Abdichtung.

Wichtige Schritte einer Kellersanierung
Ursachenanalyse: Ein Bausachverständiger prüft zuerst, ob die Feuchtigkeit durch Kondenswasser, aufsteigende Nässe oder seitlich eindringendes Wasser entsteht.
Trockenlegung & Abdichtung:
Bei aufsteigender Feuchtigkeit hilft eine nachträgliche Horizontalsperre (z. B. durch Injektionsverfahren).
Bei seitlicher Feuchtigkeit wird eine Außenabdichtung (Freilegen und Bitumenanstrich) oder alternativ eine Innenabdichtung vorgenommen.
Sanierputz: Alter, salz- oder schimmelbelasteter Putz wird abgeschlagen und durch speziellen, atmungsaktiven Sanierputz ersetzt.
Dämmung: Eine Perimeterdämmung von außen oder eine Innendämmung senkt den Wärmeverlust und beugt Kondenswasser vor.

Kostenübersicht (Richtwerte)
Bausachverständiger zur Schadensanalyse: ca. 200 Euro
Wärmedämmung / Innendämmung: ca. 70 bis 100 Euro pro Quadratmeter
Horizontalsperre per Injektion: ca. 200 Euro pro Quadratmeter bzw. per Mauersägeverfahren 200 bis 250 Euro pro laufendem Meter
Außenabdichtung und Dämmen: ca. 40 bis 60 Euro pro Quadratmeter (zzgl. Erdaushub)
Gesamtkosten für ein Haus (ca. 10x10 m Grundfläche): 20.000 bis 30.000 Euro
"""


def test_real_ai_overview_keeps_headings_as_topics_and_lead_ins_as_statements():
    answer = segment_answer("kellersanierung kosten", KELLER)
    assert labels(answer) == [INTRO_LABEL, "Wichtige Schritte einer Kellersanierung", "Kostenübersicht (Richtwerte)"]
    steps = [s.text for s in answer.topics[1].statements]
    assert steps[0].startswith("Ursachenanalyse: Ein Bausachverständiger prüft")
    assert steps[1].startswith("Trockenlegung & Abdichtung: Bei aufsteigender Feuchtigkeit")
    assert steps[2].startswith("Trockenlegung & Abdichtung: Bei seitlicher Feuchtigkeit")
    assert steps[3].startswith("Sanierputz: Alter")
    assert steps[4].startswith("Dämmung: Eine Perimeterdämmung")
    assert len(steps) == 5
    costs = [s.text for s in answer.topics[2].statements]
    assert costs[0] == "Bausachverständiger zur Schadensanalyse: ca. 200 Euro"
    assert costs[-1] == "Gesamtkosten für ein Haus (ca. 10x10 m Grundfläche): 20.000 bis 30.000 Euro"
    assert len(costs) == 5


def test_lead_ins_without_a_heading_above_still_open_topics():
    answer = segment_answer("kw", "Kosten: Die Sanierung kostet 200 Euro.\nDauer: Sie dauert zwei Wochen.")
    assert labels(answer) == ["Kosten", "Dauer"]


def test_label_only_line_without_heading_opens_a_topic():
    answer = segment_answer("kw", "Kosten:\nDie Sanierung kostet 200 Euro.\nDauer:\nSie dauert zwei Wochen.")
    assert labels(answer) == ["Kosten", "Dauer"]
    assert answer.topics[0].statements[0].text == "Die Sanierung kostet 200 Euro."


def test_content_factor_scales_the_target_but_not_the_median():
    answer = segment_answer("kw", ANSWER, content_factor=1.5)
    assert answer.median_words == 12.0 and answer.target_words == 18 and answer.content_factor == 1.5
    assert derive_target([20, 30], 2.0) == (50, 25.0)
    assert derive_target([5, 5, 6], 1.5) == (15, 5.0)  # still clamped


from aam.segment import split_statement


FAHRT = ("Die Fahrt: Transportieren Sie den Welpen gesichert in einer Transportbox oder auf dem Schoß einer Begleitperson "
         "(ausgelegt mit einem Handtuch für eventuelle Reiseübelkeit). Bringen Sie vom Züchter eine Decke mit, die nach "
         "der Mutter und den Geschwistern riecht – das spendet Trost.")


def test_a_bullet_with_several_sentences_becomes_several_statements_with_its_label():
    parts = split_statement(FAHRT)
    assert len(parts) == 2
    assert parts[0].startswith("Die Fahrt: Transportieren Sie den Welpen") and parts[0].endswith("Reiseübelkeit).")
    assert parts[1].startswith("Die Fahrt: Bringen Sie vom Züchter") and parts[1].endswith("das spendet Trost.")


def test_sentences_that_refer_back_or_are_short_stay_with_their_predecessor():
    text = ("Geben Sie dem Welpen in den ersten Tagen viel Ruhe. Das ist wichtig für ihn. Dann klappt die Eingewöhnung. "
            "Sie sollten Besuch in der ersten Woche vermeiden. Ganz sicher.")
    parts = split_statement(text)
    assert parts == ["Geben Sie dem Welpen in den ersten Tagen viel Ruhe. Das ist wichtig für ihn. Dann klappt die Eingewöhnung.",
                     "Sie sollten Besuch in der ersten Woche vermeiden. Ganz sicher."]


def test_single_sentence_lines_and_lines_that_collapse_to_one_group_stay_whole():
    long = "Eine Dachsanierung kostet je nach Umfang, Material, Dachform und Zustand des Dachstuhls zwischen 100 und 300 Euro pro Quadratmeter, wie Fachbetriebe berichten."
    assert split_statement(long) == [long]
    referring = "Ruhe geben. Das ist wichtig. Dann klappt es."
    assert split_statement(referring) == [referring]


def test_a_short_first_sentence_joins_the_next_one():
    text = "Kurz gesagt. Die KfW fördert die Dachdämmung mit zinsgünstigen Krediten. Das BAFA zahlt Zuschüsse von bis zu 20 Prozent."
    assert split_statement(text) == ["Kurz gesagt. Die KfW fördert die Dachdämmung mit zinsgünstigen Krediten.",
                                     "Das BAFA zahlt Zuschüsse von bis zu 20 Prozent."]


def test_segment_answer_splits_bullets_inside_a_topic_and_keeps_the_label():
    text = "Ankunft\n- " + FAHRT + "\n- Erste Nacht: Lassen Sie den Welpen nah bei sich schlafen."
    answer = segment_answer("welpe", text)
    [topic] = answer.topics
    assert [s.text[:30] for s in topic.statements] == ["Die Fahrt: Transportieren Sie ", "Die Fahrt: Bringen Sie vom Züc",
                                                        "Erste Nacht: Lassen Sie den We"]


AI_MODE = """Eine Mehrspartenhauseinführung (MSH) ist das moderne Standardsystem, um alle wichtigen Versorgungsleitungen gebündelt in ein Gebäude zu führen. 
SH-Netz-Shop
 +3
🛠️ Welche Leitungen werden durchgeführt?
Typischerweise bündelt das System die vier klassischen Versorgungssparten: 
Bayernwerk
 +1
Strom (Starkstrom-/Einspeisekabel)
Wasser (Trinkwasserzuleitung)
Telekommunikation (Telefon, DSL, Glasfaser/Breitband)
Gas oder alternativ Fernwärme 
Hauff-Technik
 +5
🏡 Die zwei Einbauarten
Welches Modell benötigt wird, hängt entscheidend von der Bauweise des Hauses ab: 
SH-Netz-Shop
Mit Keller (Wanddurchführung): Das System wird über ein Futterrohr in die Kellerwand eingesetzt. 
syna.de
 +1
Ohne Keller (Bodenplattendurchführung): Die Mehrsparte wird vor dem Gießen der Betonplatte einbetoniert. 
YouTube
·Andreas Zink
 +3
✨ Die wichtigsten Vorteile
100 % dicht: Die zertifizierten Systeme garantieren absolute Gas-, Druckwasser- und Radondichtigkeit. 
syna.de
Kosteneffizient: Es wird nur noch ein einziger Erdigraben von der Straße zum Haus benötigt. 
Hauff-Technik
 +1
⚠️ Wichtige Planungstipps für Bauherren
Die Anschaffung der Hauseinführung liegt in der Verantwortung des Bauherrn. 
Bayernwerk
Netzbetreiber fragen: Die lokalen Versorgungsunternehmen schreiben oft genau vor, welches Fabrikat zugelassen ist. 
SH-Netz-Shop
 +3
"""


def test_ai_mode_copy_with_source_chips_and_emoji_headings():
    answer = segment_answer("mehrspartenhauseinführung", AI_MODE)
    assert [t.label for t in answer.topics] == [
        "Einleitung", "Welche Leitungen werden durchgeführt?", "Die zwei Einbauarten", "Die wichtigsten Vorteile",
        "Wichtige Planungstipps für Bauherren"]
    assert [len(t.statements) for t in answer.topics] == [1, 5, 3, 2, 2]
    texts = [s.text for s in answer.statements]
    for chip in ("SH-Netz-Shop", "Bayernwerk", "Hauff-Technik", "syna.de", "YouTube", "Andreas Zink", "+3"):
        assert not any(chip in t for t in texts), chip
    assert "Strom (Starkstrom-/Einspeisekabel)" in texts
    assert set(answer.removed_lines) == {"SH-Netz-Shop", "+3", "Bayernwerk", "+1", "Hauff-Technik", "+5", "syna.de",
                                         "YouTube", "·Andreas Zink"}
    assert len(answer.removed_lines) == 18


def test_clean_pasted_keeps_ordinary_short_lines_that_are_no_chips():
    from aam.segment import clean_pasted

    text = "Kostenübersicht\nEine Sanierung kostet 100 Euro.\nFörderung\nDie KfW zahlt Zuschüsse."
    cleaned, dropped = clean_pasted(text)
    assert cleaned == text and dropped == []


def test_video_blocks_copied_with_the_answer_are_dropped():
    from aam.segment import clean_pasted

    text = ("Die Mehrsparte wird vor dem Gießen der Betonplatte einbetoniert.\nYouTube\n·Andreas Zink\n"
            "Mehrsparten-Hauseinführung richtig setzen\n12:34\n1,2 Mio. Aufrufe\nvor 2 Jahren\nWeitere Videos\n"
            "Biegesteife Mantelrohre führen die Leitungen nach draußen.")
    cleaned, dropped = clean_pasted(text)
    assert cleaned.split("\n") == ["Die Mehrsparte wird vor dem Gießen der Betonplatte einbetoniert.",
                                   "Mehrsparten-Hauseinführung richtig setzen",
                                   "Biegesteife Mantelrohre führen die Leitungen nach draußen."]
    assert dropped == ["YouTube", "·Andreas Zink", "12:34", "1,2 Mio. Aufrufe", "vor 2 Jahren", "Weitere Videos"]
