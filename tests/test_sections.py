from aam.models import OWN, Document
from aam.sections import (HEADING, Section, heading_level, plain_text, section_level, split_sections,
                          strip_heading_marker)


def doc(text: str) -> Document:
    return Document("https://me.de/a", text, OWN)


PAGE = "\n".join([
    "# Dachsanierung",
    "Einleitung mit einigen Worten.",
    "## Kosten",
    "Die Sanierung kostet viel.",
    "### Material",
    "Ziegel sind teuer.",
    "## Förderung",
    "Die KfW fördert das.",
])


def test_heading_helpers():
    assert HEADING.match("## Titel")
    assert not HEADING.match("Kein ## Titel")
    assert not HEADING.match("#Titel")
    assert strip_heading_marker("### Kosten  ") == "Kosten"
    assert strip_heading_marker("Normaler Text") == "Normaler Text"
    assert heading_level("## Kosten") == 2
    assert heading_level("###### Tief") == 6
    assert heading_level("Text") is None


def test_section_level_is_smallest_level_above_one_that_occurs():
    assert section_level(PAGE) == 2
    assert section_level("# T\n### A\ntext\n### B\ntext") == 3
    assert section_level("# T\n## A\ntext\n### B\ntext\n### C\ntext") == 2
    assert section_level("# T\n## A\ntext") == 2
    assert section_level("# T\n### A\ntext\n## B\ntext") == 2
    assert section_level("Nur Text") is None
    assert section_level("# A\ntext\n# B\ntext") is None


def test_h2_page_with_intro_is_split_with_h1_titled_intro():
    sections = split_sections(doc(PAGE))
    assert [(s.index, s.title, s.level) for s in sections] == [
        (0, "Dachsanierung", 0), (1, "Kosten", 2), (2, "Förderung", 2)]
    assert sections[0].text == "Einleitung mit einigen Worten."
    assert sections[1].text == "Die Sanierung kostet viel.\nMaterial\nZiegel sind teuer."
    assert sections[2].text == "Die KfW fördert das."
    assert [s.words for s in sections] == [4, 8, 4]
    assert all(s.url == "https://me.de/a" and s.origin == OWN for s in sections)
    assert isinstance(sections[0], Section)


def test_intro_without_h1_is_called_einleitung():
    sections = split_sections(doc("Vorab etwas Text.\n## A\nText a.\n## B\nText b."))
    assert [s.title for s in sections] == ["Einleitung", "A", "B"]


def test_page_with_only_h3_uses_level_three():
    sections = split_sections(doc("### A\nText a.\n### B\nText b."))
    assert [(s.title, s.level) for s in sections] == [("A", 3), ("B", 3)]


def test_page_with_a_single_h2_is_split_into_intro_and_that_section():
    sections = split_sections(doc("# Mein Titel\nEins.\n## Nur eine\nZwei."))
    assert [(s.index, s.title, s.level, s.text) for s in sections] == [
        (0, "Mein Titel", 0, "Eins."), (1, "Nur eine", 2, "Zwei.")]


def test_page_without_headings_is_gesamte_seite():
    sections = split_sections(doc("Absatz eins hier.\nAbsatz zwei hier."))
    assert [(s.title, s.level, s.words) for s in sections] == [("Gesamte Seite", 0, 6)]
    assert sections[0].text == "Absatz eins hier.\nAbsatz zwei hier."


def test_sections_without_body_are_dropped_and_indices_stay_consecutive():
    sections = split_sections(doc("## Leer\n## A\nText a.\n## B\n## C\nText c."))
    assert [(s.index, s.title) for s in sections] == [(0, "A"), (1, "C")]


def test_empty_document_has_no_sections():
    assert split_sections(doc("")) == []
    assert split_sections(doc("## A\n## B")) == []


def test_markers_never_end_up_in_text_and_trailing_spaces_are_tolerated():
    sections = split_sections(doc("## A   \nText a.\n#### Tief  \nMehr.\n## B\nText b."))
    assert [s.title for s in sections] == ["A", "B"]
    assert sections[0].text == "Text a.\nTief\nMehr."
    assert all("#" not in s.text for s in sections)


def test_late_h1_does_not_title_the_intro_and_stays_a_plain_line():
    sections = split_sections(doc("Vorab Text.\n## A\na\n## B\nb\n# Ende\nz"))
    assert [(s.title, s.level) for s in sections] == [("Einleitung", 0), ("A", 2), ("B", 2)]
    assert sections[2].text == "b\nEnde\nz"
    assert all("#" not in s.text for s in sections)


def test_second_h1_inside_the_intro_is_kept_as_a_plain_line():
    sections = split_sections(doc("# T\nintro\n# U\nmore\n## A\na\n## B\nb"))
    assert [s.title for s in sections] == ["T", "A", "B"]
    assert sections[0].text == "intro\nU\nmore"


def test_crlf_line_endings_give_clean_titles():
    sections = split_sections(doc("# T\r\n## A\r\na\r\n## B\r\nb"))
    assert [s.title for s in sections] == ["A", "B"]
    assert [s.text for s in sections] == ["a", "b"]


def test_plain_text_strips_markers_and_empty_lines():
    assert plain_text("## Kosten\n\n  Text hier.  \n### Tief\r\nMehr.") == "Kosten\nText hier.\nTief\nMehr."
    assert plain_text(doc("# T\nx")) == "T\nx"
