from aam.dom_extract import extract_dom
from aam.extract import extract_text

FILLER = ("<p>Eine Dachsanierung kostet je nach Umfang zwischen 100 und 300 Euro pro Quadratmeter. Der Preis hängt "
          "vom Material, der Dachform und dem Zustand des Dachstuhls ab. Die Arbeiten dauern bei einem Einfamilienhaus "
          "meist ein bis zwei Wochen, ein Gerüst ist für die gesamte Dauer zwingend erforderlich. Wer die Dämmung "
          "gleich mitmacht, spart später Heizkosten und bekommt eine Förderung der KfW.</p>")


def page(main: str, before: str = "", after: str = "") -> str:
    return f"<html><head><title>T</title></head><body>{before}<main>{main}</main>{after}</body></html>"


def lines(html) -> list[str]:
    return extract_dom(html).split("\n")


def test_headings_paragraphs_lists_and_tables_in_reading_order():
    html = page("<h1>Dachsanierung</h1>" + FILLER + "<h2>Kosten</h2><ul><li>Ziegel 100 Euro</li><li>Schiefer 200 Euro</li></ul>"
                "<table><tr><th>Material</th><th>Preis</th></tr><tr><td>Ziegel</td><td>100 Euro</td></tr></table>")
    out = lines(html)
    assert out[0] == "# Dachsanierung"
    assert out[1].startswith("Eine Dachsanierung kostet")
    assert out[2:] == ["## Kosten", "Ziegel 100 Euro", "Schiefer 200 Euro", "| Material | Preis |", "| Ziegel | 100 Euro |"]


def test_navigation_header_footer_and_aside_are_dropped_but_an_article_header_with_its_heading_stays():
    html = page("<header><h1>Dachsanierung</h1><p>Lesedauer 5 Minuten</p></header>" + FILLER
                + "<aside><p>Werbung für Dachziegel hier.</p></aside><footer><p>Teilen auf Facebook</p></footer>",
                before="<header><nav><ul><li><a href='/'>Start</a></li><li><a href='/x'>Kontakt</a></li></ul></nav></header>",
                after="<footer><p>Impressum und Datenschutz</p></footer>")
    text = extract_dom(html)
    assert text.startswith("# Dachsanierung\nLesedauer 5 Minuten\n")
    for word in ("Start", "Kontakt", "Werbung", "Teilen", "Impressum"):
        assert word not in text


def test_collapsed_accordion_and_its_question_button_are_kept():
    html = page("<h1>FAQ</h1>" + FILLER + "<div class='ui-accordion-item'><h3><button>Wie viel kostet eine Kellersanierung?</button></h3>"
                "<div class='ui-accordion-item-content' hidden style='display:none'><p>Je nach Umfang zwischen 500 und 1.500 Euro pro Quadratmeter.</p></div></div>"
                "<button>Menü</button>")
    out = lines(html)
    assert "### Wie viel kostet eine Kellersanierung?" in out
    assert "Je nach Umfang zwischen 500 und 1.500 Euro pro Quadratmeter." in out
    assert "Menü" not in out


def test_table_of_contents_is_dropped_but_a_list_of_links_to_other_pages_stays():
    html = page("<h1>T</h1><h2>Inhaltsübersicht</h2><ul><li><a href='#a'>Kosten</a></li><li><a href='#b'>Förderung</a></li></ul>"
                "<h2>Kosten</h2>" + FILLER + "<h2>Weiterlesen</h2><ul><li><a href='/foerderung'>Förderung der KfW</a></li>"
                "<li><a href='/daemmung'>Dämmung des Dachs</a></li></ul><p>Noch ein Satz zum Schluss hier.</p>")
    out = lines(html)
    assert "## Inhaltsübersicht" not in out
    assert out.count("Kosten") == 0 and "## Kosten" in out
    assert "Förderung der KfW" in out and "Dämmung des Dachs" in out


def test_text_after_hydration_comments_is_kept_and_inline_elements_join_without_extra_spaces():
    html = page("<h1><!----><!--[-->Ein Welpe zieht ein<!--]--></h1>" + FILLER
                + "<h2>Kosten in m<sup>2</sup></h2><p>Die Abdichtung des <strong>Kellers </strong>als Maßnahme zur Sanierung lohnt sich.</p>")
    out = lines(html)
    assert out[0] == "# Ein Welpe zieht ein"
    assert "## Kosten in m2" in out
    assert "Die Abdichtung des Kellers als Maßnahme zur Sanierung lohnt sich." in out


def test_a_short_bold_only_paragraph_acts_as_a_heading_one_level_below_the_last_one():
    html = page("<h1>Kellersanierung</h1>" + FILLER + "<h2>Dämmung</h2><p><strong>Kellerdecke dämmen</strong></p>"
                "<p>Eine Kellerdämmung sollte bei einem unbeheizten Keller erfolgen.</p>"
                "<p><strong>Das ist ein ganzer Satz in fett.</strong></p>")
    out = lines(html)
    assert "### Kellerdecke dämmen" in out
    assert "Das ist ein ganzer Satz in fett." in out  # ends with a period: a sentence, not a heading


def test_h1_outside_the_main_container_is_prepended():
    html = page(FILLER + "<h2>Kosten</h2><p>Die Kosten hängen vom Material ab, sagt jeder Dachdecker.</p>",
                before="<div class='hero'><h1>Ein Welpe zieht ein – so klappt der Start</h1></div>")
    assert lines(html)[0] == "# Ein Welpe zieht ein – so klappt der Start"


def test_teaser_sections_cards_and_boilerplate_blocks_are_dropped():
    html = page("<h1>T</h1>" + FILLER + "<div id='author-box'><p>Die Redaktion besteht aus einem engagierten Team.</p></div>"
                "<div class='similar-articles'><article><h3>Mein Welpe beißt</h3><p>Was tun bei spitzen Zähnen.</p></article></div>"
                "<a href='/x' class='x'><h5>Die Hundesteuer</h5><p>Alle Informationen auf einen Blick.</p></a>"
                "<h2>Das könnte dich auch interessieren</h2><p>Dach dämmen Schritt für Schritt.</p><h3>Wand gestalten</h3>"
                "<p>Mit Sanierputz verputzen.</p><h2>Fazit</h2><p>Eine Kellersanierung lohnt sich fast immer.</p>")
    text = extract_dom(html)
    for gone in ("Redaktion", "Welpe beißt", "Hundesteuer", "Dach dämmen", "Wand gestalten", "Sanierputz", "interessieren"):
        assert gone not in text
    assert text.endswith("## Fazit\nEine Kellersanierung lohnt sich fast immer.")


def test_duplicate_blocks_soft_hyphens_and_declared_encoding():
    html = page("<h1>T</h1>" + FILLER + FILLER + "<p>Schließ­muskeln kontrollieren ist später möglich.</p>")
    text = extract_dom(html)
    assert text.count("Eine Dachsanierung kostet") == 1
    assert "Schließmuskeln" in text
    latin = ("<html><head><meta charset='iso-8859-1'></head><body><main><h1>Dämmung</h1>" + FILLER + "</main></body></html>").encode("latin-1")
    assert extract_dom(latin).startswith("# Dämmung")


def test_a_too_short_page_yields_nothing_and_extract_text_falls_back_to_trafilatura():
    short = "<html><body><article><h1>T</h1><h2>Kosten</h2><p>Die Sanierung kostet viel Geld pro Quadratmeter heute.</p></article></body></html>"
    assert extract_dom(short) == ""
    assert "Die Sanierung kostet" in extract_text(short)
    assert extract_dom("<html><body></body></html>") == ""


def test_role_navigation_and_empty_headings_are_dropped():
    html = page("<h1>T</h1><div role='navigation'><p>Start Kontakt Impressum</p></div>" + FILLER + "<h2><span class='icon'></span></h2>"
                "<h2>Kosten</h2><h2>Leer</h2><h2>Förderung</h2><p>Die KfW zahlt Zuschüsse.</p>")
    out = lines(html)
    assert "Start Kontakt Impressum" not in out
    assert "## Kosten" not in out and "## Leer" not in out  # nothing of their own
    assert "## Förderung" in out and not any(line.strip() in ("##", "## ") for line in out)
