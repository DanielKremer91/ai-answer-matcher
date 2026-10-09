import httpx
import respx

from aam.extract import extract_text, fetch_documents
from aam.models import SOURCE

HTML = """<html><head><title>Dachsanierung Kosten</title></head><body>
<nav><a href="/">Startseite</a><a href="/kontakt">Kontakt</a></nav>
<article><h1>Dachsanierung Kosten</h1>
<p>Eine Dachsanierung kostet je nach Umfang zwischen 100 und 300 Euro pro Quadratmeter. Der Preis hängt vom Material, der Dachform und dem Zustand des Dachstuhls ab.</p>
<p>Die KfW fördert die Dachdämmung mit zinsgünstigen Krediten. Das BAFA zahlt zusätzlich Zuschüsse von bis zu 20 Prozent der förderfähigen Kosten.</p>
<p>Die Arbeiten dauern bei einem Einfamilienhaus meist ein bis zwei Wochen. Ein Gerüst ist für die gesamte Dauer zwingend erforderlich.</p>
</article><footer>Impressum und Datenschutz</footer></body></html>"""


def test_extract_text_returns_paragraphs_on_separate_lines():
    text = extract_text(HTML)
    lines = text.split("\n")
    assert any("KfW" in line for line in lines)
    assert any("Gerüst" in line for line in lines)
    assert len([line for line in lines if line.strip()]) >= 3


def test_extract_text_marks_headings_and_strips_bold_markers():
    lines = extract_text(HTML).split("\n")
    assert lines[0].startswith("# ") and "Dachsanierung Kosten" in lines[0]
    assert all(line.strip() for line in lines)
    bold = "<html><body><article><h1>T</h1><h2>Kosten</h2><p>Die <b>Sanierung</b> kostet viel Geld pro Quadratmeter heute.</p></article></body></html>"
    text = extract_text(bold)
    assert "**" not in text and "__" not in text
    assert any(line.startswith("## ") and "Kosten" in line for line in text.split("\n"))
    assert "Die Sanierung kostet" in text


def test_extract_text_drops_markdown_table_separator_rows():
    table = ("<html><body><article><h1>T</h1><p>Eine Dachsanierung kostet je nach Umfang zwischen 100 und 300 "
             "Euro pro Quadratmeter, wie üblich.</p><table><tr><th>Material</th><th>Preis</th></tr>"
             "<tr><td>Ziegel</td><td>100 Euro</td></tr></table></article></body></html>")
    lines = extract_text(table).split("\n")
    assert "| Material | Preis |" in lines and "| Ziegel | 100 Euro |" in lines
    assert not any(set(line) <= set("|-: ") for line in lines)


def test_extract_text_empty_page():
    assert extract_text("<html><body></body></html>") == ""


@respx.mock
def test_fetch_documents_collects_documents_and_failures_and_reports_progress():
    respx.get("https://ok.de/a").mock(return_value=httpx.Response(200, html=HTML))
    respx.get("https://blocked.de/").mock(return_value=httpx.Response(403))
    respx.get("https://slow.de/").mock(side_effect=httpx.ReadTimeout("slow"))
    respx.get("https://down.de/").mock(side_effect=httpx.ConnectError("down"))
    respx.get("https://empty.de/").mock(return_value=httpx.Response(200, html="<html><body></body></html>"))
    seen = []
    urls = ["https://ok.de/a", "https://blocked.de/", "https://slow.de/", "https://down.de/", "https://empty.de/"]
    with httpx.Client() as client:
        docs, failed = fetch_documents(urls, SOURCE, client, progress=lambda done, total: seen.append((done, total)))
    assert [(d.url, d.origin) for d in docs] == [("https://ok.de/a", SOURCE)]
    assert "KfW" in docs[0].text
    assert {f.url: f.reason for f in failed} == {
        "https://blocked.de/": "HTTP 403",
        "https://slow.de/": "Timeout nach 20 s",
        "https://down.de/": "Verbindungsfehler (ConnectError)",
        "https://empty.de/": "kein Main Content gefunden",
    }
    assert all(f.origin == SOURCE for f in failed)
    assert seen == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]


@respx.mock
def test_fetch_documents_sends_user_agent_and_follows_redirects():
    respx.get("https://old.de/").mock(return_value=httpx.Response(301, headers={"Location": "https://new.de/"}))
    route = respx.get("https://new.de/").mock(return_value=httpx.Response(200, html=HTML))
    with httpx.Client() as client:
        docs, failed = fetch_documents(["https://old.de/"], SOURCE, client)
    assert [d.url for d in docs] == ["https://old.de/"] and failed == []
    assert "AIAnswerMatcher" in route.calls[0].request.headers["User-Agent"]


@respx.mock
def test_fetch_documents_rejects_non_html_content_types_without_extracting(monkeypatch):
    def boom(html):
        raise AssertionError("extractor must not be called")

    monkeypatch.setattr("aam.extract.extract_text", boom)
    respx.get("https://pdf.de/a.pdf").mock(
        return_value=httpx.Response(200, content=b"%PDF-1.4", headers={"Content-Type": "application/pdf; charset=binary"}))
    with httpx.Client() as client:
        docs, failed = fetch_documents(["https://pdf.de/a.pdf"], SOURCE, client)
    assert docs == []
    assert [(f.url, f.reason) for f in failed] == [("https://pdf.de/a.pdf", "kein HTML (application/pdf)")]


@respx.mock
def test_fetch_documents_passes_responses_without_content_type_to_the_extractor():
    respx.get("https://plain.de/").mock(return_value=httpx.Response(200, content=HTML.encode("utf-8")))
    with httpx.Client() as client:
        docs, failed = fetch_documents(["https://plain.de/"], SOURCE, client)
    assert [d.url for d in docs] == ["https://plain.de/"] and failed == []


@respx.mock
def test_fetch_documents_turns_extractor_exceptions_into_failures_and_continues(monkeypatch):
    def flaky(html):
        if b"bad" in html:
            raise RuntimeError("lxml exploded")
        return "Text"

    monkeypatch.setattr("aam.extract.extract_text", flaky)
    respx.get("https://bad.de/").mock(return_value=httpx.Response(200, html="<html>bad</html>"))
    respx.get("https://good.de/").mock(return_value=httpx.Response(200, html="<html>good</html>"))
    with httpx.Client() as client:
        docs, failed = fetch_documents(["https://bad.de/", "https://good.de/"], SOURCE, client)
    assert [d.url for d in docs] == ["https://good.de/"]
    assert [(f.url, f.reason) for f in failed] == [("https://bad.de/", "kein Main Content gefunden")]


def test_extract_text_removes_inline_markdown_residue():
    html = """<html><body><article><h1>Dachsanierung <i>heute</i></h1><h2>Kosten <code>x</code> in m<sup>2</sup></h2>
<p>Der Preis liegt bei 100 Euro zzgl. MwSt.* pro Quadratmeter, ein <i>kursiver</i> Text mit <code>snake_case</code> und __init__ und <del>alt</del>.</p>
<ul><li>Erster Punkt der Liste hier lang genug.</li><li>Zweiter Punkt der Liste hier auch.</li></ul>
<table><tr><th>Art</th><th>Preis</th></tr><tr><td>Ziegel</td><td>100 Euro</td></tr></table>
<p>Am 1. Mai ruht die Arbeit auf allen Baustellen in ganz Deutschland komplett.</p>
</article></body></html>"""
    lines = extract_text(html).split("\n")
    assert lines[0].startswith("# ") and "*" not in lines[0]
    heading = next(line for line in lines if line.startswith("## "))
    assert heading == "## Kosten x in m2"
    for line in lines:
        # only the literal footnote star of the text may remain
        assert "*" not in line.replace("MwSt.*", "")
        assert "`" not in line and "<sup>" not in line and "\\" not in line and "~~" not in line
    price = next(line for line in lines if "MwSt" in line)
    assert "kursiver Text mit snake_case und __init__ und alt." in price and "MwSt.* pro" in price
    assert "Erster Punkt der Liste hier lang genug." in lines
    assert "Zweiter Punkt der Liste hier auch." in lines
    assert not any(line.startswith("- ") for line in lines)
    assert any(line.startswith("|") and line.count("|") >= 3 and "Ziegel" in line for line in lines)
    assert any(line.startswith("Am 1. Mai") for line in lines)
