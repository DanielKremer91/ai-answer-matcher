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
