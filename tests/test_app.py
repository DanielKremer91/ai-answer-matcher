import io
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from aam import gap_report
from aam.embeddings import EmbeddingError
from aam.ingest.chunks_file import chunks_to_csv
from aam.models import OWN, SOURCE, Document, FailedUrl
from tests.fakes import ANSWER, OWN_DOCS, SOURCE_DOCS, FakeProvider, sample_result

APP = str(Path(__file__).resolve().parent.parent / "app.py")
CHUNK_BUTTON = "Segmentieren und chunken"
ANALYZE_BUTTON = "Analyse starten"
KEYWORD = "dachsanierung kosten"


def table_csv(docs) -> bytes:
    rows = ["Address,Extract Main Content 1"] + [f'{d.url},"{d.text}"' for d in docs]
    return ("\n".join(rows) + "\n").encode("utf-8")


@pytest.fixture(autouse=True)
def backend(monkeypatch):
    """Fake embeddings and a fake fetcher that knows the sample pages; everything else gets HTTP 403."""
    provider = FakeProvider()
    provider.fetched = []
    known = {d.url: d for d in OWN_DOCS + SOURCE_DOCS}

    def fetch(urls, origin, client, **kwargs):
        provider.fetched.append((list(urls), origin))
        docs = [known[u] for u in urls if u in known]
        failed = [FailedUrl(u, origin, "HTTP 403") for u in urls if u not in known]
        return docs, failed

    provider.known = known  # tests may add pages to let a previously failing URL load
    monkeypatch.setattr("aam.embeddings.make_provider", lambda *args, **kwargs: provider)
    monkeypatch.setattr("aam.extract.fetch_documents", fetch)
    return provider


def button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def has_state(at: AppTest, key: str) -> bool:
    try:
        at.session_state[key]
    except KeyError:
        return False
    return True


def new_app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_input(key="keyword").input(KEYWORD)
    at.text_area(key="answer").input(ANSWER)
    return at


def start(*, source_file: bool = False, source_urls: str = "", own_urls: str = "") -> AppTest:
    at = new_app()
    if source_urls:
        at.text_area(key="source_urls").input(source_urls)
    if own_urls:
        at.text_area(key="own_urls").input(own_urls)
    else:
        at.file_uploader(key="own_upload").set_value(("frog.csv", table_csv(OWN_DOCS), "text/csv"))
    if source_file:
        at.file_uploader(key="source_upload").set_value(("quellen.csv", table_csv(SOURCE_DOCS), "text/csv"))
    return at.run()


def chunked(**kwargs) -> AppTest:
    at = start(**kwargs)
    button(at, CHUNK_BUTTON).click()
    return at.run()


def analyzed(**kwargs) -> AppTest:
    at = chunked(**kwargs)
    button(at, ANALYZE_BUTTON).click()
    return at.run()


def test_first_render_shows_four_steps_and_disabled_buttons():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert [h.value[0] for h in at.header if h.value[0].isdigit()] == ["1", "2", "3", "4", "5"]
    assert button(at, CHUNK_BUTTON).disabled is True
    assert button(at, ANALYZE_BUTTON).disabled is True
    assert at.sidebar.selectbox[0].value == "openai"


def test_upload_note_names_the_recognised_columns():
    at = start()
    assert not at.exception
    assert any("Address" in c.value and "Extract Main Content 1" in c.value and "2 Seiten" in c.value
               for c in at.caption)
    assert button(at, CHUNK_BUTTON).disabled is False
    assert button(at, ANALYZE_BUTTON).disabled is True


def test_upload_note_uses_the_singular_for_one_page():
    at = new_app()
    at.file_uploader(key="own_upload").set_value(("frog.csv", table_csv(OWN_DOCS[:1]), "text/csv"))
    at.run()
    assert any("1 Seite" in c.value and "1 Seiten" not in c.value for c in at.caption)


def test_chunking_step_shows_protocol_ai_chunks_and_embedding_count(backend):
    at = chunked(source_file=True)
    assert not at.exception and not at.error
    protocol = at.code[0].value
    assert "AI-Antwort: 3 Themen, 5 Aussagen" in protocol and "Median 12 Wörter × 1,5 → Zielgröße 18 Wörter" in protocol
    assert "Eigener Content: 2 Seiten → 3 Chunks" in protocol
    assert "Quellen: 1 Seite → 1 Chunk" in protocol  # target 18: the two short source paragraphs merge
    frame = at.dataframe[0].value
    assert list(frame.columns) == ["Thema", "Aussage", "Wortzahl"] and len(frame) == 5
    assert any("bis zu 11 Texte" in c.value for c in at.caption)
    assert backend.calls == 0 and backend.fetched == []
    assert button(at, ANALYZE_BUTTON).disabled is False


def test_full_run_with_source_file_shows_summary_tables_and_downloads():
    at = analyzed(source_file=True)
    assert not at.exception and not at.error
    assert at.session_state["result"].match.has_sources is True
    assert any(m.value.startswith("Relevanteste eigene Seite: https://me.de/kosten") for m in at.markdown)
    frames = [d.value for d in at.dataframe]
    assert any("Seitenscore" in f.columns for f in frames)
    assert any("Anzahl Aussagen" in f.columns for f in frames)
    assert any("Status" in f.columns and "Delta" in f.columns for f in frames)
    assert len(at.get("download_button")) == 2


def test_run_without_sources_works_and_explains_the_missing_benchmark():
    at = analyzed()
    assert not at.exception and not at.error
    assert at.session_state["result"].match.has_sources is False
    assert any("keinen Benchmark" in m.value for m in at.markdown)
    statement_frames = [d.value for d in at.dataframe if "Status" in d.value.columns]
    assert statement_frames and "Delta" not in statement_frames[0].columns


def test_source_urls_are_fetched_when_no_file_covers_them(backend):
    at = analyzed(source_urls="https://quelle.de/foerderung")
    assert backend.fetched == [(["https://quelle.de/foerderung"], SOURCE)]
    assert at.session_state["result"].match.has_sources is True


def test_source_urls_covered_by_the_source_file_are_not_fetched(backend):
    at = chunked(source_file=True, source_urls="https://quelle.de/foerderung")
    assert not at.error and backend.fetched == []


def test_source_url_that_is_an_own_page_is_warned_about_and_not_used_as_source(backend):
    at = analyzed(source_urls="https://me.de/kosten")
    assert not at.exception and not at.error
    assert any("gehört zum eigenen Content" in w.value for w in at.warning)
    assert at.session_state["result"].match.has_sources is False


def test_failed_source_urls_are_listed_and_the_run_continues(backend):
    at = analyzed(source_urls="https://weg.de/x")
    assert not at.exception and not at.error
    assert "Nicht geladen (Quelle): https://weg.de/x (HTTP 403)" in at.code[0].value
    assert at.session_state["result"].match.has_sources is False


def test_own_content_can_come_from_a_url_list(backend):
    at = analyzed(own_urls="https://me.de/kosten\nhttps://me.de/blog")
    assert not at.exception and not at.error
    assert backend.fetched[0] == (["https://me.de/kosten", "https://me.de/blog"], OWN)
    assert has_state(at, "result")


def test_changed_inputs_require_new_chunking():
    at = analyzed()
    at.text_area(key="answer").input(ANSWER + "\nDauer\nDie Arbeiten dauern zwei Wochen.").run()
    assert not at.exception
    assert button(at, ANALYZE_BUTTON).disabled is True
    assert any("erneut segmentieren" in w.value for w in at.warning)
    assert len(at.get("download_button")) == 0


def test_threshold_change_rematches_without_new_embeddings(backend, monkeypatch):
    import aam.pipeline

    runs = []
    real_analyze = aam.pipeline.analyze

    def counting_analyze(*args, **kwargs):
        runs.append(1)
        return real_analyze(*args, **kwargs)

    monkeypatch.setattr("aam.pipeline.analyze", counting_analyze)
    at = analyzed(source_file=True)
    assert len(runs) == 1
    assert at.session_state["result"].meta.weak_delta == 0.15
    calls = backend.calls
    at.sidebar.slider(key="weak_delta").set_value(0.5).run()
    assert not at.exception
    assert at.session_state["result"].meta.weak_delta == 0.5
    assert len(runs) == 1  # re-matched, not analysed again
    assert backend.calls == calls


def test_missing_api_key_is_reported_at_analysis(monkeypatch):
    def no_key(*args, **kwargs):
        raise EmbeddingError("openai: API-Schlüssel fehlt")

    monkeypatch.setattr("aam.embeddings.make_provider", no_key)
    at = analyzed()
    assert not at.exception
    assert any("API-Schlüssel fehlt" in e.value for e in at.error)
    assert not has_state(at, "result")


def upload_chunk_export(model: str) -> AppTest:
    data = chunks_to_csv(sample_result().own_chunks, provider="openai", model=model, target_words=15)
    at = new_app()
    at.file_uploader(key="own_upload").set_value(("chunks_embeddings.csv", data, "text/csv"))
    return at.run()


def test_chunk_export_can_be_uploaded_instead_of_content(backend):
    at = upload_chunk_export("text-embedding-3-small")
    assert any("Chunk-Embedding-Export erkannt: 3 Chunks" in c.value for c in at.caption)
    button(at, CHUNK_BUTTON).click()
    at.run()
    assert "aus Chunk-Embedding-Export" in at.code[0].value
    button(at, ANALYZE_BUTTON).click()
    at.run()
    assert not at.exception and not at.error
    assert backend.calls == 5  # only the five statements
    assert len(at.get("download_button")) == 1  # no second chunk export for uploaded chunks


def test_chunk_export_from_another_model_is_rejected():
    at = upload_chunk_export("text-embedding-3-large")
    assert any("erzeugt, eingestellt ist openai / text-embedding-3-small" in e.value for e in at.error)
    assert button(at, CHUNK_BUTTON).disabled is True


def test_table_without_content_column_shows_the_found_columns():
    at = new_app()
    at.file_uploader(key="own_upload").set_value(("falsch.csv", b"Foo,Bar\n1,2\n", "text/csv"))
    at.run()
    assert any("Gefunden: Foo, Bar" in e.value for e in at.error)
    assert button(at, CHUNK_BUTTON).disabled is True


def excel_sheet(data: bytes, sheet: str) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(data), sheet_name=sheet)


def test_rerun_with_identical_inputs_does_not_serve_stale_downloads(backend):
    url = "https://neu.de/quelle"
    at = analyzed(source_urls=url)
    assert not at.exception and not at.error
    assert "Nicht geladen (Quelle): https://neu.de/quelle (HTTP 403)" in at.code[0].value
    first = at.session_state["excel"][1]
    assert "Quelle" not in set(excel_sheet(first, "Seiten")["Herkunft"])

    backend.known[url] = Document(url, SOURCE_DOCS[0].text, SOURCE)  # the URL loads now
    button(at, CHUNK_BUTTON).click()
    at.run()
    button(at, ANALYZE_BUTTON).click()
    at.run()
    assert not at.exception and not at.error
    assert at.session_state["result"].match.has_sources is True
    second = at.session_state["excel"][1]
    assert second != first
    assert "Quelle" in set(excel_sheet(second, "Seiten")["Herkunft"])


class FlakyProvider(FakeProvider):
    """Embeds fine until armed, then fails like a server outage."""

    armed = False

    def _embed_batch(self, texts):
        if self.armed:
            raise EmbeddingError("fake: HTTP 500")
        return super()._embed_batch(texts)


def test_failed_analysis_shows_the_error_and_leaves_no_progress_bar(monkeypatch):
    provider = FlakyProvider()
    monkeypatch.setattr("aam.embeddings.make_provider", lambda *args, **kwargs: provider)
    at = chunked()
    assert not at.exception and not at.error
    provider.armed = True
    button(at, ANALYZE_BUTTON).click()
    at.run()
    assert not at.exception
    assert any("Analyse fehlgeschlagen" in e.value and "fake: HTTP 500" in e.value for e in at.error)
    assert len(at.get("progress")) == 0
    assert not has_state(at, "result")


def test_prepare_warning_is_shown_once_after_a_full_analysis():
    at = analyzed(source_urls="https://me.de/kosten")
    assert not at.exception and not at.error
    assert has_state(at, "result")
    warnings = [w.value for w in at.warning if "gehört zum eigenen Content" in w.value]
    assert len(warnings) == 1


def test_all_own_urls_failing_shows_the_reasons_with_the_error(backend):
    at = chunked(own_urls="https://weg.de/a\nhttps://weg.de/b")
    assert not at.exception
    assert len(at.error) == 1
    message = at.error[0].value
    assert "Kein eigener Content mit Text vorhanden" in message
    assert "• https://weg.de/a (HTTP 403)" in message and "• https://weg.de/b (HTTP 403)" in message
    assert not has_state(at, "prepared")


def test_failure_reasons_are_capped_at_ten_lines(backend):
    urls = "\n".join(f"https://weg.de/{i}" for i in range(12))
    at = chunked(own_urls=urls)
    message = at.error[0].value
    assert message.count("• https://weg.de/") == 10 and "… und 2 weitere" in message


def test_reused_export_shows_the_source_target_in_the_protocol(backend):
    data = chunks_to_csv(sample_result().own_chunks, provider="openai", model="text-embedding-3-small",
                         target_words=40)
    at = new_app()
    at.file_uploader(key="own_upload").set_value(("chunks_embeddings.csv", data, "text/csv"))
    at.file_uploader(key="source_upload").set_value(("quellen.csv", table_csv(SOURCE_DOCS), "text/csv"))
    at.run()
    button(at, CHUNK_BUTTON).click()
    at.run()
    assert not at.exception
    assert "geschnitten mit Zielgröße 40 Wörter wie der hochgeladene Export" in at.code[0].value
    assert at.session_state["prepared"].source_target_words == 40
    button(at, ANALYZE_BUTTON).click()
    at.run()
    assert not at.exception
    assert at.session_state["result"].meta.source_target_words == 40


def test_chunk_export_is_built_only_on_demand(monkeypatch):
    built = []
    real = chunks_to_csv

    def counting(*args, **kwargs):
        built.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr("aam.ingest.chunks_file.chunks_to_csv", counting)
    at = analyzed()
    assert not at.exception and not at.error
    assert len(at.get("download_button")) == 2
    assert built == [] and not has_state(at, "chunks_csv")


def test_result_has_a_statement_by_page_tab_with_matrix_and_list():
    at = analyzed(source_file=True)
    assert not at.exception and not at.error
    assert any(t.label == "Aussage × Seite" for t in at.tabs)
    frames = [d.value for d in at.dataframe]
    assert any("Seiten-Relevanz" in f.columns and "Bester Chunk Score" in f.columns for f in frames)
    assert any(list(f.columns[:2]) == ["Thema", "Aussage"] and "https://me.de/kosten" in f.columns for f in frames)


def test_own_content_can_be_pasted_as_text(backend):
    at = new_app()
    at.text_area(key="own_text").input(OWN_DOCS[0].text)
    at.run()
    assert button(at, CHUNK_BUTTON).disabled is False
    button(at, CHUNK_BUTTON).click()
    at.run()
    assert not at.exception and not at.error
    assert "Eigener Content: 1 Seite" in at.code[0].value
    button(at, ANALYZE_BUTTON).click()
    at.run()
    assert not at.exception and not at.error
    pages = at.session_state["result"].match.pages
    assert [p.url for p in pages] == ["Eigener Text (eingefügt)"]
    assert backend.fetched == []


def test_chunking_switches_in_the_sidebar_reach_the_protocol():
    at = start()
    at.sidebar.checkbox(key="split_long").set_value(False)
    at.sidebar.checkbox(key="merge_short").set_value(False).run()
    button(at, CHUNK_BUTTON).click()
    at.run()
    assert not at.exception and not at.error
    assert "lange Absätze nicht geteilt" in at.code[0].value
    assert "Eigener Content: 2 Seiten → 5 Chunks" in at.code[0].value


# ---------------------------------------------------------------- modes

MODE_1, MODE_2, MODE_3 = "AI-Antwort gegen Content", "Absatz gegen Text", "Content-Gap zwischen Seiten"
SHORT_STATEMENT = "Eine Neueindeckung kostet 100 bis 150 Euro pro Quadratmeter."
LONG_TEXT = "\n".join([
    "Unser Team stellt sich vor und berät seit vielen Jahren Hausbesitzer in Köln und Umgebung.",
    "Eine Neueindeckung kostet 100 bis 150 Euro pro Quadratmeter, je nach Ziegel und Aufwand.",
    "Die Förderung der KfW hängt vom Effizienzhaus-Standard und vom Zeitpunkt des Antrags ab.",
    "Vor dem Winter sollten Hausbesitzer die Dachrinnen reinigen und lose Ziegel prüfen lassen.",
    "Bei Fragen erreichen Sie uns telefonisch oder per Formular, wir melden uns am selben Tag.",
])


def set_mode(at: AppTest, mode: str) -> AppTest:
    at.radio(key="mode").set_value(mode)
    return at.run()


def headed(doc: Document) -> Document:
    """The sample page with two H2 headings, so that it is cut into two sections."""
    first, *rest = doc.text.split("\n")
    return Document(doc.url, f"## Thema Eins\n{first}\n## Thema Zwei\n" + "\n".join(rest), doc.origin)


def gap_app(backend, *, answer: str = "") -> AppTest:
    for doc in OWN_DOCS + SOURCE_DOCS:
        backend.known[doc.url] = headed(doc)
    at = set_mode(AppTest.from_file(APP, default_timeout=30).run(), MODE_3)
    at.text_area(key="gap_competitor_urls").input("https://quelle.de/foerderung")
    at.text_area(key="gap_own_urls").input("https://me.de/kosten\nhttps://me.de/blog")
    if answer:
        at.text_area(key="gap_answer").input(answer)
    return at.run()


def test_mode_radio_offers_three_modes_and_defaults_to_the_first():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.radio(key="mode").options == [MODE_1, MODE_2, MODE_3]
    assert at.radio(key="mode").value == MODE_1
    assert at.radio(key="mode").label == "Modus"


def test_manual_mode_hides_stage_one_and_shows_the_two_fields():
    at = set_mode(AppTest.from_file(APP, default_timeout=30).run(), MODE_2)
    assert not at.exception and not at.error
    assert not [h for h in at.header if "AI-Antwort" in h.value[0] or h.value[0].isdigit()]
    assert at.text_area(key="manual_short") is not None and at.text_area(key="manual_long") is not None
    assert button(at, "Vergleichen").disabled is True
    assert not [b for b in at.button if b.label == CHUNK_BUTTON]
    # chunking switches and thresholds belong to mode 1, provider and model stay
    assert not at.sidebar.slider and not at.sidebar.checkbox and not at.sidebar.number_input
    assert at.sidebar.selectbox[0].value == "openai"


def test_manual_comparison_of_a_statement_and_a_long_text_shows_both_values_and_the_hint():
    at = set_mode(AppTest.from_file(APP, default_timeout=30).run(), MODE_2)
    at.text_area(key="manual_short").input(SHORT_STATEMENT)
    at.text_area(key="manual_long").input(LONG_TEXT)
    at.run()
    button(at, "Vergleichen").click()
    at.run()
    assert not at.exception and not at.error
    assert [m.label for m in at.metric] == ["Ganztext", "Bestes Fenster"]
    whole, best = (float(m.value.replace(",", ".")) for m in at.metric)
    assert best > whole
    assert any("Eine Neueindeckung kostet" in t.value for t in at.text)
    ratio = f"{len(LONG_TEXT.split()) / len(SHORT_STATEMENT.split()):.1f}".replace(".", ",")
    assert "," in ratio
    assert [i.value for i in at.info] == [
        f"Dein Text ist {ratio}-mal so lang wie die Aussage. Der Ganztext-Wert wird durch die "
        "Länge gedrückt; das beste Fenster zeigt, ob die Aussage an einer Stelle vorkommt."]


def test_manual_hint_names_the_statement_as_longer_when_field_a_is_the_longer_text():
    at = set_mode(AppTest.from_file(APP, default_timeout=30).run(), MODE_2)
    at.text_area(key="manual_short").input(LONG_TEXT)
    at.text_area(key="manual_long").input(SHORT_STATEMENT)
    at.run()
    button(at, "Vergleichen").click()
    at.run()
    assert not at.exception and not at.error
    ratio = f"{len(LONG_TEXT.split()) / len(SHORT_STATEMENT.split()):.1f}".replace(".", ",")
    assert [i.value for i in at.info] == [
        f"Die Aussage ist {ratio}-mal so lang wie dein Text. Der Ganztext-Wert wird durch die "
        "Länge gedrückt; das beste Fenster zeigt, ob die Aussage an einer Stelle vorkommt."]


def compare_manually(at: AppTest) -> AppTest:
    at.text_area(key="manual_short").input(SHORT_STATEMENT)
    at.text_area(key="manual_long").input(LONG_TEXT)
    at.run()
    button(at, "Vergleichen").click()
    return at.run()


def test_manual_result_survives_a_rerun_and_a_mode_round_trip():
    at = compare_manually(set_mode(AppTest.from_file(APP, default_timeout=30).run(), MODE_2))
    shown = [(m.label, m.value) for m in at.metric]
    assert shown and "manual_result" in at.session_state
    at.run()  # any rerun without a click
    assert [(m.label, m.value) for m in at.metric] == shown and at.info
    set_mode(at, MODE_1)
    assert not at.metric
    set_mode(at, MODE_2)
    assert not at.exception and not at.error and not at.warning
    assert [(m.label, m.value) for m in at.metric] == shown and at.info


def test_manual_result_is_hidden_after_the_text_or_the_model_changes():
    at = compare_manually(set_mode(AppTest.from_file(APP, default_timeout=30).run(), MODE_2))
    assert at.metric
    at.text_area(key="manual_long").input(LONG_TEXT + " Noch ein Satz.").run()
    assert not at.metric and not at.info
    at.text_area(key="manual_long").input(LONG_TEXT).run()
    assert at.metric  # the same texts show the stored result again
    at.sidebar.text_input(key="model_openai").input("text-embedding-3-large").run()
    assert not at.metric


def test_manual_error_removes_the_stored_result(monkeypatch):
    at = compare_manually(set_mode(AppTest.from_file(APP, default_timeout=30).run(), MODE_2))
    assert "manual_result" in at.session_state
    monkeypatch.setattr("aam.embeddings.make_provider",
                        lambda *a, **k: (_ for _ in ()).throw(EmbeddingError("Kein API-Key.")))
    button(at, "Vergleichen").click()
    at.run()
    assert [e.value for e in at.error] == ["Kein API-Key."]
    assert "manual_result" not in at.session_state and not at.metric


def test_manual_comparison_of_similar_lengths_shows_one_value():
    at = set_mode(AppTest.from_file(APP, default_timeout=30).run(), MODE_2)
    at.text_area(key="manual_short").input(SHORT_STATEMENT)
    at.text_area(key="manual_long").input("Eine Neueindeckung kostet 120 Euro pro Quadratmeter, je nach Ziegel.")
    at.run()
    button(at, "Vergleichen").click()
    at.run()
    assert not at.exception and not at.error
    assert [m.label for m in at.metric] == ["Kosinus"]
    assert any(m.value == "Beide Texte sind ähnlich lang, der Wert gilt direkt." for m in at.markdown)
    assert not at.info


def test_manual_comparison_reports_a_provider_error(monkeypatch):
    monkeypatch.setattr("aam.embeddings.make_provider",
                        lambda *a, **k: (_ for _ in ()).throw(EmbeddingError("Kein API-Key.")))
    at = set_mode(AppTest.from_file(APP, default_timeout=30).run(), MODE_2)
    at.text_area(key="manual_short").input(SHORT_STATEMENT)
    at.text_area(key="manual_long").input(LONG_TEXT)
    at.run()
    button(at, "Vergleichen").click()
    at.run()
    assert not at.exception
    assert [e.value for e in at.error] == ["Kein API-Key."]
    assert not at.metric


def test_gap_mode_compares_sections_and_offers_the_excel(backend):
    at = gap_app(backend)
    button(at, "Abschnitte vergleichen").click()
    at.run()
    assert not at.exception and not at.error
    frame = at.dataframe[0].value
    assert "Kosinus" in frame.columns and len(frame) == 2
    assert "Text Wettbewerber" not in frame.columns and "Text eigen" not in frame.columns
    assert "Nächste AI-Aussage" not in frame.columns
    assert list(frame["Kosinus"]) == sorted(frame["Kosinus"], reverse=True)
    assert "https://quelle.de/foerderung" in at.code[0].value and "Abschnittsebene H2" in at.code[0].value
    summary = gap_report.gap_summary(at.session_state["gap_result"])
    assert "2 Abschnitte, Kosinus von" in summary and "Kandidaten für Lücken" in summary
    assert [m.value for m in at.markdown if "Die untere Hälfte der Liste" in m.value] == [summary]
    assert "Vergleich" in frame.columns
    assert any(e.label == "Texte nachlesen" for e in at.expander)
    assert [b.key for b in at.get("download_button")] == ["gap_download"]
    assert "gap_result" in at.session_state
    assert backend.fetched  # pages came through the patched fetcher


def sentence(prefix: str, n: int) -> str:
    words = [f"{prefix}{i}" for i in range(n)]
    words[0] = words[0].capitalize()
    return " ".join(words) + "."


def test_gap_mode_texts_show_the_best_window_with_its_method(backend):
    passage = " ".join(sentence(f"pass{k}x", 12) for k in range(3))
    long_text = "\n".join([sentence(f"lang{i}x", 20) for i in range(10)] + [passage]
                          + [sentence(f"mehr{i}x", 20) for i in range(10)])
    backend.known["https://wett.de/a"] = Document(
        "https://wett.de/a", f"## Thema\n{passage}\n## Anderes\n{sentence('woanders', 30)}", SOURCE)
    backend.known["https://me.de/a"] = Document("https://me.de/a", f"## Lang\n{long_text}", OWN)
    at = set_mode(AppTest.from_file(APP, default_timeout=30).run(), MODE_3)
    at.text_area(key="gap_competitor_urls").input("https://wett.de/a")
    at.text_area(key="gap_own_urls").input("https://me.de/a")
    at.run()
    button(at, "Abschnitte vergleichen").click()
    at.run()
    assert not at.exception and not at.error
    frame = at.dataframe[0].value
    assert "Fenster aus eigenem Abschnitt" in set(frame["Vergleich"])
    expander = next(e for e in at.expander if e.label == "Texte nachlesen")
    assert any(m.value == "Bestes Fenster (Fenster aus eigenem Abschnitt)" for m in expander.markdown)
    assert any(t.value == passage for t in expander.text)


def test_single_section_page_gets_the_short_note_unless_it_has_no_headings(backend):
    backend.known["https://wett.de/a"] = Document("https://wett.de/a", "## Nur eine\nEin Absatz über Kosten.", SOURCE)
    backend.known["https://wett.de/b"] = Document("https://wett.de/b", "Ein Absatz über Kosten.", SOURCE)
    at = set_mode(AppTest.from_file(APP, default_timeout=30).run(), MODE_3)
    at.text_area(key="gap_competitor_urls").input("https://wett.de/a\nhttps://wett.de/b")
    at.text_area(key="gap_own_urls").input("https://me.de/kosten")
    at.run()
    button(at, "Abschnitte vergleichen").click()
    at.run()
    assert not at.exception and not at.error
    notes = [i.value for i in at.info]
    assert "https://wett.de/a: Die Seite hat nur einen Abschnitt." in notes
    assert "https://wett.de/b: Die Seite hat nur einen Abschnitt. Verglichen wird die ganze Seite." in notes


def test_gap_mode_with_an_answer_adds_the_next_statement(backend):
    at = gap_app(backend, answer=ANSWER)
    button(at, "Abschnitte vergleichen").click()
    at.run()
    assert not at.exception and not at.error
    frame = at.dataframe[0].value
    assert "Nächste AI-Aussage" in frame.columns and "Kosinus AI-Aussage" in frame.columns
    assert "Priorität" in frame.columns
    at.radio(key="gap_order").set_value("Priorität").run()
    assert not at.exception and not at.error
    ordered = at.dataframe[0].value
    assert list(ordered["Priorität"]) == sorted(ordered["Priorität"], reverse=True)


def test_gap_mode_reads_competitor_and_own_pages_from_uploads(backend):
    at = set_mode(AppTest.from_file(APP, default_timeout=30).run(), MODE_3)
    at.file_uploader(key="gap_competitor_upload").set_value(
        ("wettbewerb.csv", table_csv([headed(d) for d in SOURCE_DOCS]), "text/csv"))
    at.file_uploader(key="gap_own_upload").set_value(
        ("frog.csv", table_csv([headed(d) for d in OWN_DOCS]), "text/csv"))
    at.run()
    button(at, "Abschnitte vergleichen").click()
    at.run()
    assert not at.exception and not at.error
    assert "Kosinus" in at.dataframe[0].value.columns
    assert backend.fetched == []


def test_gap_mode_accepts_pasted_own_text(backend):
    backend.known["https://quelle.de/foerderung"] = headed(SOURCE_DOCS[0])
    at = set_mode(AppTest.from_file(APP, default_timeout=30).run(), MODE_3)
    at.text_area(key="gap_competitor_urls").input("https://quelle.de/foerderung")
    at.text_area(key="gap_own_text").input("Die KfW fördert die Dachdämmung mit Krediten.\nDas BAFA zahlt Zuschüsse.")
    at.run()
    button(at, "Abschnitte vergleichen").click()
    at.run()
    assert not at.exception and not at.error
    assert set(at.dataframe[0].value["Beste eigene URL"]) == {"Eigener Text (eingefügt)"}


def test_gap_mode_button_needs_competitor_and_own_pages(backend):
    at = set_mode(AppTest.from_file(APP, default_timeout=30).run(), MODE_3)
    assert button(at, "Abschnitte vergleichen").disabled is True
    at.text_area(key="gap_competitor_urls").input("https://quelle.de/foerderung")
    at.run()
    assert button(at, "Abschnitte vergleichen").disabled is True


def test_gap_mode_names_the_reason_when_no_page_loads(backend):
    at = set_mode(AppTest.from_file(APP, default_timeout=30).run(), MODE_3)
    at.text_area(key="gap_competitor_urls").input("https://nirgends.de/a")
    at.text_area(key="gap_own_urls").input("https://me.de/kosten")
    at.run()
    button(at, "Abschnitte vergleichen").click()
    at.run()
    assert not at.exception
    assert len(at.error) == 1 and "https://nirgends.de/a (HTTP 403)" in at.error[0].value
    assert "gap_result" not in at.session_state


def test_gap_result_is_hidden_after_the_inputs_change(backend):
    at = gap_app(backend)
    button(at, "Abschnitte vergleichen").click()
    at.run()
    assert at.dataframe
    at.text_area(key="gap_own_urls").input("https://me.de/kosten")
    at.run()
    assert not at.dataframe and not at.get("download_button")
    assert any("Eingaben haben sich" in w.value for w in at.warning)


def test_mode_switch_keeps_the_other_modes_inputs_and_results(backend):
    at = gap_app(backend)
    button(at, "Abschnitte vergleichen").click()
    at.run()
    set_mode(at, MODE_2)
    assert not at.exception and not at.error and not at.warning and not at.dataframe
    set_mode(at, MODE_3)
    assert not at.exception and not at.error and not at.warning
    assert at.text_area(key="gap_competitor_urls").value == "https://quelle.de/foerderung"
    assert at.dataframe and at.get("download_button")


def test_stage_one_still_runs_after_a_mode_round_trip():
    at = new_app()
    set_mode(at, MODE_3)
    set_mode(at, MODE_1)
    assert not at.exception and not at.warning
    assert at.text_input(key="keyword").value == KEYWORD
    at.file_uploader(key="own_upload").set_value(("frog.csv", table_csv(OWN_DOCS), "text/csv"))
    at.run()
    button(at, CHUNK_BUTTON).click()
    at.run()
    button(at, ANALYZE_BUTTON).click()
    at.run()
    assert not at.exception and not at.error
    assert at.session_state["result"].match.has_sources is False
    assert [h.value[0] for h in at.header if h.value[0].isdigit()] == ["1", "2", "3", "4", "5"]


# ---------------------------------------------------------------- uploads across mode switches

KEPT_CAPTION = "bleibt erhalten, bis du eine andere Datei wählst."


def result_is_shown(at: AppTest) -> bool:
    return (any("Seitenscore" in d.value.columns for d in at.dataframe)
            and not [w for w in at.warning if "geändert" in w.value])


def test_stage_one_results_survive_a_round_trip_through_another_mode():
    at = analyzed(source_file=True)
    assert result_is_shown(at)
    set_mode(at, MODE_2)
    assert not at.exception and not at.dataframe
    set_mode(at, MODE_1)
    assert not at.exception and not at.error
    assert result_is_shown(at), "result must still be shown, without a 'geändert' warning"
    assert not at.warning
    assert at.file_uploader(key="own_upload").value is None  # the uploader itself is empty again
    assert any(c.value == f"Zuletzt geladen: frog.csv – {KEPT_CAPTION}" for c in at.caption)
    assert any(c.value == f"Zuletzt geladen: quellen.csv – {KEPT_CAPTION}" for c in at.caption)
    assert len(at.get("download_button")) == 2
    assert button(at, ANALYZE_BUTTON).disabled is False  # still usable without a new upload


def test_kept_upload_feeds_a_new_chunking_after_the_round_trip():
    at = start()
    set_mode(at, MODE_3)
    set_mode(at, MODE_1)
    button(at, CHUNK_BUTTON).click()
    at.run()
    assert not at.exception and not at.error
    assert "Eigener Content: 2 Seiten → 3 Chunks" in at.code[0].value


def test_clear_button_removes_a_kept_upload():
    at = start()
    assert not [b for b in at.button if b.key == "own_parsed_clear"]  # nothing to clear while the file is chosen
    set_mode(at, MODE_2)
    set_mode(at, MODE_1)
    clear = next(b for b in at.button if b.key == "own_parsed_clear")
    assert clear.label == "Datei entfernen"
    clear.click()
    at.run()
    assert not at.exception
    assert not has_state(at, "own_parsed")
    assert not [b for b in at.button if b.key == "own_parsed_clear"]
    assert not any(KEPT_CAPTION in c.value for c in at.caption)
    assert button(at, CHUNK_BUTTON).disabled is True  # no own content any more


def test_choosing_another_file_replaces_the_kept_one():
    at = start()
    set_mode(at, MODE_2)
    set_mode(at, MODE_1)
    at.file_uploader(key="own_upload").set_value(("neu.csv", table_csv(OWN_DOCS[:1]), "text/csv"))
    at.run()
    assert at.session_state["own_parsed"]["sig"][0] == "neu.csv"
    assert not any(KEPT_CAPTION in c.value and "frog.csv" in c.value for c in at.caption)
    assert any("1 Seite" in c.value for c in at.caption)


def test_gap_results_with_uploaded_tables_survive_a_round_trip(backend):
    at = set_mode(AppTest.from_file(APP, default_timeout=30).run(), MODE_3)
    at.file_uploader(key="gap_competitor_upload").set_value(
        ("wettbewerb.csv", table_csv([headed(d) for d in SOURCE_DOCS]), "text/csv"))
    at.file_uploader(key="gap_own_upload").set_value(
        ("frog.csv", table_csv([headed(d) for d in OWN_DOCS]), "text/csv"))
    at.run()
    button(at, "Abschnitte vergleichen").click()
    at.run()
    assert at.dataframe and not at.warning
    set_mode(at, MODE_1)
    assert not at.dataframe
    set_mode(at, MODE_3)
    assert not at.exception and not at.error
    assert at.dataframe and at.get("download_button")
    assert not [w for w in at.warning if "geändert" in w.value]
    assert any(c.value == f"Zuletzt geladen: wettbewerb.csv – {KEPT_CAPTION}" for c in at.caption)
    assert any(c.value == f"Zuletzt geladen: frog.csv – {KEPT_CAPTION}" for c in at.caption)
    assert button(at, "Abschnitte vergleichen").disabled is False
    next(b for b in at.button if b.key == "gap_competitor_parsed_clear").click()
    at.run()
    assert not has_state(at, "gap_competitor_parsed")
    assert button(at, "Abschnitte vergleichen").disabled is True
    assert any("Eingaben haben sich" in w.value for w in at.warning)


def test_kept_chunk_export_is_rejected_when_the_model_changed_meanwhile():
    at = upload_chunk_export("text-embedding-3-small")
    assert not at.error
    set_mode(at, MODE_2)
    at.sidebar.text_input(key="model_openai").input("text-embedding-3-large").run()
    set_mode(at, MODE_1)
    assert any("erzeugt, eingestellt ist openai / text-embedding-3-large" in e.value for e in at.error)
    assert button(at, CHUNK_BUTTON).disabled is True


@pytest.mark.parametrize("slot", ["gap_competitor_upload", "gap_own_upload"])
def test_gap_mode_rejects_a_chunk_embedding_export_in_either_slot(backend, slot):
    data = chunks_to_csv(sample_result().own_chunks, provider="openai", model="text-embedding-3-small",
                         target_words=15)
    at = set_mode(AppTest.from_file(APP, default_timeout=30).run(), MODE_3)
    at.text_area(key="gap_competitor_urls").input("https://quelle.de/foerderung")
    at.text_area(key="gap_own_urls").input("https://me.de/kosten")
    at.file_uploader(key=slot).set_value(("chunks_embeddings.csv", data, "text/csv"))
    at.run()
    assert not at.exception
    assert [e.value for e in at.error] == [
        "Ein Chunk-Embedding-Export enthält keine Überschriften. Lade hier den Frog-Export mit dem "
        "Main Content hoch."]
    assert button(at, "Abschnitte vergleichen").disabled is True


def test_stage_one_source_slot_rejects_a_chunk_embedding_export():
    data = chunks_to_csv(sample_result().own_chunks, provider="openai", model="text-embedding-3-small",
                         target_words=15)
    at = start()
    at.file_uploader(key="source_upload").set_value(("chunks_embeddings.csv", data, "text/csv"))
    at.run()
    assert any("gilt nur für den eigenen Content" in e.value for e in at.error)
    assert button(at, CHUNK_BUTTON).disabled is True


# ---------------------------------------------------------------- sidebar values and session-state writes

def test_sidebar_values_survive_a_round_trip_through_another_mode():
    at = new_app()
    at.sidebar.checkbox(key="merge_short").set_value(False)
    at.sidebar.checkbox(key="split_long").set_value(False)
    at.sidebar.number_input(key="target_override").set_value(25)
    at.sidebar.slider(key="covered_delta").set_value(0.1)
    at.sidebar.slider(key="weak_delta").set_value(0.3)
    at.run()
    set_mode(at, MODE_2)
    assert not at.sidebar.slider and not at.sidebar.checkbox
    set_mode(at, MODE_3)
    set_mode(at, MODE_1)
    assert not at.exception and not at.warning
    assert at.sidebar.checkbox(key="merge_short").value is False
    assert at.sidebar.checkbox(key="split_long").value is False
    assert at.sidebar.number_input(key="target_override").value == 25
    assert at.sidebar.slider(key="covered_delta").value == pytest.approx(0.1)
    assert at.sidebar.slider(key="weak_delta").value == pytest.approx(0.3)
    assert at.text_input(key="keyword").value == KEYWORD


def test_mode_switches_do_not_log_the_session_state_write_warning(caplog, monkeypatch):
    import streamlit.elements.lib.policies as policies

    monkeypatch.setattr(policies, "_shown_default_value_warning", False)  # Streamlit warns once per process
    at = new_app()
    at.sidebar.checkbox(key="merge_short").set_value(False)
    at.run()
    with caplog.at_level("WARNING"):
        for mode in (MODE_2, MODE_3, MODE_1, MODE_3, MODE_2, MODE_1):
            set_mode(at, mode)
    assert not at.exception
    assert not [r for r in caplog.records if "Session State API" in r.getMessage()]
    assert at.sidebar.checkbox(key="merge_short").value is False


def test_source_content_can_be_pasted_as_text(backend):
    at = start()
    at.text_area(key="source_text").input(SOURCE_DOCS[0].text)
    at.run()
    button(at, CHUNK_BUTTON).click()
    at.run()
    assert not at.exception and not at.error
    assert "Quellen: 1 Seite" in at.code[0].value
    button(at, ANALYZE_BUTTON).click()
    at.run()
    assert not at.exception and not at.error
    result = at.session_state["result"]
    assert result.match.has_sources is True
    assert any(p.url == "Quelle (eingefügt)" for p in result.match.pages)
    assert backend.fetched == []


def test_competitor_content_can_be_pasted_in_gap_mode(backend):
    at = set_mode(AppTest.from_file(APP, default_timeout=30).run(), MODE_3)
    at.text_area(key="gap_competitor_text").input("## Förderung\n" + SOURCE_DOCS[0].text)
    at.text_area(key="gap_own_text").input("## Kosten\n" + OWN_DOCS[0].text)
    at.run()
    button(at, "Abschnitte vergleichen").click()
    at.run()
    assert not at.exception and not at.error
    frame = at.dataframe[0].value
    assert set(frame["Wettbewerber-URL"]) == {"Quelle (eingefügt)"}
    assert backend.fetched == []


def test_split_paragraphs_are_listed_with_a_preview():
    long_doc = "\n".join(["Satz{i} wort{i}a wort{i}b wort{i}c wort{i}d wort{i}e wort{i}f wort{i}g wort{i}h schluss{i}.".format(i=i) for i in range(1, 9)])
    at = new_app()
    at.text_area(key="own_text").input(" ".join(long_doc.split("\n")))  # one 80-word paragraph
    at.run()
    button(at, CHUNK_BUTTON).click()
    at.run()
    assert not at.exception and not at.error
    assert any("Eigener Text (eingefügt) · „Satz1 wort1a" in m.value and "Fenster" in m.value for m in at.markdown)


def test_mode_switch_shows_an_info_text_for_the_chosen_mode():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert any(m.value.startswith("**AI-Antwort gegen Content:**") for m in at.markdown)
    at = set_mode(at, MODE_3)
    assert any(m.value.startswith("**Content-Gap zwischen Seiten:**") for m in at.markdown)
    assert not any(m.value.startswith("**AI-Antwort gegen Content:**") for m in at.markdown)


def test_step_three_warns_about_pages_too_long_for_the_model(backend):
    at = new_app()
    at.text_area(key="own_text").input(" ".join(["Wort"] * 5100) + ".")
    at.run()
    button(at, CHUNK_BUTTON).click()
    at.run()
    assert not at.exception and not at.error
    assert any("über der Grenze von 5000 Wörtern" in w.value for w in at.warning)


def test_upload_urls_can_be_restricted_to_a_selection(backend):
    at = start()
    at.multiselect(key="own_url_filter").set_value(["https://me.de/kosten"]).run()
    button(at, CHUNK_BUTTON).click()
    at.run()
    assert not at.exception and not at.error
    assert "Eigener Content: 1 Seite" in at.code[0].value
    button(at, ANALYZE_BUTTON).click()
    at.run()
    assert [p.url for p in at.session_state["result"].match.pages] == ["https://me.de/kosten"]


def test_url_selection_is_offered_only_for_uploads_with_several_pages(backend):
    at = start()
    assert any(m.key == "own_url_filter" for m in at.multiselect)
    single = new_app()
    single.file_uploader(key="own_upload").set_value(("frog.csv", table_csv(OWN_DOCS[:1]), "text/csv"))
    single.run()
    assert not any(m.key == "own_url_filter" for m in single.multiselect)


def test_gap_mode_own_upload_can_be_restricted(backend):
    at = set_mode(AppTest.from_file(APP, default_timeout=30).run(), MODE_3)
    at.file_uploader(key="gap_competitor_upload").set_value(
        ("wettbewerb.csv", table_csv([headed(d) for d in SOURCE_DOCS]), "text/csv"))
    at.file_uploader(key="gap_own_upload").set_value(
        ("frog.csv", table_csv([headed(d) for d in OWN_DOCS]), "text/csv"))
    at.run()
    at.multiselect(key="gap_own_url_filter").set_value(["https://me.de/blog"]).run()
    button(at, "Abschnitte vergleichen").click()
    at.run()
    assert not at.exception and not at.error
    frame = at.dataframe[0].value
    assert set(frame["Beste eigene URL"]) == {"https://me.de/blog"}


def test_runs_show_a_status_box_that_stays_after_completion(backend):
    at = chunked(source_file=True)
    assert [(s.label.split(" (")[0], s.state) for s in at.status] == [("Chunking abgeschlossen", "complete")]
    button(at, ANALYZE_BUTTON).click()
    at.run()
    assert [(s.label.split(" (")[0], s.state) for s in at.status] == [("Analyse abgeschlossen", "complete")]


def test_failed_analysis_marks_the_status_box_as_error(monkeypatch):
    def no_key(*args, **kwargs):
        raise EmbeddingError("openai: API-Schlüssel fehlt")

    monkeypatch.setattr("aam.embeddings.make_provider", no_key)
    at = analyzed()
    assert any(st.state == "error" and st.label == "Abgebrochen" for st in at.status)


def test_sources_have_their_own_step():
    at = AppTest.from_file(APP, default_timeout=30).run()
    headers = [h.value for h in at.header]
    assert "3. Quellen und Wettbewerber (optional)" in headers
    assert headers.index("3. Quellen und Wettbewerber (optional)") < headers.index("4. Chunking prüfen")
    assert at.text_area(key="source_urls") is not None and at.text_area(key="source_text") is not None


def test_pasted_texts_can_be_named(backend):
    at = new_app()
    at.text_area(key="own_text").input(OWN_DOCS[0].text)
    at.text_area(key="source_text").input(SOURCE_DOCS[0].text)
    at.run()
    at.text_input(key="own_label").input("meine-seite.de/kosten")
    at.text_input(key="source_label").input("obi.de Kellersanierung")
    at.run()
    button(at, CHUNK_BUTTON).click()
    at.run()
    button(at, ANALYZE_BUTTON).click()
    at.run()
    assert not at.exception and not at.error
    urls = {p.url for p in at.session_state["result"].match.pages}
    assert urls == {"meine-seite.de/kosten", "obi.de Kellersanierung"}


def test_pasted_competitor_in_gap_mode_can_be_named(backend):
    at = set_mode(AppTest.from_file(APP, default_timeout=30).run(), MODE_3)
    at.text_area(key="gap_competitor_text").input("## Förderung\n" + SOURCE_DOCS[0].text)
    at.text_area(key="gap_own_text").input("## Kosten\n" + OWN_DOCS[0].text)
    at.run()
    at.text_input(key="gap_competitor_label").input("https://www.obi.de/kellersanierung")
    at.run()
    button(at, "Abschnitte vergleichen").click()
    at.run()
    assert not at.exception and not at.error
    assert set(at.dataframe[0].value["Wettbewerber-URL"]) == {"https://www.obi.de/kellersanierung"}


def test_readiness_caption_names_what_is_missing():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_input(key="keyword").input("kw")
    at.run()
    caption = next(c.value for c in at.caption if c.value.startswith("Es fehlt noch"))
    assert "AI-Antwort (Schritt 1)" in caption and "eigener Content (Schritt 2)" in caption
    assert "Keyword" not in caption.split(".")[0]


def test_gap_mode_shows_the_chunk_factor_slider():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.radio(key="mode").set_value("Content-Gap zwischen Seiten").run()
    assert not at.exception
    assert any(s.key == "content_factor" for s in at.sidebar.slider)
