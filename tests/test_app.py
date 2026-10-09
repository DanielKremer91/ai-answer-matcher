import io
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

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
    assert [h.value[0] for h in at.header if h.value[0].isdigit()] == ["1", "2", "3", "4"]
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
    assert "AI-Antwort: 3 Themen, 5 Aussagen" in protocol and "Zielgröße 15 Wörter" in protocol
    assert "Eigener Content: 2 Seiten → 3 Chunks" in protocol
    assert "Quellen: 1 Seite → 2 Chunks" in protocol
    frame = at.dataframe[0].value
    assert list(frame.columns) == ["Thema", "Aussage", "Wortzahl"] and len(frame) == 5
    assert any("bis zu 7 Texte" in c.value for c in at.caption)
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
    assert [p.url for p in pages] == ["Eingefügter Text"]
    assert backend.fetched == []
