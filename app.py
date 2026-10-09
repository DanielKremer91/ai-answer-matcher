"""AI Answer Matcher: Streamlit UI. Collects inputs, calls the aam package, renders results."""

from __future__ import annotations

import os
import re
from pathlib import Path

import httpx
import streamlit as st

from aam import embeddings, extract, pipeline, report
from aam.cache import MemoryCache
from aam.envfile import load_env_file
from aam.ingest import chunks_file, content
from aam.ingest.tables import describe_table_info, read_table_info
from aam.ingest.urls import parse_urls, url_key
from aam.models import OWN, SOURCE, Document

PASTED_URL = "Eingefügter Text"  # stands in for the URL of pasted own content

UPLOAD_TYPES = ["csv", "tsv", "txt", "xlsx", "xlsm"]
KEY_ENV = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}
RUN_ERRORS = (ValueError, embeddings.EmbeddingError)
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

st.set_page_config(page_title="AI Answer Matcher", layout="wide")
try:
    load_env_file(Path(__file__).resolve().parent / ".env")
except (OSError, UnicodeDecodeError):
    st.warning("Die Datei .env konnte nicht gelesen werden und wird ignoriert.")
state = st.session_state
state.setdefault("cache", MemoryCache())
state.setdefault("fetched", {})  # (origin, url key) -> Document; fetched pages survive re-chunking


# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("Einstellungen")
    provider_name = st.selectbox("Embedding-Anbieter", embeddings.PROVIDERS, key="provider")
    model = st.text_input("Modell", value=embeddings.DEFAULT_EMBED_MODELS[provider_name],
                          key=f"model_{provider_name}").strip()
    api_key = base_url = None
    if provider_name == "ollama":
        base_url = st.text_input("Ollama-URL", value="http://localhost:11434", key="ollama_url").strip()
    else:
        env_name = KEY_ENV[provider_name]
        typed = st.text_input(
            "API-Key", type="password", key=f"key_{provider_name}",
            help=f"Wird nicht gespeichert. Alternativ die Umgebungsvariable {env_name} setzen "
                 "oder in eine .env-Datei neben app.py schreiben.")
        api_key = typed.strip() or os.environ.get(env_name, "")
        if not api_key:
            st.caption(f"Noch kein Key vorhanden ({env_name}).")
    st.divider()
    target_override = st.number_input(
        "Chunk-Zielgröße in Wörtern (0 = aus der AI-Antwort ableiten)",
        min_value=0, max_value=300, value=0, step=1, key="target_override")
    merge_short = st.checkbox(
        "Kurze Absätze zusammenlegen", value=True, key="merge_short",
        help="Absätze unter der halben Zielgröße (z. B. Überschriften) werden mit dem nächsten Absatz "
             "zu einem Chunk verbunden. Aus: jede Zeile bleibt ein eigener Chunk.")
    split_long = st.checkbox(
        "Lange Absätze teilen", value=True, key="split_long",
        help="Absätze über der doppelten Zielgröße werden an Satzgrenzen in Fenster geteilt, jedes Fenster "
             "wird einzeln eingebettet, das beste zählt. Aus: lange Absätze bleiben ganz; ihr Vektor ist dann "
             "ein Durchschnitt über alle Gedanken darin und einzelne Aussagen gehen unter.")
    covered_delta = st.slider(
        "Schwelle „abgedeckt“", 0.0, 0.3, 0.05, 0.01, key="covered_delta",
        help="Der beste eigene Chunk darf höchstens so weit unter dem besten Quell-Chunk liegen.")
    weak_delta = st.slider(
        "Schwelle „schwach“", 0.0, 0.5, 0.15, 0.01, key="weak_delta",
        help="Bis zu diesem Abstand gilt eine Aussage als schwach abgedeckt, darunter als Lücke.")
    if weak_delta < covered_delta:
        st.warning("Die Schwelle „schwach“ liegt unter „abgedeckt“ und wird auf denselben Wert gesetzt.")
        weak_delta = covered_delta


# ---------------------------------------------------------------- helpers

def make_provider():
    return embeddings.make_provider(provider_name, model, api_key=api_key, base_url=base_url)


def fallback_embed():
    """Embedding function for the semantic segmentation fallback; None while no provider can be built."""
    try:
        provider = make_provider()
    except embeddings.EmbeddingError:
        return None

    def embed(texts):
        try:
            return embeddings.embed_cached(provider, texts, state["cache"])
        except embeddings.EmbeddingError:
            return [None] * len(texts)  # segmentation then keeps the structural result

    return embed


def parse_upload(file, origin: str, slot: str):
    """Parse an upload once per file and model; the parsed result lives in the session."""
    if file is None:
        state.pop(slot, None)
        return None
    signature = (file.name, file.size, getattr(file, "file_id", None), provider_name, model)
    cached = state.get(slot)
    if cached and cached["sig"] == signature:
        return cached
    parsed = {"sig": signature, "error": None, "docs": [], "chunks": None, "stored_target": None, "note": ""}
    try:
        df, info = read_table_info(file, file.name)
        if origin == OWN and chunks_file.is_chunks_table(info.columns):
            parsed["chunks"], parsed["stored_target"] = chunks_file.chunks_from_frame(
                df, provider=provider_name, model=model)
            parsed["note"] = (f"Chunk-Embedding-Export erkannt: {len(parsed['chunks'])} Chunks, "
                              f"Zielgröße {parsed['stored_target']} Wörter.")
        else:
            parsed["docs"], found = content.documents_from_frame(df, origin=origin)
            pages = f"{found.documents} {'Seite' if found.documents == 1 else 'Seiten'}"
            parsed["note"] = (f"{describe_table_info(info)} · URL-Spalte „{found.url_column}“ · "
                              f"Content-Spalte „{found.content_column}“ · {pages}")
            if found.empty:
                parsed["note"] += f", davon {found.empty} ohne Text"
    except ValueError as exc:
        parsed["error"] = str(exc)
    state[slot] = parsed
    return parsed


def show_upload_note(parsed) -> None:
    if parsed is None:
        return
    if parsed["error"]:
        st.error(parsed["error"])
    else:
        st.caption(parsed["note"])


def fetch_missing(urls, origin: str, known: set):
    """Documents for URLs no uploaded table covers. Pages fetched earlier in the session are reused."""
    wanted = [u for u in urls if url_key(u) not in known]
    todo = [u for u in wanted if (origin, url_key(u)) not in state["fetched"]]
    failed = []
    if todo:
        bar = st.progress(0.0, text=f"Lade {len(todo)} URLs …")
        try:
            with httpx.Client() as client:
                docs, failed = extract.fetch_documents(
                    todo, origin, client,
                    progress=lambda done, total: bar.progress(done / total, text=f"Lade URL {done} von {total} …"))
        finally:
            bar.empty()
        for doc in docs:
            state["fetched"][(origin, url_key(doc.url))] = doc
    found = [state["fetched"][(origin, url_key(u))] for u in wanted if (origin, url_key(u)) in state["fetched"]]
    return found, list(failed)


def failure_lines(failed, limit: int = 10) -> list[str]:
    """One line per URL that could not be loaded, so an error about missing content shows the reason."""
    lines = [f"• {f.url} ({f.reason})" for f in failed[:limit]]
    if len(failed) > limit:
        lines.append(f"… und {len(failed) - limit} weitere")
    return lines


def cached_bytes(slot: str, key, build):
    """Build a download once per result instead of on every rerun."""
    entry = state.get(slot)
    if not entry or entry[0] != key:
        entry = (key, build())
        state[slot] = entry
    return entry[1]


def show_result(result, shown_warnings=()) -> None:
    st.subheader("Ergebnis")
    st.markdown(report.summary(result))
    st.caption(f"Seitenvektoren: {result.meta.whole_pages} Seiten komplett eingebettet, "
               f"{result.meta.mean_pages} als Durchschnitt ihrer Chunks (zu lang für das Modell oder aus einem "
               "Chunk-Embedding-Export).")
    for warning in result.meta.warnings:
        if warning not in shown_warnings:  # warnings from the chunking step are already on screen
            st.warning(warning)
    tab_pages, tab_topics, tab_statements, tab_matrix = st.tabs(
        ["Seiten", "Themen", "Aussagen", "Aussage × Seite"])
    with tab_pages:
        st.caption("Ganze AI-Antwort gegen ganze Seite. Eigene Seiten und Quellen in einem Ranking.")
        st.dataframe(report.pages_frame(result), width="stretch", hide_index=True)
    with tab_topics:
        st.caption("Bester Chunk der eigenen Domain pro Themengebiet der AI-Antwort.")
        st.dataframe(report.topics_frame(result), width="stretch", hide_index=True)
    with tab_statements:
        st.caption("Bester Chunk der eigenen Domain pro einzelner Aussage (AI-Chunk).")
        st.dataframe(report.statements_frame(result), width="stretch", hide_index=True)
    with tab_matrix:
        st.caption(
            "Jede Aussage gegen jede Seite. „Bester Chunk Score“: Gibt es auf der Seite eine passende Passage? "
            "„Seiten-Relevanz“: Passt die Seite als Ganzes zum Thema der Aussage? Pro Aussage die zehn besten "
            "eigenen Seiten und alle Quellen.")
        matrix = report.statement_pages_matrix(result)
        if matrix is not None:
            st.markdown("**Matrix:** bester Chunk Score pro Aussage und Seite")
            st.dataframe(matrix, width="stretch", hide_index=True)
        else:
            st.caption("Für eine Matrix sind es zu viele Seiten. Die Liste unten und das Excel enthalten alle Werte.")
        st.dataframe(report.statement_pages_frame(result), width="stretch", hide_index=True)

    slug = re.sub(r"[^a-z0-9]+", "-", result.meta.keyword.lower()).strip("-")[:40] or "lauf"
    signature = state["result_sig"]
    excel = cached_bytes("excel", (signature, result.meta.covered_delta, result.meta.weak_delta),
                         lambda: report.to_excel(result))
    left, right = st.columns(2)
    left.download_button("Excel herunterladen", data=excel, mime=XLSX_MIME, key="download_excel",
                         file_name=f"ai-answer-matcher_{slug}_{result.meta.date}.xlsx")
    if not result.meta.prechunked:
        export = dict(chunks=result.own_chunks, provider=provider_name, model=model,
                      target_words=result.meta.target_words)  # bound now, the callable runs later
        right.download_button(
            "Chunk-Embeddings herunterladen", mime="text/csv", key="download_chunks",
            data=lambda: chunks_file.chunks_to_csv(  # built only when the user clicks
                export["chunks"], provider=export["provider"], model=export["model"],
                target_words=export["target_words"]),
            file_name=f"chunks_embeddings_{result.meta.date}.csv",
            help="Beim nächsten Keyword statt des Frog-Exports hochladen. "
                 "Dann wird nur noch die AI-Antwort eingebettet.")


# ---------------------------------------------------------------- page

st.title("AI Answer Matcher")
st.caption("Wie nah ist dein Content an der Antwort von AI Overviews oder AI Mode? "
           "Vergleich per Embeddings und Kosinus-Ähnlichkeit, auf Seiten- und auf Passagen-Ebene.")

st.header("1. AI-Antwort")
with st.expander("Was passiert in diesem Schritt?"):
    st.markdown(
        "Die AI-Antwort ist der Maßstab der Analyse. Das Tool zerlegt sie in **Themen** (Überschriften, fette "
        "Einleitungswörter) und **Aussagen** (einzelne Zeilen oder Bullets). Jede Aussage wird später einzeln "
        "mit deinem Content verglichen.\n\n"
        "Die **Quell-URLs** sind die Seiten, die die AI zitiert. Sie dienen als Benchmark: Pro Aussage siehst "
        "du, wie weit dein bester Absatz hinter dem besten Absatz der Quellen liegt. Ohne Quellen läuft die "
        "Analyse trotzdem, der Status zeigt dann nur die relative Stärke innerhalb deines eigenen Contents.\n\n"
        "Tipp: Zeilen entfernen, die nicht zur Antwort gehören, etwa Hinweise wie „KI-Antworten können Fehler "
        "enthalten“ oder Titel von Quellenkarten.")
keyword = st.text_input("Keyword oder Prompt", key="keyword", placeholder="z. B. dachsanierung kosten")
answer_text = st.text_area("AI-Antwort (Text aus AI Overview oder AI Mode einfügen)", key="answer", height=260)
source_urls = parse_urls(st.text_area(
    "Quell-URLs der AI-Antwort (optional, eine pro Zeile)", key="source_urls", height=110,
    help="Die zitierten Quellen dienen als Benchmark. Ohne Quellen läuft die Analyse trotzdem."))

st.header("2. Eigener Content")
with st.expander("Was passiert in diesem Schritt?"):
    st.markdown(
        "Dein Content wird in **Chunks** geschnitten, also in Absätze von etwa der Länge einer Aussage der "
        "AI-Antwort. Grenzen liegen an Absätzen und Sätzen, nie mitten im Satz. Deshalb braucht das Tool den "
        "Main Content mit Zeilenumbrüchen, am besten aus einem Screaming-Frog-Export.\n\n"
        "Vier Wege: ein **Frog-Export** mit URL- und Content-Spalte (ganze Domain oder Teilbereich), eine "
        "**URL-Liste** für eine Handvoll Seiten, die das Tool selbst lädt, ein **Chunk-Embedding-Export** "
        "aus einem früheren Lauf, damit die Domain nicht noch einmal eingebettet werden muss, oder für einen "
        "schnellen Test der **eingefügte Text** einer Seite.\n\n"
        "Optional kannst du den Content der Quell-URLs als Datei im selben Format hochladen, falls sich die "
        "Seiten nicht laden lassen oder du dieselbe Extraktion wie für deinen Content willst.")
own_file = st.file_uploader(
    "Screaming-Frog-Export mit URL und Main Content, oder Chunk-Embedding-Export aus einem früheren Lauf",
    type=UPLOAD_TYPES, key="own_upload")
own_parsed = parse_upload(own_file, OWN, "own_parsed")
show_upload_note(own_parsed)
own_urls = parse_urls(st.text_area(
    "Eigene URLs (optional, eine pro Zeile). Das Tool lädt den Content selbst.", key="own_urls", height=110))
own_text = st.text_area(
    "Eigenen Content als Text einfügen (optional, ein Absatz pro Zeile)", key="own_text", height=160,
    help="Für einen schnellen Test ohne Export: den Main Content einer Seite hier einfügen. "
         "Er erscheint in den Ergebnissen als „Eingefügter Text“.").strip()
pasted_docs = [Document(PASTED_URL, own_text, OWN)] if own_text else []
source_file = st.file_uploader(
    "Optional: Content der Quell-URLs als Datei (gleiches Format wie der Frog-Export)",
    type=UPLOAD_TYPES, key="source_upload")
source_parsed = parse_upload(source_file, SOURCE, "source_parsed")
show_upload_note(source_parsed)

prechunked = bool(own_parsed and own_parsed["chunks"] is not None)
if prechunked and (own_urls or pasted_docs):
    st.info("Mit einem Chunk-Embedding-Export werden eigene URLs und eingefügter Text nicht zusätzlich verwendet.")
upload_error = any(p is not None and p["error"] for p in (own_parsed, source_parsed))
has_own = prechunked or bool(own_parsed and own_parsed["docs"]) or bool(own_urls) or bool(pasted_docs)
ready = bool(keyword.strip() and answer_text.strip() and has_own and not upload_error)


def input_signature() -> tuple:
    """Everything the chunking depends on; a change makes the prepared state stale."""
    return (keyword.strip(), answer_text.strip(), tuple(source_urls), tuple(own_urls), own_text,
            own_parsed["sig"] if own_parsed else None, source_parsed["sig"] if source_parsed else None,
            int(target_override), merge_short, split_long, provider_name, model)


st.header("3. Chunking prüfen")
with st.expander("Was passiert in diesem Schritt?"):
    st.markdown(
        "Noch ohne Embedding-Kosten: Das Tool lädt die URLs, zerlegt die AI-Antwort und schneidet deinen Content "
        "und die Quellen in Chunks. Das **Protokoll** zeigt, wie viele Themen und Aussagen erkannt wurden, "
        "welche Zielgröße daraus folgt und wie viele Chunks entstanden sind.\n\n"
        "Prüfe hier, ob die Zerlegung der Antwort stimmt und ob die Beispiel-Chunks sinnvolle Absätze sind. "
        "Passt die Chunk-Größe nicht, kannst du sie in der Seitenleiste überschreiben und neu chunken.")
if not ready:
    st.caption("Es fehlen noch Keyword, AI-Antwort oder eigener Content.")
if st.button("Segmentieren und chunken", disabled=not ready, type="primary"):
    failed_own: list = []
    try:
        table_own = (own_parsed["docs"] if own_parsed else []) + pasted_docs
        fetched_own, failed_own = ([], []) if prechunked else fetch_missing(
            own_urls, OWN, {url_key(d.url) for d in table_own})
        table_sources = source_parsed["docs"] if source_parsed else []
        fetched_sources, failed_sources = fetch_missing(
            source_urls, SOURCE, {url_key(d.url) for d in table_sources})
        state["prepared"] = pipeline.prepare(
            keyword, answer_text,
            own_docs=pipeline.combine_documents(table_own, fetched_own),
            source_docs=pipeline.combine_documents(table_sources, fetched_sources),
            own_chunks=own_parsed["chunks"] if prechunked else None,
            stored_target=own_parsed["stored_target"] if prechunked else None,
            failed=failed_own + failed_sources,
            target_override=int(target_override) or None,
            embed=fallback_embed(),
            merge_short=merge_short, split_long=split_long,
        )
        state["prepared_sig"] = input_signature()
        state.pop("result", None)
        state.pop("excel", None)  # downloads belong to the previous result
    except RUN_ERRORS as exc:
        state.pop("prepared", None)
        st.error("\n\n".join([str(exc), *failure_lines(failed_own)]))

prepared = state.get("prepared")
fresh = prepared is not None and state.get("prepared_sig") == input_signature()
if prepared is not None and not fresh:
    st.warning("Die Eingaben haben sich seit dem letzten Chunking geändert. "
               "Bitte erneut segmentieren und chunken.")
if fresh:
    st.code("\n".join(report.protocol_lines(
        prepared.answer, prepared.own_stats, prepared.source_stats, prepared.failed,
        prechunked=prepared.prechunked, source_target_words=prepared.source_target_words)), language=None, wrap_lines=True)
    for warning in prepared.warnings:
        st.warning(warning)
    st.markdown("**AI-Chunks:** so wurde die Antwort in Themen und Aussagen zerlegt")
    st.dataframe(report.answer_frame(prepared.answer), width="stretch", hide_index=True)
    with st.expander("Beispiel-Chunks aus dem eigenen Content"):
        for chunk in prepared.own_chunks[:3]:
            st.markdown(f"**{chunk.url}** · Chunk {chunk.index} · {chunk.words} Wörter")
            st.write(chunk.text)
    if prepared.source_chunks:
        with st.expander("Beispiel-Chunks aus den Quellen"):
            for chunk in prepared.source_chunks[:3]:
                st.markdown(f"**{chunk.url}** · Chunk {chunk.index} · {chunk.words} Wörter")
                st.write(chunk.text)
    split_examples = list(prepared.own_stats.split_examples) + (
        list(prepared.source_stats.split_examples) if prepared.source_stats else [])
    if split_examples:
        with st.expander(f"Geteilte Absätze ({prepared.own_stats.split + (prepared.source_stats.split if prepared.source_stats else 0)})"):
            st.caption("Diese Absätze waren länger als die doppelte Zielgröße und wurden an Satzgrenzen in "
                       "Fenster geteilt. Jedes Fenster wird einzeln eingebettet, das beste zählt. "
                       "Abschaltbar in der Seitenleiste.")
            for url, words, windows in split_examples:
                st.markdown(f"- {url} · {words} Wörter → {windows} Fenster")
    st.caption(f"Im nächsten Schritt werden bis zu {prepared.texts_to_embed} Texte eingebettet.")

st.header("4. Analysieren")
with st.expander("Was passiert in diesem Schritt?"):
    st.markdown(
        "Alle Texte laufen durch dasselbe Embedding-Modell und werden per Kosinus-Ähnlichkeit verglichen, "
        "auf mehreren Ebenen:\n\n"
        "- **Seiten:** ganze AI-Antwort gegen ganze Seite. Welche Seite passt insgesamt am besten?\n"
        "- **Themen** und **Aussagen:** pro Themengebiet und pro Aussage der beste Absatz deiner Domain, "
        "verglichen mit dem besten Absatz der Quellen. Daraus folgt der Status „abgedeckt“, „schwach“ oder "
        "„Lücke“.\n"
        "- **Aussage × Seite:** jede Aussage gegen jede Seite, mit bestem Absatz und Relevanz der ganzen Seite.\n\n"
        "Der Nutzen liegt in den Lücken: Aussagen, zu denen dir eine eigenständige Passage fehlt. Ähnlichkeit "
        "ist nötig, um zitiert zu werden, aber nicht hinreichend. Die Antwort nachzuschreiben ist kein Ziel.")
if st.button("Analyse starten", disabled=not fresh, type="primary"):
    bar = None
    try:
        provider = make_provider()
        bar = st.progress(0.0, text="Embeddings werden erstellt …")
        state["result"] = pipeline.analyze(
            prepared, provider, state["cache"], covered_delta=covered_delta, weak_delta=weak_delta,
            progress=lambda done, total: bar.progress(done / max(total, 1),
                                                      text=f"Embeddings: {done} von {total} Texten"))
        state["result_sig"] = state["prepared_sig"]
        state.pop("excel", None)  # a new result must not be served with the old download bytes
    except RUN_ERRORS as exc:
        state.pop("result", None)
        st.error(f"Analyse fehlgeschlagen: {exc}")
    finally:
        if bar is not None:
            bar.empty()

result = state.get("result")
if result is not None and fresh and state.get("result_sig") == state.get("prepared_sig"):
    if (result.meta.covered_delta, result.meta.weak_delta) != (covered_delta, weak_delta):
        result = state["result"] = pipeline.rematch(result, covered_delta=covered_delta, weak_delta=weak_delta)
    show_result(result, shown_warnings=prepared.warnings)
