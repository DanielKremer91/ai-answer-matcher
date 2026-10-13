"""AI Answer Matcher: Streamlit UI. Collects inputs, calls the aam package, renders results."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import httpx
import streamlit as st
from typing import Optional

from aam import compare, embeddings, extract, gap, gap_report, gap_statements, pipeline, report, segment
from aam.cache import MemoryCache
from aam.envfile import load_env_file
from aam.ingest import chunks_file, content
from aam.ingest.tables import describe_table_info, read_table_info
from aam.ingest.urls import parse_urls, url_key
from aam.models import OWN, SOURCE, Document

PASTED_URL = "Eigener Text (eingefügt)"  # default name of pasted own content when no label is given
PASTED_SOURCE_URL = "Quelle (eingefügt)"  # same for pasted source / competitor content


def pasted_label(key: str, default: str, what: str) -> str:
    """A short name or URL for pasted text; it appears wherever the tool shows a URL."""
    value = st.text_input(f"Bezeichnung oder URL für den eingefügten {what}", value=kept(key, ""), key=key,
                          placeholder="z. B. obi.de Kellersanierung oder https://…").strip()
    return value or default

UPLOAD_TYPES = ["csv", "tsv", "txt", "xlsx", "xlsm"]
KEY_ENV = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}
RUN_ERRORS = (ValueError, embeddings.EmbeddingError)
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
GAP_CHUNKS_ERROR = ("Ein Chunk-Embedding-Export enthält keine Überschriften. Lade hier den Frog-Export mit dem "
                    "Main Content hoch.")
SOURCE_CHUNKS_ERROR = ("Ein Chunk-Embedding-Export gilt nur für den eigenen Content. Lade hier den Frog-Export "
                       "mit dem Main Content der Quellen hoch.")

st.set_page_config(page_title="AI Answer Matcher", layout="wide")
try:
    load_env_file(Path(__file__).resolve().parent / ".env")
except (OSError, UnicodeDecodeError):
    st.warning("Die Datei .env konnte nicht gelesen werden und wird ignoriert.")
state = st.session_state
state.setdefault("cache", MemoryCache())
state.setdefault("fetched", {})  # (origin, url key) -> Document; fetched pages survive re-chunking

MODE_STAGE1, MODE_MANUAL, MODE_GAP = "AI-Antwort gegen Content", "Absatz gegen Text", "Content-Gap zwischen Seiten"
# A widget loses its value while another mode runs, because it is not rendered then. Writing a value into the
# key of a rendered widget logs a Streamlit warning, so the value of every widget is copied under a separate key
# at the end of each run (`remember_inputs`) and read back as the widget's `value=` when it is rendered again.
MODE_KEYS = {
    MODE_STAGE1: ("keyword", "answer", "source_urls", "own_urls", "own_text",
                  "target_override", "merge_short", "split_long", "covered_delta", "weak_delta",
                  "content_factor", "source_text", "own_url_filter", "own_label", "source_label"),
    MODE_MANUAL: ("manual_short", "manual_long"),
    MODE_GAP: ("gap_competitor_urls", "gap_competitor_text", "gap_own_urls", "gap_own_text", "gap_keyword",
               "gap_answer", "gap_own_url_filter", "gap_competitor_label", "gap_own_label", "content_factor",
               "gap_statement_target", "covered_delta", "weak_delta"),
}
COPY_PREFIX = "kept_"


def kept(key: str, default):
    """Value of the widget `key` from its last render in any mode, `default` before the first one."""
    return state.get(COPY_PREFIX + key, default)


def remember_inputs() -> None:
    """Copy the input values aside; see the comment at MODE_KEYS. A widget that was not rendered in this run
    still has its value here when it changed in the very run that switched the mode."""
    for keys in MODE_KEYS.values():
        for key in keys:
            if key in state:
                state[COPY_PREFIX + key] = state[key]


st.title("AI Answer Matcher")
st.caption("Wie nah ist dein Content an der Antwort von AI Overviews oder AI Mode? "
           "Vergleich per Embeddings und Kosinus-Ähnlichkeit, auf Seiten- und auf Passagen-Ebene.")
MODE_INFO = {
    MODE_STAGE1: "**AI-Antwort gegen Content:** Du fügst eine AI-Antwort ein, dazu deinen Content (Frog-Export, "
                 "URLs oder Text) und optional die zitierten Quellen. Ergebnis: Welche Seite passt insgesamt "
                 "(Seitenscore, unabhängig vom Chunking), welcher Absatz deckt welche Aussage ab, wo sind die "
                 "Lücken gegenüber den Quellen.",
    MODE_MANUAL: "**Absatz gegen Text:** Zwei Felder, ein Wert. Eine Aussage oder die ganze AI-Antwort links, "
                 "ein Absatz oder dein ganzer Text rechts. Ist der Text viel länger als die Aussage, zeigt das "
                 "Tool zusätzlich das beste Fenster. Keine Dateien, kein Chunking-Schritt.",
    MODE_GAP: "**Content-Gap zwischen Seiten:** Eine Wettbewerber-Seite gegen deine Seite oder Domain, Abschnitt "
              "gegen Abschnitt nach Überschriften. Die sortierte Liste zeigt, welche Abschnitte des Wettbewerbers "
              "bei dir keinen Gegenpart haben. Mit AI-Antwort auch, welche davon für die Antwort zählen.",
}
mode = st.radio("Modus", [MODE_STAGE1, MODE_MANUAL, MODE_GAP], key="mode", horizontal=True)
st.markdown(MODE_INFO[mode])


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
    if mode in (MODE_STAGE1, MODE_GAP):
        st.divider()
        content_factor = st.slider(
            "Chunk-Faktor (Content zu Aussage)", 1.0, 3.0, kept("content_factor", 1.5), 0.1, key="content_factor",
            help="Website-Text braucht für denselben Sachverhalt mehr Wörter als eine verdichtete AI-Aussage. "
                 "Die Chunk-Zielgröße ist der Median der Aussagenlänge mal diesem Faktor. "
                 "1,0 = gleiche Größe, 1,5 = anderthalbfach (Standard)."
                 + (" Im Gap-Modus bestimmt der Faktor nur die Fenstergröße für die Spalte „Kosinus AI-Aussage“ "
                    "und wirkt nur, wenn eine AI-Antwort eingefügt ist." if mode == MODE_GAP else ""))
    else:
        content_factor = 1.5
    if mode == MODE_STAGE1:
        target_override = st.number_input(
            "Chunk-Zielgröße in Wörtern (0 = aus der AI-Antwort ableiten)",
            min_value=0, max_value=300, value=kept("target_override", 0), step=1, key="target_override")
        merge_short = st.checkbox(
            "Kurze Absätze zusammenlegen", value=kept("merge_short", True), key="merge_short",
            help="Absätze unter der halben Zielgröße (z. B. Überschriften) werden mit dem nächsten Absatz "
                 "zu einem Chunk verbunden. Aus: jede Zeile bleibt ein eigener Chunk.")
        split_long = st.checkbox(
            "Lange Absätze teilen", value=kept("split_long", True), key="split_long",
            help="Absätze über der doppelten Zielgröße werden an Satzgrenzen in Fenster geteilt, jedes Fenster "
                 "wird einzeln eingebettet, das beste zählt. Aus: lange Absätze bleiben ganz; ihr Vektor ist dann "
                 "ein Durchschnitt über alle Gedanken darin und einzelne Aussagen gehen unter.")
        st.caption("Status einer Aussage, gemessen am Abstand zwischen deinem besten Chunk und dem besten "
                   "Quell-Chunk (Kosinus eigen minus Kosinus Quelle):")
        covered_delta = st.slider(
            "Schwelle „abgedeckt“", 0.0, 0.3, kept("covered_delta", 0.05), 0.01, key="covered_delta",
            help="Abgedeckt: Dein bester Chunk liegt höchstens so weit unter dem besten Quell-Chunk. "
                 "Beispiel bei 0,05: Quelle 0,85, du 0,81 → Abstand 0,04 → abgedeckt.")
        weak_delta = st.slider(
            "Schwelle „schwach“", 0.0, 0.5, kept("weak_delta", 0.15), 0.01, key="weak_delta",
            help="Schwach: Der Abstand liegt zwischen der Schwelle „abgedeckt“ und dieser Schwelle. "
                 "Lücke: Der Abstand ist größer als diese Schwelle. "
                 "Beispiel bei 0,05 und 0,15: Quelle 0,85, du 0,75 → Abstand 0,10 → schwach; "
                 "du 0,60 → Abstand 0,25 → Lücke.")
        if weak_delta < covered_delta:
            st.warning("Die Schwelle „schwach“ liegt unter „abgedeckt“ und wird auf denselben Wert gesetzt.")
            weak_delta = covered_delta
    elif mode == MODE_GAP:
        target_override, merge_short, split_long = 0, True, True
        gap_statement_target = st.number_input(
            "Chunk-Zielgröße Aussagen-Ebene (Wörter)", min_value=10, max_value=120,
            value=kept("gap_statement_target", gap_statements.DEFAULT_STATEMENT_TARGET), step=5, key="gap_statement_target",
            help="Beide Seiten werden innerhalb ihrer Abschnitte in Chunks dieser Größe geschnitten, ein bis zwei "
                 "Sätze. Jeder Wettbewerber-Chunk gilt als eine Aussage und bekommt deinen besten Chunk als Gegenpart.")
        st.caption("Status einer Wettbewerber-Aussage, gemessen am Abstand zwischen deinem besten Chunk und der "
                   "Referenz aus dem Wettbewerber-Text selbst (nächste Nachbar-Aussage, Spezifität):")
        covered_delta = st.slider(
            "Schwelle „abgedeckt“", 0.0, 0.3, kept("covered_delta", 0.05), 0.01, key="covered_delta",
            help="Abgedeckt: Dein bester Chunk liegt höchstens so weit unter der Referenz. "
                 "Beispiel bei 0,05: Referenz 0,75, du 0,72 → Abstand 0,03 → abgedeckt.")
        weak_delta = st.slider(
            "Schwelle „schwach“", 0.0, 0.5, kept("weak_delta", 0.15), 0.01, key="weak_delta",
            help="Schwach: Der Abstand liegt zwischen den Schwellen, du bist beim Thema, nicht bei der Aussage. "
                 "Lücke: Der Abstand ist größer als diese Schwelle.")
        if weak_delta < covered_delta:
            st.warning("Die Schwelle „schwach“ liegt unter „abgedeckt“ und wird auf denselben Wert gesetzt.")
            weak_delta = covered_delta
    else:  # not shown in the other modes; stage-1 values stay unused there
        target_override, merge_short, split_long, covered_delta, weak_delta = 0, True, True, 0.05, 0.15


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


def parse_upload(file, origin: str, slot: str, *, chunks_error: str = ""):
    """Parse an upload once per file and model; the parsed result lives in the session.

    Without a file (the uploader forgets its file when another mode runs) the parsed upload of an earlier run
    stays in use until `forget_upload` removes it. A chunk-embedding export is accepted for own content only
    when `chunks_error` is empty; otherwise that text is the error.
    """
    cached = state.get(slot)
    if file is None:
        if cached and cached["chunks"] is not None and cached["sig"][3:5] != (provider_name, model):
            old_provider, old_model = cached["sig"][3:5]
            return {**cached, "error": f"Der Export wurde mit {old_provider} / {old_model} erzeugt, eingestellt ist "
                                       f"{provider_name} / {model}. Bitte Anbieter und Modell in der Seitenleiste "
                                       "anpassen oder die Datei erneut wählen."}
        return cached
    signature = (file.name, file.size, getattr(file, "file_id", None), provider_name, model)
    if cached and cached["sig"] == signature:
        return cached
    parsed = {"sig": signature, "error": None, "docs": [], "chunks": None, "stored_target": None, "note": ""}
    try:
        df, info = read_table_info(file, file.name)
        if chunks_file.is_chunks_table(info.columns):
            if chunks_error:
                raise ValueError(chunks_error)
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


def url_filter(parsed, key: str) -> Optional[set]:
    """Multiselect over the URLs of a parsed upload; None (no selection) means every URL."""
    if not parsed or parsed["error"]:
        return None
    urls = ([c.url for c in parsed["chunks"]] if parsed["chunks"] is not None else [d.url for d in parsed["docs"]])
    options = list(dict.fromkeys(urls))
    if len(options) < 2:
        return None
    chosen = st.multiselect("Nur diese URLs aus dem Upload verwenden (leer = alle)", options,
                            default=[u for u in kept(key, []) if u in options], key=key,
                            help="Den Export der ganzen Domain hochladen und hier die Seiten wählen, die in "
                                 "diese Analyse eingehen sollen.")
    return set(chosen) or None


def filtered(parsed, selected: Optional[set]):
    """The upload's documents and chunks, restricted to the selected URLs."""
    if not parsed:
        return [], None, None
    docs = parsed["docs"]
    chunks = parsed["chunks"]
    if selected is not None:
        docs = [d for d in docs if d.url in selected]
        chunks = [c for c in chunks if c.url in selected] if chunks is not None else None
    return docs, chunks, parsed["stored_target"]



def show_upload_note(parsed, file, slot: str) -> None:
    if parsed is None:
        return
    if file is None:  # the parsed upload of an earlier run is still in use
        st.caption(f"Zuletzt geladen: {parsed['sig'][0]} – bleibt erhalten, bis du eine andere Datei wählst.")
        st.button("Datei entfernen", key=f"{slot}_clear", type="tertiary", on_click=lambda: state.pop(slot, None))
    if parsed["error"]:
        st.error(parsed["error"])
    else:
        st.caption(parsed["note"])


class RunFeedback:
    """A status box with spinner, step lines and a progress bar; it stays visible after the run."""

    def __init__(self, label: str):
        self.started = time.time()
        self.status = st.status(label, expanded=True)
        self.bar = self.status.progress(0.0, text="Vorbereitung …")

    def step(self, text: str) -> None:
        self.status.write(text)

    def progress(self, done: int, total: int, what: str) -> None:
        self.bar.progress(done / max(total, 1), text=f"{what}: {done} von {total}")

    def done(self, label: str) -> None:
        self.bar.empty()
        self.status.update(label=f"{label} ({time.time() - self.started:.0f} s)", state="complete", expanded=False)

    def failed(self) -> None:
        self.bar.empty()
        self.status.update(label="Abgebrochen", state="error", expanded=True)


def fetch_missing(urls, origin: str, known: set, feedback: Optional[RunFeedback] = None):
    """Documents for URLs no uploaded table covers. Pages fetched earlier in the session are reused."""
    wanted = [u for u in urls if url_key(u) not in known]
    todo = [u for u in wanted if (origin, url_key(u)) not in state["fetched"]]
    failed = []
    if todo:
        label = "Quell-URLs" if origin == SOURCE else "eigene URLs"
        if feedback:
            feedback.step(f"Lade {len(todo)} {label} …")
        with httpx.Client() as client:
            docs, failed = extract.fetch_documents(
                todo, origin, client,
                progress=lambda done, total: feedback.progress(done, total, "Geladene URLs") if feedback else None)
        for doc in docs:
            state["fetched"][(origin, url_key(doc.url))] = doc
        if feedback:
            feedback.step(f"{len(docs)} von {len(todo)} {label} geladen"
                          + (f", {len(failed)} nicht ladbar" if failed else ""))
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
        st.caption("Pro Thema der AI-Antwort die beste Passage deiner Seiten und der Quellen. Die Passagen sind so groß "
                   "wie das Thema (Spalte „Vergleich“), nicht so klein wie ein einzelner Chunk.")
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



def render_stage1() -> None:
    """Mode 1: the AI answer against own content (steps 1 to 4)."""
    st.header("1. AI-Antwort")
    with st.expander("Was passiert in diesem Schritt?"):
        st.markdown(
            "Die AI-Antwort ist der Maßstab der Analyse. Das Tool zerlegt sie in **Themen** (Überschriften, fette "
            "Einleitungswörter) und **Aussagen** (einzelne Zeilen oder Bullets). Jede Aussage wird später einzeln "
            "mit deinem Content verglichen.\n\n"
            "Tipp: Zeilen entfernen, die nicht zur Antwort gehören, etwa Hinweise wie „KI-Antworten können Fehler "
            "enthalten“ oder Titel von Quellenkarten.")
    keyword = st.text_input("Keyword oder Prompt", value=kept("keyword", ""), key="keyword", placeholder="z. B. dachsanierung kosten")
    answer_text = st.text_area("AI-Antwort (Text aus AI Overview oder AI Mode einfügen)", value=kept("answer", ""),
                             key="answer", height=260)

    st.header("2. Eigener Content")
    with st.expander("Was passiert in diesem Schritt?"):
        st.markdown(
            "Dein Content wird in **Chunks** geschnitten, also in Absätze von etwa der Länge einer Aussage der "
            "AI-Antwort. Grenzen liegen an Absätzen und Sätzen, nie mitten im Satz. Deshalb braucht das Tool den "
            "Main Content mit Zeilenumbrüchen, am besten aus einem Screaming-Frog-Export.\n\n"
            "Vier Wege: ein **Frog-Export** mit URL- und Content-Spalte (ganze Domain oder Teilbereich), eine "
            "**URL-Liste** für eine Handvoll Seiten, die das Tool selbst lädt, ein **Chunk-Embedding-Export** "
            "aus einem früheren Lauf, damit die Domain nicht noch einmal eingebettet werden muss, oder für einen "
            "schnellen Test der **eingefügte Text** einer Seite.")
    own_file = st.file_uploader(
        "Screaming-Frog-Export mit URL und Main Content, oder Chunk-Embedding-Export aus einem früheren Lauf",
        type=UPLOAD_TYPES, key="own_upload")
    own_parsed = parse_upload(own_file, OWN, "own_parsed")
    show_upload_note(own_parsed, own_file, "own_parsed")
    own_selected = url_filter(own_parsed, "own_url_filter")
    own_docs_sel, own_chunks_sel, own_stored_target = filtered(own_parsed, own_selected)
    own_urls = parse_urls(st.text_area(
        "Eigene URLs (optional, eine pro Zeile). Das Tool lädt den Content selbst.",
        value=kept("own_urls", ""), key="own_urls", height=110))
    own_text = st.text_area(
        "Eigenen Content als Text einfügen (optional, ein Absatz pro Zeile)", value=kept("own_text", ""),
        key="own_text", height=160,
        help="Für einen schnellen Test ohne Export: den Main Content einer Seite hier einfügen.").strip()
    own_label = pasted_label("own_label", PASTED_URL, "eigenen Text") if own_text else PASTED_URL
    pasted_docs = [Document(own_label, own_text, OWN)] if own_text else []

    st.header("3. Quellen und Wettbewerber (optional)")
    with st.expander("Was passiert in diesem Schritt?"):
        st.markdown(
            "Die **Quellen** sind die Seiten, die die AI zitiert, oder andere Wettbewerber, gegen die du dich messen "
            "willst. Sie sind der Benchmark: Pro Aussage siehst du, wie weit dein bester Absatz hinter dem besten "
            "Absatz der Quellen liegt, und daraus entsteht der Status „abgedeckt“, „schwach“ oder „Lücke“.\n\n"
            "Drei Wege: **URLs** eintragen, die das Tool selbst lädt, einen **Frog-Export** der Quellseiten "
            "hochladen (dieselbe Extraktion wie für deinen Content, gut für Seiten, die sich nicht laden lassen), "
            "oder den **Text** einer Quelle einfügen.\n\n"
            "Ohne Quellen läuft die Analyse trotzdem; der Status zeigt dann nur die relative Stärke innerhalb "
            "deines eigenen Contents. Quellen, die zu deinem eigenen Content gehören, werden nicht als Benchmark "
            "gewertet.")
    source_urls = parse_urls(st.text_area(
        "Quell-URLs (eine pro Zeile). Das Tool lädt den Content selbst.", value=kept("source_urls", ""),
        key="source_urls", height=110))
    source_file = st.file_uploader(
        "Frog-Export der Quellseiten (URL und Main Content)", type=UPLOAD_TYPES, key="source_upload")
    source_parsed = parse_upload(source_file, SOURCE, "source_parsed", chunks_error=SOURCE_CHUNKS_ERROR)
    show_upload_note(source_parsed, source_file, "source_parsed")
    source_text = st.text_area(
        "Quell-Content als Text einfügen (ein Absatz pro Zeile)", value=kept("source_text", ""),
        key="source_text", height=120,
        help="Der Main Content einer zitierten Quelle, zum Beispiel aus der Zwischenablage.").strip()
    source_label = pasted_label("source_label", PASTED_SOURCE_URL, "Quell-Text") if source_text else PASTED_SOURCE_URL
    pasted_sources = [Document(source_label, source_text, SOURCE)] if source_text else []

    prechunked = bool(own_parsed and own_parsed["chunks"] is not None)
    if prechunked and (own_urls or pasted_docs):
        st.info("Mit einem Chunk-Embedding-Export werden eigene URLs und eingefügter Text nicht zusätzlich verwendet.")
    upload_error = any(p is not None and p["error"] for p in (own_parsed, source_parsed))
    has_own = (prechunked and bool(own_chunks_sel)) or bool(own_docs_sel) or bool(own_urls) or bool(pasted_docs)
    ready = bool(keyword.strip() and answer_text.strip() and has_own and not upload_error)


    def input_signature() -> tuple:
        """Everything the chunking depends on; a change makes the prepared state stale."""
        return (keyword.strip(), answer_text.strip(), tuple(source_urls), tuple(own_urls), own_text, own_label,
                source_label,
                own_parsed["sig"] if own_parsed else None, tuple(sorted(own_selected)) if own_selected else None,
                source_parsed["sig"] if source_parsed else None,
                int(target_override), merge_short, split_long, round(content_factor, 2), source_text,
                provider_name, model)


    st.header("4. Chunking prüfen")
    with st.expander("Was passiert in diesem Schritt?"):
        st.markdown(
            "Noch ohne Embedding-Kosten: Das Tool lädt die URLs, zerlegt die AI-Antwort und schneidet deinen Content "
            "und die Quellen in Chunks. Das **Protokoll** zeigt, wie viele Themen und Aussagen erkannt wurden, "
            "welche Zielgröße daraus folgt und wie viele Chunks entstanden sind.\n\n"
            "Prüfe hier, ob die Zerlegung der Antwort stimmt und ob die Beispiel-Chunks sinnvolle Absätze sind. "
            "Passt die Chunk-Größe nicht, kannst du sie in der Seitenleiste überschreiben und neu chunken.")
    if not ready:
        missing = [name for name, ok in (("Keyword oder Prompt (Schritt 1)", bool(keyword.strip())),
                                         ("AI-Antwort (Schritt 1)", bool(answer_text.strip())),
                                         ("eigener Content (Schritt 2)", has_own)) if not ok]
        if upload_error:
            missing.append("ein Upload mit Fehler (siehe Meldung oben)")
        st.caption("Es fehlt noch: " + ", ".join(missing) + ". Hinweis: Streamlit übernimmt Text aus einem "
                   "Feld erst, wenn du es verlässt (außerhalb klicken) oder Strg/Cmd+Enter drückst.")
    if st.button("Segmentieren und chunken", disabled=not ready, type="primary"):
        failed_own: list = []
        feedback = RunFeedback("Segmentieren und chunken läuft …")
        try:
            table_own = own_docs_sel + pasted_docs
            fetched_own, failed_own = ([], []) if prechunked else fetch_missing(
                own_urls, OWN, {url_key(d.url) for d in table_own}, feedback)
            table_sources = (source_parsed["docs"] if source_parsed else []) + pasted_sources
            fetched_sources, failed_sources = fetch_missing(
                source_urls, SOURCE, {url_key(d.url) for d in table_sources}, feedback)
            feedback.step("Zerlege die AI-Antwort und schneide den Content in Chunks …")
            state["prepared"] = pipeline.prepare(
                keyword, answer_text,
                own_docs=pipeline.combine_documents(table_own, fetched_own),
                source_docs=pipeline.combine_documents(table_sources, fetched_sources),
                own_chunks=own_chunks_sel if prechunked else None,
                stored_target=own_stored_target if prechunked else None,
                failed=failed_own + failed_sources,
                target_override=int(target_override) or None, content_factor=content_factor,
                embed=fallback_embed(),
                merge_short=merge_short, split_long=split_long,
            )
            state["prepared_sig"] = input_signature()
            state.pop("result", None)
            state.pop("excel", None)  # downloads belong to the previous result
            feedback.done("Chunking abgeschlossen")
        except RUN_ERRORS as exc:
            state.pop("prepared", None)
            feedback.failed()
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
                for url, words, windows, preview in split_examples:
                    st.markdown(f"- {url} · „{preview}“ · {words} Wörter → {windows} Fenster")
        for line in report.overlong_pages(prepared.page_texts, provider_name):
            st.warning(line)
        st.caption(f"Im nächsten Schritt werden bis zu {prepared.texts_to_embed} Texte eingebettet.")

    st.header("5. Analysieren")
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
        feedback = RunFeedback("Analyse läuft …")
        try:
            provider = make_provider()
            feedback.step(f"Bette bis zu {prepared.texts_to_embed} Texte mit {provider_name} / {model} ein …")
            state["result"] = pipeline.analyze(
                prepared, provider, state["cache"], covered_delta=covered_delta, weak_delta=weak_delta,
                progress=lambda done, total: feedback.progress(done, total, "Embeddings"))
            feedback.step("Berechne Kosinus-Ähnlichkeiten und Status …")
            state["result_sig"] = state["prepared_sig"]
            state.pop("excel", None)  # a new result must not be served with the old download bytes
            feedback.done("Analyse abgeschlossen")
        except RUN_ERRORS as exc:
            state.pop("result", None)
            feedback.failed()
            st.error(f"Analyse fehlgeschlagen: {exc}")

    result = state.get("result")
    if result is not None and fresh and state.get("result_sig") == state.get("prepared_sig"):
        if (result.meta.covered_delta, result.meta.weak_delta) != (covered_delta, weak_delta):
            result = state["result"] = pipeline.rematch(result, covered_delta=covered_delta, weak_delta=weak_delta)
        show_result(result, shown_warnings=prepared.warnings)


# ---------------------------------------------------------------- mode 2: paragraph against text

def show_comparison(comparison) -> None:
    if comparison.windowed:
        left, right = st.columns(2)
        left.metric("Ganztext", report._de(comparison.whole_score))
        right.metric("Bestes Fenster", report._de(comparison.score))
        st.caption(f"Bestes Fenster ({comparison.window_words} Wörter)")
        st.text(comparison.best_window)
        ratio = f"{comparison.ratio:.1f}".replace(".", ",")
        subject, other = ("Die Aussage", "dein Text") if comparison.swapped else ("Dein Text", "die Aussage")
        st.info(f"{subject} ist {ratio}-mal so lang wie {other}. Der Ganztext-Wert wird durch "
                "die Länge gedrückt; das beste Fenster zeigt, ob die Aussage an einer Stelle vorkommt.")
    else:
        st.metric("Kosinus", report._de(comparison.score))
        st.markdown("Beide Texte sind ähnlich lang, der Wert gilt direkt.")


def render_manual() -> None:
    """Mode 2: one statement or paragraph against a text."""
    st.header("Absatz gegen Text")
    with st.expander("Was passiert in diesem Modus?"):
        st.markdown(
            "Du vergleichst eine einzelne **Aussage oder einen Absatz** der AI-Antwort mit **deinem Text**, "
            "ohne Export und ohne Chunking.\n\n"
            "Sind beide Texte ähnlich lang (der längere höchstens viermal so lang), gilt ein Kosinus-Wert "
            "zwischen den ganzen Texten. Ist dein Text deutlich länger, wird er in Fenster von der Länge der "
            "Aussage geschnitten. Du siehst dann zwei Werte: den **Ganztext-Wert**, den die Länge nach unten "
            "zieht, und den Wert des **besten Fensters**, der zeigt, ob die Aussage an einer Stelle vorkommt.")
    short_text = st.text_area("Aussage oder Absatz der AI-Antwort", value=kept("manual_short", ""), key="manual_short", height=140)
    long_text = st.text_area("Dein Text", value=kept("manual_long", ""), key="manual_long", height=260)
    ready = bool(short_text.strip() and long_text.strip())
    if not ready:
        st.caption("Es fehlen noch die Aussage oder dein Text.")
    signature = (short_text.strip(), long_text.strip(), provider_name, model)
    if st.button("Vergleichen", key="manual_button", disabled=not ready, type="primary"):
        try:
            provider = make_provider()
            comparison = compare.compare_texts(
                short_text, long_text, lambda texts: embeddings.embed_cached(provider, texts, state["cache"]),
                provider_name)
            state["manual_result"] = (signature, comparison)
        except RUN_ERRORS as exc:
            state.pop("manual_result", None)
            st.error(str(exc))
            return
    stored = state.get("manual_result")
    if stored is not None and stored[0] == signature:  # a result of other texts or another model is hidden
        show_comparison(stored[1])


# ---------------------------------------------------------------- mode 3: content gap

GAP_TEXT_COLUMNS = ["Text Wettbewerber", "Text eigen"]
GAP_TEXT_ROWS = 30  # rows shown under "Texte nachlesen"; the Excel has all of them


def render_gap_result(result) -> None:
    st.subheader("Ergebnis")
    pages = result.pages
    first_note = pages[0].note if pages else ""
    summary = gap_report.gap_summary(result)
    if summary and summary != first_note:
        st.markdown(summary)
    for gap_page in pages:
        if gap_page.note:
            st.info(gap_page.note if len(pages) == 1 else f"{gap_page.url}: {gap_page.note}")
    st.code("\n".join(gap_report.gap_protocol_lines(result)), language=None, wrap_lines=True)
    if result.answer is not None:
        order = st.radio("Sortierung", ["Kosinus eigen", "Priorität"], horizontal=True, key="gap_order",
                         help="Priorität = Kosinus zur nächsten AI-Aussage minus dein Kosinus: oben stehen "
                              "Abschnitte, die für die AI-Antwort zählen und dir fehlen.")
    else:
        order = "Kosinus eigen"
    if order == "Priorität":
        st.caption("Sortiert nach Priorität, absteigend: nah an einer Aussage der AI-Antwort und weit weg von "
                   "deinem Content. Das sind die Lücken, die sich zuerst lohnen.")
        frame = gap_report.priority_frame(result)
    else:
        st.caption("Sortiert nach Kosinus, absteigend. Die Zeilen am Ende sind die Kandidaten für Lücken: "
                   "Abschnitte des Wettbewerbers, zu denen dein Content kaum einen Gegenpart hat.")
        frame = gap_report.sections_frame(result)
    st.markdown("**Themen-Ebene: Abschnitte**")
    st.dataframe(frame.drop(columns=GAP_TEXT_COLUMNS), width="stretch", hide_index=True)
    if result.statements:
        st.markdown("**Aussagen-Ebene: jede Aussage des Wettbewerbers mit deinem besten Chunk**")
        st.caption("Lücken zuerst, dann schwach, dann abgedeckt. Status aus dem Delta: dein Kosinus minus die höhere "
                   "der beiden Referenzen aus dem Wettbewerber-Text (Nachbar-Aussagen desselben Abschnitts, "
                   "Spezifität deines Chunks). Null oder mehr heißt: Du sagst dasselbe.")
        st.dataframe(gap_report.statements_frame(result).drop(columns=["Bester eigener Chunk"]),
                     width="stretch", hide_index=True)
    rows = [(gap_page, row) for gap_page in pages for row in gap_page.rows]
    with st.expander("Texte nachlesen"):
        if len(rows) > GAP_TEXT_ROWS:
            st.caption(f"Gezeigt werden die ersten {GAP_TEXT_ROWS} von {len(rows)} Zeilen. "
                       "Alle Texte stehen im Excel.")
        for gap_page, row in rows[:GAP_TEXT_ROWS]:
            st.markdown(f"**{gap_page.url}** · Rang {row.rank} · Kosinus {report._de(row.score)}")
            st.markdown(f"Wettbewerber: **{row.competitor.title}**")
            st.text(row.competitor.text)
            st.markdown(f"Eigener Abschnitt ({row.own.url}): **{row.own.title}**")
            st.text(row.own.text)
            if row.best_window:
                st.markdown(f"Bestes Fenster ({row.method})")
                st.text(row.best_window)
    slug = re.sub(r"[^a-z0-9]+", "-", result.keyword.lower()).strip("-")[:40] or "lauf"
    excel = cached_bytes("gap_excel", state["gap_sig"], lambda: gap_report.gap_to_excel(result))
    st.download_button("Excel herunterladen", data=excel, mime=XLSX_MIME, key="gap_download",
                       file_name=f"ai-answer-matcher_content-gap_{slug}_{result.date}.xlsx")


def render_gap() -> None:
    """Mode 3: sections of competitor pages against the sections of own pages."""
    st.header("Content-Gap zwischen Seiten")
    with st.expander("Was passiert in diesem Modus?"):
        st.markdown(
            "Das Tool schneidet jede Seite an ihren Überschriften in **Abschnitte** und vergleicht jeden "
            "Abschnitt der Wettbewerber-Seite mit allen Abschnitten deines Contents. Pro Wettbewerber-Abschnitt "
            "bleibt dein bester Abschnitt, mit dem Kosinus-Wert der beiden.\n\n"
            "Die Liste ist **absteigend nach Kosinus sortiert**. Die Zeilen am Ende sind die **Kandidaten für "
            "Lücken**: Themen, die der Wettbewerber behandelt und zu denen du keinen ähnlichen Abschnitt hast. "
            "Ist ein Abschnitt viel länger als der andere, zählt der bessere von Ganztext- und Fensterwert; "
            "die Spalte „Vergleich“ nennt, woher der Wert stammt. Das ist die **Themen-Ebene**: Hast du zu "
            "diesem Thema überhaupt einen Abschnitt?\n\n"
            "Darunter die **Aussagen-Ebene**: Beide Seiten werden innerhalb ihrer Abschnitte in Chunks von ein bis "
            "zwei Sätzen geschnitten. Jeder Wettbewerber-Chunk gilt als eine Aussage und bekommt deinen besten "
            "Chunk. Ob das dieselbe Aussage ist, entscheidet der Vergleich mit zwei Referenzen aus dem "
            "Wettbewerber-Text selbst: den Nachbar-Aussagen desselben Abschnitts (gleiches Thema, andere Aussage) "
            "und der Spezifität deines Chunks (passt er zu allem ein bisschen?). So braucht es keine feste "
            "Schwelle, die vom Modell abhängt. Pro Abschnitt entsteht ein **Urteil**: Themenlücke, Detaillücken, "
            "nur beim Thema oder abgedeckt. Die Zusammenfassung oben nennt zuerst die Themenlücken, dann die "
            "Detaillücken mit den fehlenden Aussagen.\n\n"
            "Wettbewerber und eigener Content kommen aus URLs, die das Tool lädt, oder aus einem "
            "Screaming-Frog-Export mit Überschriften im Main Content (siehe README). Optional kannst du eine "
            "AI-Antwort einfügen: Dann steht in jeder Zeile zusätzlich die nächste Aussage der Antwort.")

    st.subheader("Wettbewerber")
    comp_file = st.file_uploader("Screaming-Frog-Export der Wettbewerber-Seiten (URL und Main Content)",
                                 type=UPLOAD_TYPES, key="gap_competitor_upload")
    comp_parsed = parse_upload(comp_file, SOURCE, "gap_competitor_parsed", chunks_error=GAP_CHUNKS_ERROR)
    show_upload_note(comp_parsed, comp_file, "gap_competitor_parsed")
    comp_urls = parse_urls(st.text_area("Wettbewerber-URLs (eine pro Zeile). Das Tool lädt den Content selbst.",
                                        value=kept("gap_competitor_urls", ""), key="gap_competitor_urls",
                                        height=110))

    comp_text = st.text_area("Wettbewerber-Content als Text einfügen (optional, ein Absatz pro Zeile, "
                             "Überschriften als „## Überschrift“)", value=kept("gap_competitor_text", ""),
                             key="gap_competitor_text", height=160).strip()
    comp_label = (pasted_label("gap_competitor_label", PASTED_SOURCE_URL, "Wettbewerber-Text") if comp_text
                  else PASTED_SOURCE_URL)

    st.subheader("Eigener Content")
    own_file = st.file_uploader("Screaming-Frog-Export deiner Seiten (URL und Main Content)",
                                type=UPLOAD_TYPES, key="gap_own_upload")
    own_parsed = parse_upload(own_file, OWN, "gap_own_parsed", chunks_error=GAP_CHUNKS_ERROR)
    show_upload_note(own_parsed, own_file, "gap_own_parsed")
    gap_own_selected = url_filter(own_parsed, "gap_own_url_filter")
    gap_own_docs, _, _ = filtered(own_parsed, gap_own_selected)
    own_urls = parse_urls(st.text_area("Eigene URLs (eine pro Zeile)", value=kept("gap_own_urls", ""),
                                  key="gap_own_urls", height=110))
    own_text = st.text_area("Eigenen Content als Text einfügen (optional, ein Absatz pro Zeile, Überschriften "
                            "als „## Überschrift“)", value=kept("gap_own_text", ""), key="gap_own_text",
                            height=160).strip()
    gap_own_label = pasted_label("gap_own_label", PASTED_URL, "eigenen Text") if own_text else PASTED_URL

    st.subheader("AI-Antwort (optional)")
    gap_keyword = st.text_input("Keyword oder Prompt", value=kept("gap_keyword", ""), key="gap_keyword")
    gap_answer = st.text_area("AI-Antwort", value=kept("gap_answer", ""), key="gap_answer", height=160,
                              help="Mit Antwort bekommt jede Zeile die nächste Aussage der Antwort und deren Kosinus.")

    upload_error = any(p is not None and p["error"] for p in (comp_parsed, own_parsed))
    has_competitors = bool(comp_urls or comp_text or (comp_parsed and comp_parsed["docs"]))
    has_own = bool(own_urls or own_text or gap_own_docs)
    ready = has_competitors and has_own and not upload_error
    if not ready:
        st.caption("Es fehlen noch Wettbewerber-Seiten oder eigener Content.")

    signature = (tuple(comp_urls), comp_text, comp_label, tuple(own_urls), own_text, gap_own_label,
                 gap_keyword.strip(), gap_answer.strip(), round(content_factor, 2), int(gap_statement_target),
                 round(covered_delta, 3), round(weak_delta, 3),
                 comp_parsed["sig"] if comp_parsed else None, own_parsed["sig"] if own_parsed else None,
                 tuple(sorted(gap_own_selected)) if gap_own_selected else None,
                 provider_name, model)

    if st.button("Abschnitte vergleichen", key="gap_button", disabled=not ready, type="primary"):
        failed: list = []
        feedback = RunFeedback("Abschnittsvergleich läuft …")
        try:
            table_comp = (comp_parsed["docs"] if comp_parsed else []) + (
                [Document(comp_label, comp_text, SOURCE)] if comp_text else [])
            fetched_comp, failed_comp = fetch_missing(comp_urls, SOURCE, {url_key(d.url) for d in table_comp},
                                                     feedback)
            table_own = gap_own_docs + (
                [Document(gap_own_label, own_text, OWN)] if own_text else [])
            fetched_own, failed_own = fetch_missing(own_urls, OWN, {url_key(d.url) for d in table_own}, feedback)
            failed = failed_comp + failed_own
            answer = segment.segment_answer(gap_keyword, gap_answer) if gap_answer.strip() else None
            provider = make_provider()
            feedback.step(f"Bilde Abschnitte und bette sie mit {provider_name} / {model} ein …")
            state["gap_result"] = gap.analyze_gap(
                pipeline.combine_documents(table_comp, fetched_comp),
                pipeline.combine_documents(table_own, fetched_own),
                lambda texts: embeddings.embed_cached(
                    provider, texts, state["cache"],
                    progress=lambda done, total: feedback.progress(done, total, "Embeddings")),
                provider_name, model, answer=answer, keyword=gap_keyword.strip(), content_factor=content_factor,
                statement_target=int(gap_statement_target), covered_delta=covered_delta, weak_delta=weak_delta)
            state["gap_sig"] = signature
            state["gap_failed"] = failure_lines(failed)
            state.pop("gap_excel", None)
            feedback.done("Vergleich abgeschlossen")
        except RUN_ERRORS as exc:
            state.pop("gap_result", None)
            feedback.failed()
            st.error("\n\n".join([str(exc), *failure_lines(failed)]))

    result = state.get("gap_result")
    if result is None:
        return
    if state.get("gap_sig") != signature:
        st.warning("Die Eingaben haben sich seit dem letzten Vergleich geändert. "
                   "Bitte erneut vergleichen.")
        return
    if state.get("gap_failed"):
        st.warning("\n\n".join(["Nicht geladen:", *state["gap_failed"]]))
    render_gap_result(result)


# ---------------------------------------------------------------- page

{MODE_STAGE1: render_stage1, MODE_MANUAL: render_manual, MODE_GAP: render_gap}[mode]()
remember_inputs()
