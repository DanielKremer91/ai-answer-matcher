# Gap Mode and Manual Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add layout-based sections, a two-text comparison with windowing, a content-gap mode between pages, a manual comparison mode and a mode switch in the Streamlit app, plus the "Nähe zum Quell-Chunk" column in stage 1.

**Architecture:** New modules `aam/sections.py` (headings → sections), `aam/compare.py` (two texts → Comparison with windowing), `aam/gap.py` (pages → section pairs → GapResult), `aam/gap_report.py` (frames, Excel). `app.py` gets a mode radio; stage-1 code stays as it is. No module under `aam/` imports streamlit.

**Tech Stack:** as stage 1 (Python 3.11, numpy, pandas, httpx, trafilatura, Streamlit 1.65, pytest).

**Spec:** `docs/superpowers/specs/2026-10-09-gap-and-manual-modes-design.md` (binding) on top of `docs/superpowers/specs/2026-10-03-ai-answer-matcher-design.md`.

## Global Constraints

- Repo `/Users/Daniel/Downloads/ai-answer-matcher`, branch `feat/stage1`, never push. Run tests with `.venv/bin/pytest -q` (295 tests green at the start; they must stay green unless a change is mandated below).
- No streamlit under `aam/`; all HTTP via httpx; tests never touch the network; the fake embedder in `tests/fakes.py` (`FakeProvider`, `bow_vector`) is used for every embedding in tests.
- User-facing strings German (typographic quotes „ “ are fine), code and comments English.
- Reuse what exists: `aam.chunk._windows(paragraph, target_words)` for windows, `aam.match.unit_rows`/`weighted_mean` for vectors, `aam.embeddings.embed_cached` + `fits_whole_text`, `aam.ingest.urls.url_key`, `aam.extract.fetch_documents`, `aam.ingest.content.documents_from_frame`, `aam.segment.segment_answer`, `aam.report._frame`/`_excel_frame`/`_count`/`_de`.
- Commit after every task with explicit `git add` paths; end every commit message with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## File Structure

| File | Responsibility |
|---|---|
| `aam/sections.py` | `Section` dataclass, `split_sections(doc) -> list[Section]`, `section_level(text)`, `strip_heading_marker(line)`, `heading_level(line)` |
| `aam/extract.py` (modify) | trafilatura markdown output, bold markers removed |
| `aam/chunk.py` (modify) | strip `#` markers from paragraph lines before chunking |
| `aam/compare.py` | `Comparison` dataclass, `compare_texts(short, long, embed, provider_name) -> Comparison`, `text_vector(text, embed, provider_name)`, type alias `Embed` |
| `aam/gap.py` | `GapRow`, `GapPage`, `GapResult`, `analyze_gap(...)` |
| `aam/gap_report.py` | `sections_frame`, `gap_protocol_lines`, `gap_summary`, `gap_to_excel` |
| `aam/match.py` / `aam/models.py` / `aam/report.py` (modify) | `StatementMatch.own_vs_source` and the column „Nähe zum Quell-Chunk“ |
| `app.py` (modify) | mode radio; manual mode; gap mode |
| tests | `tests/test_sections.py`, `tests/test_compare.py`, `tests/test_gap.py`, `tests/test_gap_report.py`, additions to `tests/test_match.py`, `tests/test_report.py`, `tests/test_extract.py`, `tests/test_chunk.py`, `tests/test_app.py` |

The embedding callable used by `compare` and `gap` has the signature `embed(texts: list[str]) -> list[Optional[np.ndarray]]` (what `lambda t: embed_cached(provider, t, cache)` gives) plus the provider name for `fits_whole_text`. Define the type alias `Embed` in `aam/compare.py` and reuse it in `aam/gap.py`.

---

### Task 1: Sections from headings; markdown extraction; markers stripped in chunking

**Files:** create `aam/sections.py`, `tests/test_sections.py`; modify `aam/extract.py`, `aam/chunk.py`, `tests/test_extract.py`, `tests/test_chunk.py`.

**Produces:**
- `aam.sections.HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")`, `strip_heading_marker(line) -> str` (returns the title for a heading line, the line unchanged otherwise), `heading_level(line) -> Optional[int]`.
- `Section(url: str, origin: str, index: int, title: str, level: int, text: str, words: int)` (frozen dataclass; `text` is the section body, one paragraph per line, headings of lower levels kept as plain lines without markers; `level` is the heading level of the section, 0 for „Einleitung“/whole page).
- `section_level(text) -> Optional[int]`: smallest level number > 1 that occurs at least twice; None otherwise.
- `split_sections(doc: Document) -> list[Section]` per spec section 3: text before the first section heading → title = H1 text if an H1 line exists else „Einleitung“; sections without body text are dropped; when `section_level` is None the whole page (markers stripped) is one section titled with the H1 text or „Gesamte Seite“; `index` counts from 0 in page order.
- `aam.extract.extract_text(html)` now calls trafilatura with `output_format="markdown"` and strips bold/italic markers (`**`, `__`) from the lines; headings arrive as `## Title` lines; empty lines removed.
- `aam.chunk._chunk_document` applies `strip_heading_marker` to every line before anything else (so stage-1 chunks never contain `##`).

**Tests to write (RED first):**
- sections: H2 page with intro → [H1-titled intro, H2 sections]; H3 inside H2 stays as text line; page with only H3s (two of them) → level 3; page with one H2 only → whole page, title from H1; page without headings → „Gesamte Seite“; heading without body dropped; `words` correct; markers never in `text`; a `#`-line with trailing spaces.
- extract: the existing HTML fixture in tests/test_extract.py now yields a line starting with `# ` or `## ` for the headings (trafilatura markdown) and no `**`; existing assertions keep passing (adjust only if trafilatura's markdown output legitimately changes line content — report it).
- chunk: a document `"## Kosten\nDie Sanierung kostet 100 Euro pro Quadratmeter und hängt vom Material ab."` at target 20 gives one chunk `"Kosten Die Sanierung …"` (marker gone, heading merged).

Commit: "Add layout sections, markdown extraction and marker-free chunking".

---

### Task 2: Two-text comparison with windowing

**Files:** create `aam/compare.py`, `tests/test_compare.py`.

**Produces:**
- `Embed = Callable[[list[str]], list[Optional[np.ndarray]]]`.
- `Comparison(score: float, whole_score: float, windowed: bool, best_window: str, window_words: int, short_words: int, long_words: int)`.
- `LENGTH_RATIO = 4`; target bounds reuse `aam.segment.MIN_TARGET/MAX_TARGET`.
- `text_vector(text, embed, provider_name) -> Optional[np.ndarray]`: whole embedding when `fits_whole_text(provider_name, text)`, else the word-weighted mean of the vectors of `aam.chunk._windows(text, 80)`; normalised; None when no vector could be produced.
- `compare_texts(short: str, long: str, embed, provider_name) -> Comparison`: the function swaps the arguments when `short` has more words than `long`. If `len(long.split()) <= LENGTH_RATIO * len(short.split())` → `score = whole_score = cos(text_vector(short), text_vector(long))`, `windowed=False`, `best_window=""`, `window_words=0`. Else target = clamp(len(short words), 15, 80); windows = `_windows(long, target) + _windows(long, 2*target)` (deduplicated, order kept); all windows embedded in one `embed` call (plus the short text and, when it fits, the long text); `score` = max cosine over windows, `best_window` = that window, `window_words` its word count; `whole_score` = cos(short, text_vector(long)); `windowed=True`.
- Raises `ValueError` (German) when a text is empty.

**Tests (fake embedder, assertions on behaviour):** ratio exactly 4 → not windowed; ratio above 4 → windowed; the best window contains the sentence that repeats the short text (build the long text from unrelated sentences plus the short text); `score >= whole_score` in the windowed case for that input; both window sizes are tried (a case where the matching content spans two sentences and only the 2x window reaches the higher score — construct and assert `window_words > target`); swapped arguments give the same result; empty text raises; `text_vector` falls back to the mean for a text above the provider limit (monkeypatch `aam.compare.fits_whole_text`) and the embed callable is then called with windows only.

Commit: "Add two-text comparison with best-window scoring".

---

### Task 3: Gap analysis

**Files:** create `aam/gap.py`, `tests/test_gap.py`.

**Produces:**
- `GapRow(competitor: Section, own: Section, score: float, windowed: bool, best_window: str, statement: Optional[Statement], statement_score: Optional[float], rank: int)`.
- `GapPage(url: str, level: Optional[int], sections: int, words: int, page_score: float, best_own_url: str, rows: list[GapRow], note: str)` — `note` is „Die Seite hat nur einen Abschnitt, verglichen wird die ganze Seite.“ when `level is None`, else "".
- `GapResult(pages: list[GapPage], own_pages: list[tuple[str, int, int]] (url, level or 0, sections), answer: Optional[SegmentedAnswer], provider: str, model: str, date: str, keyword: str)`.
- `analyze_gap(competitors: Sequence[Document], own: Sequence[Document], embed, provider_name, model, *, answer: Optional[SegmentedAnswer] = None, keyword: str = "", today=None) -> GapResult`:
  1. `split_sections` for every page; raise `ValueError` (German) when a side has no section with text.
  2. Section vectors with `compare.text_vector` (all texts of a run go through as few `embed` calls as reasonable; the callable is cache-backed).
  3. For every competitor section × own section: score via the rule of `compare_texts` — whole vectors when the ratio is at most 4, otherwise windows of the longer section (both sizes) against the shorter; implement an internal helper that reuses the precomputed section vectors and only embeds windows when needed (calling `compare_texts` for every pair is acceptable if it goes through the cache-backed callable; say in the report which you did).
  4. Keep the best own section per competitor section; `rank` by descending score; `rows` sorted descending.
  5. With `answer`: statement vectors via `embed`; for every competitor section the statement with the highest cosine to the section vector (`statement`, `statement_score`).
  6. `page_score` = cosine between the competitor page vector (`text_vector` of the whole page text) and the best own page vector; `best_own_url` that page.

**Tests:** one competitor with 3 sections vs one own page with 2 matching sections → rows sorted descending, the unmatched competitor section ranks last with the lowest score; two own pages → the row names the URL of the better one; windowed pair (competitor section 20 words vs own section 200 words) → `windowed=True` and `best_window` non-empty; with an answer → `statement` set and the "Förderung" section maps to the KfW statement (use `tests.fakes.ANSWER` / `segment_answer`); without headings on the competitor page → one row, `level is None`, note set; no own text → `ValueError`.

Commit: "Add content-gap analysis between pages".

---

### Task 4: Gap report and Excel

**Files:** create `aam/gap_report.py`, `tests/test_gap_report.py`.

**Produces:**
- `sections_frame(result: GapResult) -> DataFrame` with columns `Keyword/Prompt, Wettbewerber-URL, Rang, Abschnitt (Wettbewerber), Wörter, Beste eigene URL, Eigener Abschnitt, Wörter eigen, Kosinus, Bestes Fenster, Nächste AI-Aussage, Kosinus AI-Aussage, Text Wettbewerber, Text eigen` (the two AI columns only when `result.answer` is not None; scores rounded to 4 decimals; `Keyword/Prompt` is the keyword or an empty string).
- `gap_protocol_lines(result) -> list[str]`: per competitor page `„{url}: Abschnittsebene H{level}, {n} Abschnitte, {w} Wörter, Ganztext-Wert {score} gegen {best_own_url}“` (for `level is None`: „Abschnittsebene keine“) plus the note, then per own page `„Eigene Seite {url}: Abschnittsebene H{level}, {n} Abschnitte“`; German singular/plural via `aam.report._count`; scores with comma decimal (`aam.report._de`).
- `gap_summary(result) -> str`: for the first competitor page: „{url}: {k} von {n} Abschnitten haben bei dir einen Gegenpart mit Kosinus über {median}; die {m} Abschnitte am Ende der Liste sind die Kandidaten für Lücken.“ — `median` = median score over the rows (comma decimal, 2 digits); `k` = rows with score ≥ median; `m` = `n - k`. Keep the wording exactly.
- `gap_to_excel(result) -> bytes` with sheets „Abschnitte“ and „Protokoll“ (Angabe/Wert rows: Datum, Anbieter, Modell, then one row per protocol line), cells sanitised with `aam.report._excel_frame`.

**Tests:** column lists with and without answer; sorting kept; Excel has the two sheets and `Keyword/Prompt` first; protocol lines wording for 1 and 2 sections; summary wording.

Commit: "Add gap report tables and Excel".

---

### Task 5: Stage-1 column „Nähe zum Quell-Chunk“

**Files:** modify `aam/models.py` (`StatementMatch.own_vs_source: Optional[float] = None`), `aam/match.py` (cosine between the best own chunk vector and the best source chunk vector per statement, only with sources), `aam/report.py` (`statements_frame`: column „Nähe zum Quell-Chunk“ after „Delta“, only with sources), tests in `tests/test_match.py` and `tests/test_report.py`; update the column-list assertion in the existing statements-frame test. The README table row for „Aussagen“ mentions the new column.

Commit: "Add closeness of own and source chunk per statement".

---

### Task 6: App modes

**Files:** modify `app.py`, `tests/test_app.py`, `README.md`.

**Behaviour:**
- A radio `st.radio("Modus", [...], key="mode", horizontal=True)` directly under the title with the options „AI-Antwort gegen Content“, „Absatz gegen Text“, „Content-Gap zwischen Seiten“. Default the first. The existing stage-1 UI renders only in mode 1 (wrap it in a function or an `if`); sidebar provider/model/key always; target size, chunking switches and thresholds only in mode 1.
- Mode „Absatz gegen Text“: explanation expander; `st.text_area(key="manual_short")`, `st.text_area(key="manual_long")`, button „Vergleichen“ (key `manual_button`); result via `compare.compare_texts` with `embed = lambda t: embeddings.embed_cached(provider, t, state["cache"])`; display per spec section 5 (two `st.metric` values when windowed, the window text, the German hint; one value and the sentence „Beide Texte sind ähnlich lang, der Wert gilt direkt.“ otherwise). Errors as `st.error`.
- Mode „Content-Gap zwischen Seiten“: explanation expander; competitor URLs text area (key `gap_competitor_urls`) or upload (key `gap_competitor_upload`); own URLs text area (key `gap_own_urls`), own upload (key `gap_own_upload`), own pasted text (key `gap_own_text`); optional keyword (key `gap_keyword`) and AI answer (key `gap_answer`); button „Abschnitte vergleichen“ (key `gap_button`). Loading via `extract.fetch_documents` through the module attribute (tests patch it), uploads via `documents_from_frame`. Result: `gap_summary`, protocol as `st.code(..., wrap_lines=True)`, one `st.dataframe` of `sections_frame` without the two text columns, an expander „Texte nachlesen“ with the two texts per row (cap 30 rows), download button „Excel herunterladen“ (key `gap_download`). Errors as `st.error`. The gap mode keeps its own session keys (`gap_result`).
- README: a section „Drei Modi“ describing the three modes in the tone of the existing text; mention the Frog snippet `examples/screaming-frog-main-content.js` for headings.

**Tests (AppTest, fake backend as in the existing fixture):** radio present with three options; switching to mode 2 hides the stage-1 headers („1. AI-Antwort“) and shows the two fields; manual comparison with a short vs long text shows two metrics and the hint; manual with similar lengths shows one value; gap mode with competitor and own pages (fake fetcher returns texts with headings — add a local helper that prepends `"## Kosten\n"`-style lines to `OWN_DOCS`/`SOURCE_DOCS` texts) produces a dataframe with „Kosinus“ and a download button; gap mode with an answer adds „Nächste AI-Aussage“; switching back to mode 1 still runs the stage-1 flow (reuse `analyzed()` after a switch).

Commit: "App: mode switch, manual comparison and content-gap mode".

---

### Task 7: End-to-end check

Run `.venv/bin/pytest -q` (all green), `grep -rn streamlit aam/` (clean), boot the app on a spare port (200), and run the gap mode once through the package with the example files (`examples/beispiel-content.csv` as own, `examples/beispiel-quellen.csv` as competitors, FakeProvider) printing the sections frame. Fix what it uncovers in the owning module with a test; commit.
