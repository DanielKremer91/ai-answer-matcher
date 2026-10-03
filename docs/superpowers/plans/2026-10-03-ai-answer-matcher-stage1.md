# AI Answer Matcher Stage 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit tool that matches a pasted AI Overview / AI Mode answer against own content (and optionally the cited sources) with embeddings and cosine similarity, on page level and chunk level, with Excel export.

**Architecture:** A Python package `aam` with small single-purpose modules (ingest, extract, segment, chunk, embeddings, match, pipeline, report) and a thin `app.py` that only collects inputs, calls the package and renders results. No module in `aam` imports Streamlit. The embedding provider layer, the table reader, the env-file loader and the file cache are copied from the sibling repo `/Users/Daniel/Downloads/broken-link-matcher` (package `blm`).

**Tech Stack:** Python 3.11, Streamlit, httpx, pandas, numpy, trafilatura, openpyxl, pytest, respx, uv.

**Spec:** `docs/superpowers/specs/2026-10-03-ai-answer-matcher-design.md` (read it first). This plan covers stage 1 only. SerpApi fetch and the validation study (spec section 12) are out of scope.

## Global Constraints

- Repo root: `/Users/Daniel/Downloads/ai-answer-matcher`. Local git, branch `main`. **Never push.**
- Python 3.11 via `uv`; run everything with `.venv/bin/python` and `.venv/bin/pytest`.
- No module under `aam/` imports `streamlit`.
- All HTTP goes through `httpx` so tests can mock it. Tests never touch the network.
- No LLM calls except embeddings. Segmentation and chunking are deterministic.
- All user-facing strings (UI labels, error messages, Excel headers) are German. Code, comments and docstrings are English.
- Symmetric comparison: every text is embedded without query/passage prefix by the same model.
- Status thresholds default to 0.05 (abgedeckt) and 0.15 (schwach); target size is clamped to 15..80 words.
- `Keyword/Prompt` is the first column of every Excel sheet.
- The tool must run to completion when no source URL is given or when every source URL fails.
- Commit after every task. End commit messages with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## File Structure

| File | Responsibility |
|---|---|
| `aam/models.py` | Dataclasses shared by all modules |
| `aam/text.py` | Word count, whitespace normalisation, German sentence splitting, short labels |
| `aam/cache.py` | `JsonCache` (copied) plus `MemoryCache` for one Streamlit session |
| `aam/envfile.py` | `.env` loader (copied) |
| `aam/embeddings/` | Provider layer OpenAI / Gemini / Ollama and `embed_cached` (copied, Gemini gets a task type) |
| `aam/ingest/tables.py` | Robust CSV/Excel reader (copied) |
| `aam/ingest/urls.py` | Parse a URL text field, comparison key for URLs |
| `aam/ingest/content.py` | Table with URL + main content → `Document` list |
| `aam/ingest/chunks_file.py` | Write and read the chunk-embedding export |
| `aam/extract.py` | Fetch URLs and extract main content with trafilatura |
| `aam/segment.py` | AI answer → topics and statements, target size |
| `aam/chunk.py` | Documents → chunks around the target size, statistics |
| `aam/match.py` | Cosine matching: page ranking, topic hits, statement hits, status |
| `aam/pipeline.py` | `prepare` (segment + chunk), `analyze` (embed + match), `rematch` |
| `aam/report.py` | DataFrames, summary text, protocol lines, Excel bytes |
| `app.py` | Streamlit UI |
| `tests/fakes.py` | Deterministic fake embedding provider and shared sample data |
| `examples/` | Sample answer text and sample content CSV for the README walkthrough |

---

### Task 1: Scaffold, copied infrastructure, models, fakes

**Files:**
- Create: `requirements.txt`, `requirements-dev.txt`, `aam/__init__.py`, `aam/ingest/__init__.py`, `aam/models.py`, `tests/__init__.py`, `tests/fakes.py`, `tests/test_models.py`, `tests/test_memory_cache.py`, `tests/test_fakes.py`
- Copy and adapt from `/Users/Daniel/Downloads/broken-link-matcher`: `blm/embeddings/*.py` → `aam/embeddings/`, `blm/cache.py`, `blm/envfile.py`, `blm/ingest/tables.py`, `tests/conftest.py`, `tests/test_embeddings.py`, `tests/test_cache.py`, `tests/test_envfile.py`, `tests/test_tables.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `aam.models`: constants `OWN = "own"`, `SOURCE = "source"`; dataclasses `Document(url, text, origin)`, `FailedUrl(url, origin, reason)`, `Statement(id, topic_id, text, words)`, `Topic(id, label, statements: tuple)`, `SegmentedAnswer(keyword, topics: tuple, target_words, median_words, method)` with property `statements -> list[Statement]`, `Chunk(url, origin, index, text, words, vector=None)`, `ChunkStats(documents, chunks, median_words, p10_words, p90_words, merged, split, empty_urls=())`, `Hit(url, text, score)`, `StatementMatch(statement, topic_label, own, source, delta, status)`, `TopicMatch(topic, own, source, delta, counts)`, `PageRow(rank, url, origin, page_score, best_chunk_score, covered, total)`, `MatchResult(pages, topics, statements, has_sources, status_labels)`, `RunMeta(...)`, `Result(...)`.
  - `aam.embeddings`: `PROVIDERS`, `DEFAULT_EMBED_MODELS`, `EmbeddingError`, `EmbeddingProvider`, `make_provider(name, model, *, api_key=None, base_url=None, client=None)`, `embed_cached(provider, texts, cache, *, progress=None) -> list[Optional[np.ndarray]]`.
  - `aam.cache`: `JsonCache`, `MemoryCache()` with `get(namespace, key)`, `set(namespace, key, value)`, `clear()`.
  - `aam.ingest.tables`: `read_table_info(source, filename=None) -> (DataFrame, TableInfo)`, `read_table`, `describe_table_info`, `TableInfo`.
  - `aam.envfile.load_env_file(path)`.
  - `tests.fakes`: `FakeProvider(dim=64)` (an `EmbeddingProvider` named `"fake"`, model `"fake-model"`, attribute `calls` counting embedded texts), `bow_vector(text, dim=64)`, `ANSWER`, `OWN_DOCS`, `SOURCE_DOCS`.

- [ ] **Step 1: Create the environment files**

`requirements.txt`:

```text
# Runtime dependencies (also what Streamlit Community Cloud installs).
streamlit>=1.64,<2
httpx>=0.27,<1
pandas>=2.2,<4
numpy>=1.26,<3
trafilatura>=1.12,<3
lxml>=5,<7
openpyxl>=3.1,<4
```

`requirements-dev.txt`:

```text
# Runtime plus test tools: pip install -r requirements-dev.txt
-r requirements.txt
pytest>=8,<10
respx>=0.21,<1
```

- [ ] **Step 2: Copy the shared infrastructure and rename the package**

```bash
cd /Users/Daniel/Downloads/ai-answer-matcher
SRC=/Users/Daniel/Downloads/broken-link-matcher
mkdir -p aam/embeddings aam/ingest tests examples
cp "$SRC"/blm/embeddings/__init__.py "$SRC"/blm/embeddings/base.py "$SRC"/blm/embeddings/openai.py "$SRC"/blm/embeddings/gemini.py "$SRC"/blm/embeddings/ollama.py aam/embeddings/
cp "$SRC"/blm/cache.py "$SRC"/blm/envfile.py aam/
cp "$SRC"/blm/ingest/tables.py aam/ingest/
cp "$SRC"/tests/conftest.py "$SRC"/tests/test_embeddings.py "$SRC"/tests/test_cache.py "$SRC"/tests/test_envfile.py "$SRC"/tests/test_tables.py tests/
touch aam/__init__.py aam/ingest/__init__.py tests/__init__.py
perl -pi -e 's/\bblm\b/aam/g' aam/embeddings/*.py aam/cache.py aam/envfile.py aam/ingest/tables.py tests/*.py
grep -rn "blm" aam tests || echo "no blm references left"
```

Expected: `no blm references left`.

`tests/test_tables.py` contains one test that imports a module this repo does not have. Delete the whole function `test_read_table_drops_info_and_backlinks_module_reexports_it` (the `def` line through the blank lines before the next `@pytest.mark.parametrize`).

In `aam/embeddings/__init__.py` replace the module docstring with `"""Embedding providers. Every text of a run goes through the same provider and model."""`. In `aam/ingest/tables.py` replace the second paragraph of the module docstring with `Shared by the content import and the chunk-embedding import. Every cell comes back as a string (blank cells as ""), so callers parse numbers themselves.` In `aam/cache.py` replace the module docstring with `"""Embedding caches: a JSON-per-entry file cache and an in-memory cache for one session."""`.

- [ ] **Step 3: Create the virtual environment and run the copied tests**

```bash
cd /Users/Daniel/Downloads/ai-answer-matcher
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/pytest -q
```

Expected: all copied tests pass (no failures, no errors).

- [ ] **Step 4: Write the failing tests for the new pieces**

`tests/test_models.py`:

```python
from aam.models import OWN, SOURCE, SegmentedAnswer, Statement, Topic


def test_origin_constants():
    assert (OWN, SOURCE) == ("own", "source")


def test_statements_are_flattened_in_topic_order():
    s1 = Statement(0, 0, "a b", 2)
    s2 = Statement(1, 1, "c", 1)
    answer = SegmentedAnswer("kw", (Topic(0, "T1", (s1,)), Topic(1, "T2", (s2,))), 15, 1.5, "structure")
    assert answer.statements == [s1, s2]
```

`tests/test_memory_cache.py`:

```python
from aam.cache import MemoryCache


def test_memory_cache_roundtrip_namespaces_and_clear():
    cache = MemoryCache()
    assert cache.get("embeddings", "k") is None
    cache.set("embeddings", "k", {"v": [1.0]})
    assert cache.get("embeddings", "k") == {"v": [1.0]}
    assert cache.get("other", "k") is None
    assert cache.clear() == 1
    assert cache.get("embeddings", "k") is None
```

`tests/test_fakes.py`:

```python
import numpy as np

from aam.embeddings import embed_cached
from aam.cache import MemoryCache
from tests.fakes import FakeProvider


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_fake_provider_is_deterministic_and_rewards_shared_words():
    p = FakeProvider()
    a, b, c = p.embed(["Kosten Dach Euro", "Kosten Dach Preis", "Hund Katze Maus"])
    assert cos(a, b) > cos(a, c)
    assert p.embed(["Kosten Dach Euro"])[0].tolist() == a.tolist()


def test_fake_provider_counts_texts_and_memory_cache_avoids_repeats():
    p = FakeProvider()
    cache = MemoryCache()
    embed_cached(p, ["eins", "zwei"], cache)
    embed_cached(p, ["eins", "drei"], cache)
    assert p.calls == 3
```

Append to `tests/test_embeddings.py`:

```python
@respx.mock
def test_gemini_requests_semantic_similarity_task_type():
    import json

    route = respx.post(GEMINI_URL).mock(return_value=httpx.Response(200, json={"embeddings": [{"values": [1.0, 0.0]}]}))
    make_provider("gemini", "gemini-embedding-001", api_key="g").embed(["a"])
    body = json.loads(route.calls[0].request.content)
    assert body["requests"][0]["taskType"] == "SEMANTIC_SIMILARITY"
```

- [ ] **Step 5: Run the new tests to verify they fail**

Run: `.venv/bin/pytest tests/test_models.py tests/test_memory_cache.py tests/test_fakes.py tests/test_embeddings.py -q`
Expected: FAIL (`ModuleNotFoundError: aam.models`, `ImportError: MemoryCache`, `KeyError: 'taskType'`).

- [ ] **Step 6: Implement**

`aam/models.py`:

```python
"""Data structures shared by all modules. No logic beyond one convenience property."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

OWN = "own"
SOURCE = "source"


@dataclass(frozen=True)
class Document:
    url: str
    text: str
    origin: str  # OWN or SOURCE


@dataclass(frozen=True)
class FailedUrl:
    url: str
    origin: str
    reason: str  # German, shown in the protocol


@dataclass(frozen=True)
class Statement:
    id: int
    topic_id: int
    text: str
    words: int


@dataclass(frozen=True)
class Topic:
    id: int
    label: str
    statements: tuple[Statement, ...]


@dataclass(frozen=True)
class SegmentedAnswer:
    keyword: str
    topics: tuple[Topic, ...]
    target_words: int  # chunk size used for own content and sources
    median_words: float  # median statement length the target was derived from
    method: str  # "structure" or "semantic"

    @property
    def statements(self) -> list[Statement]:
        return [s for topic in self.topics for s in topic.statements]


@dataclass
class Chunk:
    url: str
    origin: str
    index: int
    text: str
    words: int
    vector: Optional[np.ndarray] = None


@dataclass(frozen=True)
class ChunkStats:
    documents: int
    chunks: int
    median_words: float
    p10_words: float
    p90_words: float
    merged: int
    split: int
    empty_urls: tuple[str, ...] = ()


@dataclass(frozen=True)
class Hit:
    """Best matching chunk for a statement or topic."""

    url: str
    text: str
    score: float


@dataclass(frozen=True)
class StatementMatch:
    statement: Statement
    topic_label: str
    own: Hit
    source: Optional[Hit]
    delta: Optional[float]  # own score minus source score; None without sources
    status: str  # "" when no status can be given


@dataclass(frozen=True)
class TopicMatch:
    topic: Topic
    own: Hit
    source: Optional[Hit]
    delta: Optional[float]
    counts: dict  # status label -> number of statements


@dataclass(frozen=True)
class PageRow:
    rank: int
    url: str
    origin: str
    page_score: float
    best_chunk_score: float
    covered: int
    total: int


@dataclass(frozen=True)
class MatchResult:
    pages: list
    topics: list
    statements: list
    has_sources: bool
    status_labels: tuple[str, ...]


@dataclass(frozen=True)
class RunMeta:
    keyword: str
    date: str  # ISO date of the run
    provider: str
    model: str
    dimension: int
    target_words: int
    median_words: float
    method: str
    covered_delta: float
    weak_delta: float
    failed: tuple[FailedUrl, ...]
    warnings: tuple[str, ...]
    prechunked: bool = False  # own content came from a chunk-embedding export


@dataclass(frozen=True)
class Result:
    meta: RunMeta
    match: MatchResult
    answer: SegmentedAnswer
    own_chunks: list
    source_chunks: list
    own_stats: ChunkStats
    source_stats: Optional[ChunkStats]
    statement_vectors: list  # kept so thresholds can change without re-embedding
```

Append to `aam/cache.py`:

```python


class MemoryCache:
    """Cache for one Streamlit session with the JsonCache interface; nothing touches the disk."""

    def __init__(self):
        self._data: dict[tuple[str, str], dict] = {}

    def get(self, namespace: str, key: str) -> Optional[dict]:
        return self._data.get((namespace, key))

    def set(self, namespace: str, key: str, value: dict) -> None:
        self._data[(namespace, key)] = value

    def clear(self) -> int:
        removed = len(self._data)
        self._data.clear()
        return removed
```

In `aam/embeddings/gemini.py`, replace the `body = ...` line in `_embed_batch` with:

```python
        # SEMANTIC_SIMILARITY: both sides of every comparison are passages, not query vs. document.
        body = {"requests": [{"model": f"models/{self.model}", "taskType": "SEMANTIC_SIMILARITY",
                              "content": {"parts": [{"text": t}]}} for t in texts]}
```

`tests/fakes.py`:

```python
"""Deterministic stand-ins for tests: a bag-of-words embedding provider and sample data."""

from __future__ import annotations

import hashlib
import re

import numpy as np

from aam.embeddings.base import EmbeddingProvider
from aam.models import OWN, SOURCE, Document


def bow_vector(text: str, dim: int = 64) -> np.ndarray:
    """Hash every word into one of `dim` buckets; shared words give a high cosine."""
    vector = np.zeros(dim, dtype=np.float32)
    for word in re.findall(r"\w+", text.lower()):
        vector[int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % dim] += 1.0
    return vector


class FakeProvider(EmbeddingProvider):
    name = "fake"
    batch_size = 64

    def __init__(self, dim: int = 64):
        super().__init__("fake-model")
        self.dim = dim
        self.calls = 0  # number of texts embedded

    def _embed_batch(self, texts):
        self.calls += len(texts)
        return [bow_vector(t, self.dim).tolist() for t in texts]


ANSWER = """Eine Dachsanierung kostet je nach Umfang zwischen 100 und 300 Euro pro Quadratmeter.

Kosten im Überblick
Eine Neueindeckung ohne Dämmung liegt bei 100 bis 150 Euro pro Quadratmeter.
Mit Dämmung steigen die Kosten auf 200 bis 300 Euro pro Quadratmeter.

Förderung
Die KfW fördert die Dachdämmung mit zinsgünstigen Krediten.
Das BAFA zahlt Zuschüsse von bis zu 20 Prozent.
"""

OWN_DOCS = [
    Document(
        "https://me.de/kosten",
        "Kosten der Dachsanierung\n"
        "Eine Neueindeckung ohne Dämmung liegt bei 100 bis 150 Euro pro Quadratmeter.\n"
        "Mit Dämmung steigen die Kosten auf 200 bis 300 Euro pro Quadratmeter.",
        OWN,
    ),
    Document(
        "https://me.de/blog",
        "Unser Team stellt sich vor.\n"
        "Wir sind seit 20 Jahren als Dachdecker in Köln tätig und bilden regelmäßig aus.",
        OWN,
    ),
]

SOURCE_DOCS = [
    Document(
        "https://quelle.de/foerderung",
        "Die KfW fördert die Dachdämmung mit zinsgünstigen Krediten.\n"
        "Das BAFA zahlt Zuschüsse von bis zu 20 Prozent.",
        SOURCE,
    ),
]
```

- [ ] **Step 7: Run the whole suite**

Run: `.venv/bin/pytest -q`
Expected: PASS, no failures.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "Scaffold package with copied provider layer, models and test fakes

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Text helpers and answer segmentation

**Files:**
- Create: `aam/text.py`, `aam/segment.py`
- Test: `tests/test_text.py`, `tests/test_segment.py`

**Interfaces:**
- Consumes: `aam.models.SegmentedAnswer`, `Statement`, `Topic`; `tests.fakes.FakeProvider`, `ANSWER`.
- Produces:
  - `aam.text.count_words(text) -> int`, `normalize_space(text) -> str`, `split_sentences(text) -> list[str]`, `short_label(text, max_words=8) -> str`.
  - `aam.segment.segment_answer(keyword: str, text: str, *, target_override: Optional[int] = None, embed: Optional[Callable[[list[str]], Sequence[Optional[np.ndarray]]]] = None) -> SegmentedAnswer`. Raises `ValueError` (German message) when no statement is found.
  - `aam.segment.derive_target(word_counts: list[int]) -> tuple[int, float]` returning `(target_words, median_words)`.
  - Constants `MIN_TARGET = 15`, `MAX_TARGET = 80`, `INTRO_LABEL = "Einleitung"`.

Segmentation rules (spec section 5, with one clarification): a blank line starts a new topic **only when the answer contains no heading and no lead-in at all**. When explicit markers exist they alone define the topics, otherwise a second paragraph under a heading would be torn out of its topic.

- [ ] **Step 1: Write the failing tests**

`tests/test_text.py`:

```python
from aam.text import count_words, normalize_space, short_label, split_sentences


def test_count_words_and_normalize_space():
    assert count_words("  Eine  Dachsanierung\tkostet Geld ") == 4
    assert normalize_space("  a \n b\t c ") == "a b c"


def test_split_sentences_basic():
    assert split_sentences("Das Dach ist alt. Es muss saniert werden! Wirklich? Ja.") == [
        "Das Dach ist alt.", "Es muss saniert werden!", "Wirklich?", "Ja."]


def test_split_sentences_keeps_abbreviations_together():
    text = "Das kostet z. B. 500 Euro. Die Dämmung kostet ca. 100 Euro bzw. etwas mehr. Siehe Nr. 3 der Liste."
    assert split_sentences(text) == [
        "Das kostet z. B. 500 Euro.",
        "Die Dämmung kostet ca. 100 Euro bzw. etwas mehr.",
        "Siehe Nr. 3 der Liste.",
    ]


def test_split_sentences_keeps_ordinals_together():
    assert split_sentences("Am 1. Januar steigt der Preis. Das gilt bundesweit.") == [
        "Am 1. Januar steigt der Preis.", "Das gilt bundesweit."]


def test_split_sentences_does_not_mistake_word_endings_for_abbreviations():
    assert split_sentences("Die Dämmung ist Pflicht und es bleibt ein Rest. Danach folgt der Estrich.") == [
        "Die Dämmung ist Pflicht und es bleibt ein Rest.", "Danach folgt der Estrich."]


def test_split_sentences_empty():
    assert split_sentences("   ") == []


def test_short_label():
    assert short_label("Kosten im Überblick:") == "Kosten im Überblick"
    assert short_label("eins zwei drei vier fünf sechs sieben acht neun zehn") == "eins zwei drei vier fünf sechs sieben acht …"
```

`tests/test_segment.py`:

```python
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
    assert sorted(s.text for s in answer.statements) == ["Dachdämmung", "Kellerdämmung"]
    assert "Fassadendämmung" in labels(answer)


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_text.py tests/test_segment.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'aam.text'`.

- [ ] **Step 3: Implement `aam/text.py`**

```python
"""Small text helpers shared by segmentation and chunking."""

from __future__ import annotations

import re

# A full stop after one of these does not end a sentence.
_ABBREVIATIONS = (
    "z. b.", "z.b.", "d. h.", "d.h.", "u. a.", "u.a.", "o. ä.", "o.ä.",
    "ca.", "bzw.", "nr.", "usw.", "inkl.", "ggf.", "evtl.", "vgl.", "zzgl.",
    "sog.", "mind.", "max.", "dr.", "prof.", "st.", "abs.",
)
# Only as a whole word: "ist." must not count as the abbreviation "st.".
_ABBREVIATION = re.compile(
    r"(?:^|[\s(])(?:" + "|".join(re.escape(a) for a in _ABBREVIATIONS) + r")$", re.IGNORECASE)
_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ0-9„\"(\[])")
_ORDINAL = re.compile(r"(?:^|\s)\d{1,2}\.$")  # "am 1. Januar"
_SINGLE_LETTER = re.compile(r"(?:^|[\s(])[A-Za-zÄÖÜäöü]\.$")  # the "z." of "z. B."


def count_words(text: str) -> int:
    return len(text.split())


def normalize_space(text: str) -> str:
    return " ".join(text.split())


def _continues(piece: str) -> bool:
    """True when the piece ends in an abbreviation or ordinal, so the sentence goes on."""
    return (bool(_ABBREVIATION.search(piece))
            or bool(_ORDINAL.search(piece))
            or bool(_SINGLE_LETTER.search(piece)))


def split_sentences(text: str) -> list[str]:
    """Split German text into sentences; whitespace is normalised."""
    text = normalize_space(text)
    if not text:
        return []
    sentences: list[str] = []
    for part in _BOUNDARY.split(text):
        if sentences and _continues(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {part}"
        else:
            sentences.append(part)
    return sentences


def short_label(text: str, max_words: int = 8) -> str:
    """Topic label from running text: at most `max_words` words, no trailing punctuation."""
    words = text.split()
    if len(words) <= max_words:
        return text.strip().rstrip(".:;,").strip()
    return " ".join(words[:max_words]) + " …"
```

- [ ] **Step 4: Implement `aam/segment.py`**

```python
"""Split an AI answer into topics and statements and derive the chunk target size.

Structure first: headings, lead-ins ("Kosten: ...") and, when neither exists, blank lines.
If that yields a single topic and an embedding function is available, adjacent sentences
are compared and the answer is cut where their similarity drops.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from aam.models import SegmentedAnswer, Statement, Topic
from aam.text import count_words, normalize_space, short_label, split_sentences

MIN_TARGET = 15
MAX_TARGET = 80
MAX_HEADING_WORDS = 10
MAX_LEAD_WORDS = 5
MAX_STATEMENT_WORDS = 60
MIN_FALLBACK_SENTENCES = 6
INTRO_LABEL = "Einleitung"

Embed = Callable[[list[str]], Sequence[Optional[np.ndarray]]]

_BULLET = re.compile(r"^\s*(?:[-*•–·▪◦]|\d{1,2}[.)])\s+")
_MD_HEADING = re.compile(r"^\s*#{1,6}\s+")
_WHOLE_BOLD = re.compile(r"^\s*(\*\*|__)(.+?)\1\s*:?\s*$")
_LEAD_IN = re.compile(r"^(?P<label>[^:.!?]{1,80}):\s+(?P<rest>\S.*)$")
_SENTENCE_END = re.compile(r"[.!?]$")


@dataclass
class _Line:
    text: str
    bullet: bool
    forced_heading: bool  # markdown heading or a line that is bold as a whole
    after_blank: bool


@dataclass
class _Draft:
    label: str
    texts: list[str] = field(default_factory=list)
    forced: bool = False


def _parse_lines(text: str) -> list[_Line]:
    lines: list[_Line] = []
    blank = False
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if not raw.strip():
            blank = True
            continue
        bullet = bool(_BULLET.match(raw))
        body = _BULLET.sub("", raw, count=1) if bullet else raw
        forced = bool(_MD_HEADING.match(body)) or bool(_WHOLE_BOLD.match(body))
        body = _MD_HEADING.sub("", body, count=1)
        body = normalize_space(body.replace("**", "").replace("__", ""))
        if not body:
            continue
        lines.append(_Line(body, bullet, forced, blank))
        blank = False
    return lines


def _is_heading(line: _Line, has_more: bool) -> bool:
    if not has_more:
        return False  # a heading needs text after it
    if line.forced_heading:
        return True
    if line.bullet:
        return False
    return count_words(line.text) <= MAX_HEADING_WORDS and not _SENTENCE_END.search(line.text)


def _lead_in(line: _Line) -> Optional[re.Match]:
    match = _LEAD_IN.match(line.text)
    if match and count_words(match["label"]) <= MAX_LEAD_WORDS:
        return match
    return None


def _structure_topics(lines: list[_Line]) -> list[_Draft]:
    last = len(lines) - 1
    explicit = any(_is_heading(line, i < last) or _lead_in(line) for i, line in enumerate(lines))
    drafts: list[_Draft] = []
    for i, line in enumerate(lines):
        if _is_heading(line, i < last):
            drafts.append(_Draft(line.text.rstrip(":").strip(), forced=line.forced_heading))
            continue
        lead = _lead_in(line)
        if lead:
            drafts.append(_Draft(lead["label"].strip(), [lead["rest"].strip()]))
            continue
        if not drafts:
            drafts.append(_Draft(INTRO_LABEL))
        elif not explicit and line.after_blank and not line.bullet and drafts[-1].texts:
            drafts.append(_Draft(short_label(line.text)))
        drafts[-1].texts.append(line.text)
    return _fold_empty(drafts)


def _fold_empty(drafts: list[_Draft]) -> list[_Draft]:
    """A heading without statements keeps its text: it becomes a statement of the topic before it."""
    out: list[_Draft] = []
    for draft in drafts:
        if draft.texts:
            out.append(draft)
        elif not draft.forced:
            if not out:
                out.append(_Draft(INTRO_LABEL))
            out[-1].texts.append(draft.label)
    return out


def _split_long(texts: list[str]) -> list[str]:
    out: list[str] = []
    for text in texts:
        if count_words(text) > MAX_STATEMENT_WORDS:
            out.extend(split_sentences(text))
        else:
            out.append(text)
    return out


def _semantic_topics(sentences: list[str], embed: Embed) -> Optional[list[_Draft]]:
    vectors = embed(sentences)
    if any(v is None for v in vectors):
        return None
    matrix = np.vstack(vectors).astype(np.float32)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    similarities = np.sum(matrix[:-1] * matrix[1:], axis=1)
    threshold = float(similarities.mean() - similarities.std())
    groups = [[sentences[0]]]
    for sentence, similarity in zip(sentences[1:], similarities):
        if similarity < threshold:
            groups.append([sentence])
        else:
            groups[-1].append(sentence)
    if len(groups) < 2:
        return None
    return [_Draft(short_label(group[0]), group) for group in groups]


def derive_target(word_counts: list[int]) -> tuple[int, float]:
    """Chunk target size (median, clamped to MIN_TARGET..MAX_TARGET) and the raw median."""
    median = float(statistics.median(word_counts))
    return int(min(MAX_TARGET, max(MIN_TARGET, round(median)))), median


def segment_answer(keyword: str, text: str, *, target_override: Optional[int] = None,
                   embed: Optional[Embed] = None) -> SegmentedAnswer:
    drafts = [_Draft(d.label, _split_long(d.texts)) for d in _structure_topics(_parse_lines(text))]
    if not drafts:
        raise ValueError("Die AI-Antwort enthält keine erkennbare Aussage.")
    method = "structure"
    if len(drafts) == 1 and embed is not None:
        sentences = [s for t in drafts[0].texts for s in split_sentences(t)]
        if len(sentences) >= MIN_FALLBACK_SENTENCES:
            semantic = _semantic_topics(sentences, embed)
            if semantic:
                drafts, method = semantic, "semantic"
    topics: list[Topic] = []
    next_id = 0
    for topic_id, draft in enumerate(drafts):
        statements = []
        for statement_text in draft.texts:
            statements.append(Statement(next_id, topic_id, statement_text, count_words(statement_text)))
            next_id += 1
        topics.append(Topic(topic_id, draft.label, tuple(statements)))
    derived, median = derive_target([s.words for t in topics for s in t.statements])
    target = int(target_override) if target_override else derived
    return SegmentedAnswer(keyword.strip(), tuple(topics), target, median, method)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_text.py tests/test_segment.py -q`
Expected: PASS.

If `test_semantic_fallback_splits_prose_at_the_topic_change` fails because hash collisions of the fake embedder blur the boundary, change words in `PROSE` (keep three sentences per theme with heavy word overlap inside a theme and none across). Do not weaken the assertions.

- [ ] **Step 6: Commit**

```bash
git add aam/text.py aam/segment.py tests/test_text.py tests/test_segment.py
git commit -m "Add answer segmentation into topics and statements

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Chunking

**Files:**
- Create: `aam/chunk.py`
- Test: `tests/test_chunk.py`

**Interfaces:**
- Consumes: `aam.models.Document`, `Chunk`, `ChunkStats`; `aam.text.count_words`, `normalize_space`, `split_sentences`.
- Produces:
  - `aam.chunk.chunk_document(doc: Document, target_words: int) -> tuple[list[Chunk], int, int]` returning `(chunks, merged, split)`.
  - `aam.chunk.chunk_documents(docs: Sequence[Document], target_words: int) -> tuple[list[Chunk], ChunkStats]`.
  - `aam.chunk.stats_for(chunks: Sequence[Chunk], *, documents: int, merged: int = 0, split: int = 0, empty_urls: Sequence[str] = ()) -> ChunkStats`.

Rules (spec section 6): paragraphs come from line breaks. A paragraph above twice the target is cut into sentence windows of about the target size with one sentence of overlap. A piece below half the target is merged with the following piece as long as the result stays within twice the target; a short last piece is merged with the one before it.

- [ ] **Step 1: Write the failing tests**

`tests/test_chunk.py`:

```python
from aam.chunk import chunk_document, chunk_documents, stats_for
from aam.models import OWN, Chunk, Document


def sentence(i: int, n: int = 12) -> str:
    """A sentence with exactly n words that the sentence splitter recognises."""
    return f"Satz{i} " + " ".join(f"wort{i}{j}" for j in range(n - 2)) + f" schluss{i}."


def doc(text: str, url: str = "https://me.de/a") -> Document:
    return Document(url, text, OWN)


def test_short_paragraphs_are_merged_with_their_neighbours():
    text = "\n".join(["Kosten der Dachsanierung im Überblick", sentence(1, 22), sentence(2, 3)])
    chunks, merged, split = chunk_document(doc(text), 20)
    assert len(chunks) == 1
    assert chunks[0].words == 30
    assert chunks[0].text.startswith("Kosten der Dachsanierung im Überblick Satz1")
    assert (merged, split) == (2, 0)
    assert (chunks[0].url, chunks[0].origin, chunks[0].index, chunks[0].vector) == ("https://me.de/a", OWN, 0, None)


def test_no_merge_when_the_result_would_exceed_twice_the_target():
    text = "\n".join(["Kosten der Dachsanierung im Überblick", sentence(1, 38)])
    chunks, merged, split = chunk_document(doc(text), 20)
    assert [c.words for c in chunks] == [5, 38]
    assert (merged, split) == (0, 0)


def test_long_paragraph_is_cut_into_windows_with_one_sentence_overlap():
    paragraph = " ".join(sentence(i) for i in range(1, 6))  # 60 words, target 20
    chunks, merged, split = chunk_document(doc(paragraph), 20)
    assert [c.text for c in chunks] == [
        f"{sentence(1)} {sentence(2)}",
        f"{sentence(2)} {sentence(3)}",
        f"{sentence(3)} {sentence(4)}",
        f"{sentence(4)} {sentence(5)}",
    ]
    assert [c.index for c in chunks] == [0, 1, 2, 3]
    assert (merged, split) == (0, 1)


def test_window_remainder_is_kept():
    paragraph = " ".join(sentence(i) for i in range(1, 7))  # 72 words, target 30
    chunks, _, _ = chunk_document(doc(paragraph), 30)
    assert [c.words for c in chunks] == [36, 36, 24]
    assert chunks[2].text == f"{sentence(5)} {sentence(6)}"


def test_single_oversized_sentence_stays_one_chunk():
    chunks, merged, split = chunk_document(doc(sentence(1, 50)), 20)
    assert [c.words for c in chunks] == [50]
    assert (merged, split) == (0, 0)


def test_whitespace_is_normalised_and_blank_lines_are_ignored():
    chunks, _, _ = chunk_document(doc(f"\n\n  {sentence(1, 20)}  \n\n\t\n{sentence(2, 20)}\n"), 20)
    assert [c.text for c in chunks] == [sentence(1, 20), sentence(2, 20)]


def test_empty_document_yields_no_chunks_and_is_reported():
    chunks, stats = chunk_documents([doc(" \n ", "https://me.de/leer"), doc(sentence(1, 20))], 20)
    assert len(chunks) == 1
    assert stats.documents == 2
    assert stats.empty_urls == ("https://me.de/leer",)


def test_stats_describe_the_word_distribution():
    docs = [doc(sentence(1, n), f"https://me.de/{n}") for n in (10, 20, 30)]
    chunks, stats = chunk_documents(docs, 20)
    assert [c.words for c in chunks] == [10, 20, 30]
    assert (stats.documents, stats.chunks, stats.merged, stats.split) == (3, 3, 0, 0)
    assert (stats.median_words, stats.p10_words, stats.p90_words) == (20.0, 12.0, 28.0)


def test_stats_for_without_chunks():
    stats = stats_for([], documents=1, empty_urls=["u"])
    assert (stats.chunks, stats.median_words, stats.empty_urls) == (0, 0.0, ("u",))


def test_stats_for_existing_chunks():
    chunks = [Chunk("u", OWN, 0, "a b", 2), Chunk("u", OWN, 1, "a b c d", 4)]
    assert stats_for(chunks, documents=1).median_words == 3.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_chunk.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'aam.chunk'`.

- [ ] **Step 3: Implement `aam/chunk.py`**

```python
"""Cut documents into chunks whose size follows the statements of the AI answer."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from aam.models import Chunk, ChunkStats, Document
from aam.text import count_words, normalize_space, split_sentences


def _windows(paragraph: str, target_words: int) -> list[str]:
    """Sentence windows of about `target_words` words; each starts with the previous window's last sentence."""
    limit = 2 * target_words
    out: list[str] = []
    current: list[str] = []
    fresh = 0  # sentences in `current` that are not part of an emitted window yet
    for sentence in split_sentences(paragraph):
        if count_words(sentence) > limit:
            if fresh:
                out.append(" ".join(current))
            out.append(sentence)  # cannot be cut any further
            current, fresh = [], 0
            continue
        current.append(sentence)
        fresh += 1
        if sum(count_words(s) for s in current) >= target_words:
            out.append(" ".join(current))
            current, fresh = [sentence], 0
    if fresh:
        out.append(" ".join(current))
    return out


def chunk_document(doc: Document, target_words: int) -> tuple[list[Chunk], int, int]:
    """Return (chunks, merged paragraphs, split paragraphs) for one document."""
    limit = 2 * target_words
    half = target_words / 2
    pieces: list[str] = []
    split = 0
    for line in doc.text.split("\n"):
        paragraph = normalize_space(line)
        if not paragraph:
            continue
        if count_words(paragraph) > limit:
            windows = _windows(paragraph, target_words)
            if len(windows) > 1:
                split += 1
            pieces.extend(windows)
        else:
            pieces.append(paragraph)

    texts: list[str] = []
    merged = 0
    for piece in pieces:
        if texts and count_words(texts[-1]) < half and count_words(texts[-1]) + count_words(piece) <= limit:
            texts[-1] = f"{texts[-1]} {piece}"
            merged += 1
        else:
            texts.append(piece)
    if len(texts) >= 2 and count_words(texts[-1]) < half and count_words(texts[-2]) + count_words(texts[-1]) <= limit:
        tail = texts.pop()
        texts[-1] = f"{texts[-1]} {tail}"
        merged += 1

    chunks = [Chunk(doc.url, doc.origin, i, text, count_words(text)) for i, text in enumerate(texts)]
    return chunks, merged, split


def stats_for(chunks: Sequence[Chunk], *, documents: int, merged: int = 0, split: int = 0,
              empty_urls: Sequence[str] = ()) -> ChunkStats:
    words = [c.words for c in chunks]
    if not words:
        return ChunkStats(documents, 0, 0.0, 0.0, 0.0, merged, split, tuple(empty_urls))
    p10, median, p90 = (float(v) for v in np.percentile(words, [10, 50, 90]))
    return ChunkStats(documents, len(words), median, p10, p90, merged, split, tuple(empty_urls))


def chunk_documents(docs: Sequence[Document], target_words: int) -> tuple[list[Chunk], ChunkStats]:
    chunks: list[Chunk] = []
    merged = split = 0
    empty: list[str] = []
    for doc in docs:
        doc_chunks, doc_merged, doc_split = chunk_document(doc, target_words)
        if not doc_chunks:
            empty.append(doc.url)
        chunks.extend(doc_chunks)
        merged += doc_merged
        split += doc_split
    return chunks, stats_for(chunks, documents=len(docs), merged=merged, split=split, empty_urls=empty)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_chunk.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aam/chunk.py tests/test_chunk.py
git commit -m "Add structural chunking around the answer-derived target size

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Content ingest and URL extraction

**Files:**
- Create: `aam/ingest/urls.py`, `aam/ingest/content.py`, `aam/extract.py`
- Test: `tests/test_urls.py`, `tests/test_content.py`, `tests/test_extract.py`

**Interfaces:**
- Consumes: `aam.ingest.tables.read_table_info`, `TableInfo`; `aam.models.Document`, `FailedUrl`, `OWN`.
- Produces:
  - `aam.ingest.urls.parse_urls(text: str) -> list[str]` (adds `https://` when the scheme is missing, drops fragments, de-duplicates, skips tokens without a dotted host), `url_key(url: str) -> str` (host lowercased + path without trailing slash + query; scheme ignored).
  - `aam.ingest.content.ContentInfo(url_column, content_column, rows, documents, empty, duplicates)`, `documents_from_frame(df, *, origin=OWN) -> tuple[list[Document], ContentInfo]`, `read_content_table(source, filename=None, *, origin=OWN) -> tuple[list[Document], ContentInfo]`. Both raise `ValueError` (German) when a column is missing.
  - `aam.extract.extract_text(html) -> str`, `fetch_documents(urls, origin, client: httpx.Client, *, timeout: float = 20.0, progress: Optional[Callable[[int, int], None]] = None) -> tuple[list[Document], list[FailedUrl]]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_urls.py`:

```python
from aam.ingest.urls import parse_urls, url_key


def test_parse_urls_adds_scheme_dedupes_and_drops_garbage():
    text = "https://a.de/x\n a.de/x/ \nwww.b.de/y#frag,\n\nkeineurl\nhttp://c.de/z?q=1"
    assert parse_urls(text) == ["https://a.de/x", "https://www.b.de/y", "http://c.de/z?q=1"]


def test_parse_urls_empty():
    assert parse_urls("") == []
    assert parse_urls(None) == []


def test_url_key_ignores_scheme_case_and_trailing_slash():
    assert url_key("HTTPS://A.de/x/") == url_key("http://a.de/x") == "a.de/x"
    assert url_key("https://a.de/x?p=1") == "a.de/x?p=1"
    assert url_key("https://a.de/") == "a.de"
```

`tests/test_content.py`:

```python
import io

import pandas as pd
import pytest

from aam.ingest.content import documents_from_frame, read_content_table
from aam.models import OWN, SOURCE

FROG_CSV = (
    "Address,Extract Main Content 1,Title\n"
    'https://me.de/a,"Absatz eins.\nAbsatz zwei.",A\n'
    "https://me.de/b,,B\n"
    "https://me.de/a/,Doppelt,C\n"
)


def test_frog_export_is_read_with_paragraphs_empty_rows_and_duplicates():
    docs, info = read_content_table(io.BytesIO(FROG_CSV.encode("utf-8")), "frog.csv")
    assert [d.url for d in docs] == ["https://me.de/a", "https://me.de/b"]
    assert docs[0].text == "Absatz eins.\nAbsatz zwei."
    assert docs[0].origin == OWN
    assert (info.url_column, info.content_column) == ("Address", "Extract Main Content 1")
    assert (info.rows, info.documents, info.empty, info.duplicates) == (3, 2, 1, 1)


def test_utf16_tab_export_with_german_headers():
    data = "Adresse\tSeiteninhalt\nhttps://me.de/ä\tGrüße aus Köln\n".encode("utf-16")
    docs, info = read_content_table(io.BytesIO(data), "export.csv", origin=SOURCE)
    assert docs[0].text == "Grüße aus Köln"
    assert docs[0].origin == SOURCE
    assert (info.url_column, info.content_column) == ("Adresse", "Seiteninhalt")


def test_content_column_is_found_by_substring_when_no_exact_name_matches():
    df = pd.DataFrame({"URL": ["https://me.de/a"], "Custom Main Content": ["Text"]})
    docs, info = documents_from_frame(df)
    assert info.content_column == "Custom Main Content"
    assert docs[0].text == "Text"


def test_exact_column_name_wins_over_substring():
    df = pd.DataFrame({"url": ["https://me.de/a"], "Content Type": ["text/html"], "Main Content": ["Text"]})
    assert documents_from_frame(df)[1].content_column == "Main Content"


def test_rows_without_url_are_skipped():
    df = pd.DataFrame({"url": ["", "https://me.de/a"], "text": ["x", "y"]})
    docs, info = documents_from_frame(df)
    assert [d.text for d in docs] == ["y"]
    assert info.documents == 1


@pytest.mark.parametrize("columns", [["Foo", "Bar"], ["url", "Bar"], ["Foo", "text"]])
def test_missing_columns_raise_with_the_found_names(columns):
    df = pd.DataFrame({c: ["x"] for c in columns})
    with pytest.raises(ValueError, match=f"Gefunden: {columns[0]}, {columns[1]}"):
        documents_from_frame(df)
```

`tests/test_extract.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_urls.py tests/test_content.py tests/test_extract.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `aam/ingest/urls.py`**

```python
"""Parse a text field with one URL per line and compare URLs loosely."""

from __future__ import annotations

import re
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

_SCHEME = re.compile(r"^https?://", re.IGNORECASE)


def url_key(url: str) -> str:
    """Comparison key: scheme, host case and a trailing slash do not matter."""
    parts = urlsplit(url.strip())
    key = parts.netloc.lower() + parts.path.rstrip("/")
    return f"{key}?{parts.query}" if parts.query else key


def parse_urls(text: Optional[str]) -> list[str]:
    """URLs in input order, without fragments and duplicates; tokens without a dotted host are dropped."""
    urls: list[str] = []
    seen: set[str] = set()
    for token in (text or "").split():
        token = token.strip(",;")
        if not token:
            continue
        if not _SCHEME.match(token):
            token = f"https://{token}"
        parts = urlsplit(token)
        if "." not in parts.netloc:
            continue
        url = urlunsplit((parts.scheme.lower(), parts.netloc, parts.path, parts.query, ""))
        key = url_key(url)
        if key in seen:
            continue
        seen.add(key)
        urls.append(url)
    return urls
```

- [ ] **Step 4: Implement `aam/ingest/content.py`**

```python
"""Read a table with a URL column and a main-content column (e.g. a Screaming Frog export)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

from aam.ingest.tables import read_table_info
from aam.ingest.urls import url_key
from aam.models import OWN, Document

URL_COLUMNS = ("url", "address", "adresse", "page", "seite")
CONTENT_COLUMNS = ("extract main content 1", "main content", "content", "text", "body text",
                   "seiteninhalt", "inhalt")
_CONTENT_HINTS = ("content", "inhalt", "text")


@dataclass(frozen=True)
class ContentInfo:
    url_column: str
    content_column: str
    rows: int
    documents: int
    empty: int  # documents without text
    duplicates: int  # rows dropped because the URL appeared before


def _exact(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    by_lower = {c.strip().lower(): c for c in reversed(columns)}  # first column wins on ties
    for candidate in candidates:
        if candidate in by_lower:
            return by_lower[candidate]
    return None


def _content_column(columns: Sequence[str], url_column: Optional[str]) -> Optional[str]:
    exact = _exact(columns, CONTENT_COLUMNS)
    if exact:
        return exact
    for column in columns:
        if column != url_column and any(hint in column.lower() for hint in _CONTENT_HINTS):
            return column
    return None


def documents_from_frame(df: pd.DataFrame, *, origin: str = OWN) -> tuple[list[Document], ContentInfo]:
    columns = [str(c) for c in df.columns]
    url_column = _exact(columns, URL_COLUMNS)
    content_column = _content_column(columns, url_column)
    if url_column is None or content_column is None:
        raise ValueError(
            "Spalten nicht gefunden. Erwartet werden eine URL-Spalte (z. B. „Address“ oder „URL“) und eine "
            "Content-Spalte (z. B. „Extract Main Content 1“ oder „Main Content“). "
            f"Gefunden: {', '.join(columns)}"
        )
    docs: list[Document] = []
    seen: set[str] = set()
    empty = duplicates = 0
    for url, text in zip(df[url_column], df[content_column]):
        url = str(url).strip()
        text = "" if text is None else str(text).strip()
        if not url:
            continue
        key = url_key(url)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        if not text:
            empty += 1
        docs.append(Document(url, text, origin))
    return docs, ContentInfo(url_column, content_column, len(df), len(docs), empty, duplicates)


def read_content_table(source, filename: Optional[str] = None, *, origin: str = OWN) -> tuple[list[Document], ContentInfo]:
    df, _ = read_table_info(source, filename)
    return documents_from_frame(df, origin=origin)
```

- [ ] **Step 5: Implement `aam/extract.py`**

```python
"""Fetch pages and extract their main content. Failures are collected, never raised."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import httpx
import trafilatura

from aam.models import Document, FailedUrl

USER_AGENT = "Mozilla/5.0 (compatible; AIAnswerMatcher/1.0)"


def extract_text(html) -> str:
    """Main content with one paragraph per line; "" when nothing usable is found."""
    text = trafilatura.extract(html, include_comments=False, include_tables=True, favor_recall=True)
    return text.strip() if text else ""


def fetch_documents(urls: Sequence[str], origin: str, client: httpx.Client, *, timeout: float = 20.0,
                    progress: Optional[Callable[[int, int], None]] = None) -> tuple[list[Document], list[FailedUrl]]:
    docs: list[Document] = []
    failed: list[FailedUrl] = []
    for position, url in enumerate(urls, start=1):
        try:
            response = client.get(url, timeout=timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT})
        except httpx.TimeoutException:
            failed.append(FailedUrl(url, origin, f"Timeout nach {timeout:.0f} s"))
        except (httpx.HTTPError, httpx.InvalidURL, UnicodeError) as exc:
            failed.append(FailedUrl(url, origin, f"Verbindungsfehler ({type(exc).__name__})"))
        else:
            if response.status_code >= 400:
                failed.append(FailedUrl(url, origin, f"HTTP {response.status_code}"))
            else:
                text = extract_text(response.content)
                if text:
                    docs.append(Document(url, text, origin))
                else:
                    failed.append(FailedUrl(url, origin, "kein Main Content gefunden"))
        if progress:
            progress(position, len(urls))
    return docs, failed
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_urls.py tests/test_content.py tests/test_extract.py -q`
Expected: PASS.

If trafilatura returns nothing for the `HTML` fixture, lengthen the three paragraphs of the fixture (trafilatura discards very short pages). Do not add a fallback extractor.

- [ ] **Step 7: Commit**

```bash
git add aam/ingest/urls.py aam/ingest/content.py aam/extract.py tests/test_urls.py tests/test_content.py tests/test_extract.py
git commit -m "Add content table ingest, URL parsing and main-content extraction

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Matching

**Files:**
- Create: `aam/match.py`
- Test: `tests/test_match.py`

**Interfaces:**
- Consumes: `aam.models` (`SegmentedAnswer`, `Statement`, `Topic`, `Chunk`, `Hit`, `StatementMatch`, `TopicMatch`, `PageRow`, `MatchResult`, `OWN`, `SOURCE`).
- Produces:
  - `aam.match.match(answer: SegmentedAnswer, statement_vectors: Sequence[np.ndarray], chunks: Sequence[Chunk], *, covered_delta: float = 0.05, weak_delta: float = 0.15) -> MatchResult`. `statement_vectors[i]` belongs to `answer.statements[i]`. Chunks without a vector are ignored. Raises `ValueError` when no own chunk has a vector.
  - `delta_status(delta, covered_delta, weak_delta) -> str`, `quartile_statuses(scores) -> list[str]`.
  - Constants `STATUS_COVERED = "abgedeckt"`, `STATUS_WEAK = "schwach"`, `STATUS_GAP = "Lücke"`, `DELTA_LABELS`, `QUARTILE_LABELS = ("stark", "über Durchschnitt", "unter Durchschnitt", "schwach")`.

Definitions (spec section 7): page vector = word-weighted mean of the page's chunk vectors, answer vector = word-weighted mean of the statement vectors, topic vector = word-weighted mean of the topic's statement vectors, all L2-normalised. With sources the status of a statement comes from `delta = best own score − best source score`. Without sources it comes from quartiles of the best own scores and needs at least 4 statements. `PageRow.covered` with sources counts statements this page covers (its best chunk within `covered_delta` of the best source); without sources it counts statements for which this page holds the best own chunk.

- [ ] **Step 1: Write the failing tests**

`tests/test_match.py`:

```python
import numpy as np
import pytest
from pytest import approx

from aam.match import QUARTILE_LABELS, delta_status, match, quartile_statuses
from aam.models import OWN, SOURCE, Chunk, SegmentedAnswer, Statement, Topic


def vec(*values):
    return np.array(values, dtype=np.float32)


def chunk(url, origin, index, values, words=10):
    return Chunk(url, origin, index, f"{url}#{index}", words, vec(*values))


def answer(topic_sizes, words=None):
    """Answer with len(topic_sizes) topics; statement i has words[i] words (default 10)."""
    topics, sid = [], 0
    for tid, size in enumerate(topic_sizes):
        statements = []
        for _ in range(size):
            statements.append(Statement(sid, tid, f"s{sid}", (words or {}).get(sid, 10)))
            sid += 1
        topics.append(Topic(tid, f"T{tid}", tuple(statements)))
    return SegmentedAnswer("kw", tuple(topics), 15, 10.0, "structure")


S2 = [vec(1, 0), vec(0, 1)]
CHUNKS = [chunk("/a", OWN, 0, (1, 0)), chunk("/b", OWN, 0, (0.6, 0.8)), chunk("q", SOURCE, 0, (0, 1))]


def test_statement_hits_delta_and_status_with_sources():
    result = match(answer([1, 1]), S2, CHUNKS)
    first, second = result.statements
    assert (first.own.url, first.own.text, first.source.url) == ("/a", "/a#0", "q")
    assert first.own.score == approx(1.0) and first.delta == approx(1.0)
    assert first.status == "abgedeckt"
    assert first.topic_label == "T0"
    assert second.own.url == "/b" and second.own.score == approx(0.8)
    assert second.source.score == approx(1.0) and second.delta == approx(-0.2)
    assert second.status == "Lücke"
    assert result.has_sources is True
    assert result.status_labels == ("abgedeckt", "schwach", "Lücke")


def test_thresholds_are_configurable():
    assert match(answer([1, 1]), S2, CHUNKS, weak_delta=0.25).statements[1].status == "schwach"
    assert match(answer([1, 1]), S2, CHUNKS, covered_delta=0.25, weak_delta=0.3).statements[1].status == "abgedeckt"


@pytest.mark.parametrize("delta,expected", [
    (0.3, "abgedeckt"), (-0.05, "abgedeckt"), (-0.0501, "schwach"),
    (-0.15, "schwach"), (-0.151, "Lücke"),
])
def test_delta_status_boundaries(delta, expected):
    assert delta_status(delta, 0.05, 0.15) == expected


def test_page_ranking_mixes_own_and_source_pages():
    result = match(answer([1, 1]), S2, CHUNKS)
    assert [p.rank for p in result.pages] == [1, 2, 3]
    assert result.pages[0].url == "/b" and result.pages[0].page_score == approx(0.9899, abs=1e-3)
    pages = {p.url: p for p in result.pages}
    assert pages["/a"].page_score == approx(0.7071, abs=1e-3)
    assert pages["/a"].best_chunk_score == approx(1.0)
    assert (pages["/a"].origin, pages["/a"].covered, pages["/a"].total) == (OWN, 1, 2)
    assert pages["/b"].covered == 1
    assert (pages["q"].origin, pages["q"].covered) == (SOURCE, 2)


def test_page_vector_is_weighted_by_chunk_length():
    chunks = [chunk("/a", OWN, 0, (1, 0), words=30), chunk("/a", OWN, 1, (0, 1), words=10)]
    result = match(answer([1]), [vec(1, 0)], chunks)
    assert len(result.pages) == 1
    assert result.pages[0].page_score == approx(0.9487, abs=1e-3)
    assert result.pages[0].best_chunk_score == approx(1.0)


def test_topic_vector_is_weighted_by_statement_length_and_counts_statuses():
    chunks = [chunk("/a", OWN, 0, (1, 0)), chunk("/b", OWN, 0, (0, 1)), chunk("q", SOURCE, 0, (0, 1))]
    result = match(answer([2], words={0: 10, 1: 30}), S2, chunks)
    topic = result.topics[0]
    assert topic.topic.label == "T0"
    assert topic.own.url == "/b" and topic.own.score == approx(0.9487, abs=1e-3)
    assert topic.source.url == "q" and topic.delta == approx(0.0, abs=1e-6)
    assert topic.counts == {"abgedeckt": 2}


def test_without_sources_status_comes_from_quartiles():
    statements = [vec(1, 0), vec(0.9, 0.436), vec(0.6, 0.8), vec(0, 1)]
    result = match(answer([4]), statements, [chunk("/a", OWN, 0, (1, 0)), chunk("/b", OWN, 0, (-1, 0))])
    assert [m.status for m in result.statements] == ["stark", "über Durchschnitt", "unter Durchschnitt", "schwach"]
    assert all(m.source is None and m.delta is None for m in result.statements)
    assert result.has_sources is False and result.status_labels == QUARTILE_LABELS
    assert result.topics[0].source is None and result.topics[0].delta is None
    assert result.topics[0].counts == {"stark": 1, "über Durchschnitt": 1, "unter Durchschnitt": 1, "schwach": 1}
    pages = {p.url: p for p in result.pages}
    assert set(pages) == {"/a", "/b"}
    assert (pages["/a"].covered, pages["/b"].covered, pages["/a"].total) == (4, 0, 4)


def test_without_sources_and_fewer_than_four_statements_there_is_no_status():
    result = match(answer([3]), [vec(1, 0), vec(0, 1), vec(1, 1)], [chunk("/a", OWN, 0, (1, 0))])
    assert [m.status for m in result.statements] == ["", "", ""]
    assert result.topics[0].counts == {}


def test_quartile_statuses():
    assert quartile_statuses([0.9, 0.7, 0.5, 0.3]) == ["stark", "über Durchschnitt", "unter Durchschnitt", "schwach"]
    assert quartile_statuses([0.9, 0.7, 0.5]) == ["", "", ""]


def test_chunks_without_vector_are_ignored_and_missing_own_content_raises():
    no_vector = Chunk("/x", OWN, 0, "x", 5)
    result = match(answer([1]), [vec(1, 0)], [no_vector, chunk("/a", OWN, 0, (1, 0))])
    assert [p.url for p in result.pages] == ["/a"]
    with pytest.raises(ValueError, match="Kein eigener Content"):
        match(answer([1]), [vec(1, 0)], [no_vector, chunk("q", SOURCE, 0, (1, 0))])


def test_vectors_are_normalised_before_comparison():
    result = match(answer([1]), [vec(5, 0)], [chunk("/a", OWN, 0, (3, 0))])
    assert result.statements[0].own.score == approx(1.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_match.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'aam.match'`.

- [ ] **Step 3: Implement `aam/match.py`**

```python
"""Cosine matching between a segmented AI answer and chunks of own content and cited sources."""

from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence

import numpy as np

from aam.models import (OWN, SOURCE, Chunk, Hit, MatchResult, PageRow, SegmentedAnswer,
                        StatementMatch, TopicMatch)

STATUS_COVERED = "abgedeckt"
STATUS_WEAK = "schwach"
STATUS_GAP = "Lücke"
DELTA_LABELS = (STATUS_COVERED, STATUS_WEAK, STATUS_GAP)
QUARTILE_LABELS = ("stark", "über Durchschnitt", "unter Durchschnitt", "schwach")
MIN_STATEMENTS_FOR_QUARTILES = 4
_EPS = 1e-6  # float32 cosines must not flip a status at the threshold


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def weighted_mean(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted mean of the rows, L2-normalised."""
    mean = (matrix * weights[:, None]).sum(axis=0) / max(float(weights.sum()), 1e-12)
    return mean / max(float(np.linalg.norm(mean)), 1e-12)


def delta_status(delta: float, covered_delta: float, weak_delta: float) -> str:
    if delta >= -covered_delta - _EPS:
        return STATUS_COVERED
    if delta >= -weak_delta - _EPS:
        return STATUS_WEAK
    return STATUS_GAP


def quartile_statuses(scores: Sequence[float]) -> list[str]:
    """Relative strength inside the own content; no status below four statements."""
    if len(scores) < MIN_STATEMENTS_FOR_QUARTILES:
        return [""] * len(scores)
    p25, p50, p75 = np.percentile(scores, [25, 50, 75])
    out = []
    for score in scores:
        if score >= p75:
            out.append(QUARTILE_LABELS[0])
        elif score >= p50:
            out.append(QUARTILE_LABELS[1])
        elif score >= p25:
            out.append(QUARTILE_LABELS[2])
        else:
            out.append(QUARTILE_LABELS[3])
    return out


def _matrix(chunks: Sequence[Chunk]) -> np.ndarray:
    return unit_rows(np.vstack([c.vector for c in chunks]).astype(np.float32))


def _hit(chunks: Sequence[Chunk], index: int, score: float) -> Hit:
    chunk = chunks[int(index)]
    return Hit(chunk.url, chunk.text, float(score))


def _words(chunks: Sequence[Chunk], indices: Sequence[int]) -> np.ndarray:
    return np.array([chunks[i].words for i in indices], dtype=np.float32)


def match(answer: SegmentedAnswer, statement_vectors: Sequence[np.ndarray], chunks: Sequence[Chunk], *,
          covered_delta: float = 0.05, weak_delta: float = 0.15) -> MatchResult:
    statements = answer.statements
    s_matrix = unit_rows(np.vstack(statement_vectors).astype(np.float32))
    s_weights = np.array([s.words for s in statements], dtype=np.float32)
    answer_vector = weighted_mean(s_matrix, s_weights)

    own = [c for c in chunks if c.origin == OWN and c.vector is not None]
    sources = [c for c in chunks if c.origin == SOURCE and c.vector is not None]
    if not own:
        raise ValueError("Kein eigener Content mit Embedding vorhanden.")
    has_sources = bool(sources)

    own_matrix = _matrix(own)
    sim_own = s_matrix @ own_matrix.T
    own_best = sim_own.max(axis=1)
    own_arg = sim_own.argmax(axis=1)
    source_matrix = sim_source = source_best = None
    if has_sources:
        source_matrix = _matrix(sources)
        sim_source = s_matrix @ source_matrix.T
        source_best = sim_source.max(axis=1)
        statuses = [delta_status(float(own_best[i] - source_best[i]), covered_delta, weak_delta)
                    for i in range(len(statements))]
    else:
        statuses = quartile_statuses([float(v) for v in own_best])

    labels = {topic.id: topic.label for topic in answer.topics}
    statement_matches = []
    for i, statement in enumerate(statements):
        own_hit = _hit(own, own_arg[i], own_best[i])
        source_hit: Optional[Hit] = None
        delta: Optional[float] = None
        if has_sources:
            j = int(sim_source[i].argmax())
            source_hit = _hit(sources, j, sim_source[i, j])
            delta = own_hit.score - source_hit.score
        statement_matches.append(StatementMatch(statement, labels[statement.topic_id], own_hit, source_hit,
                                                delta, statuses[i]))

    position = {statement.id: i for i, statement in enumerate(statements)}
    topic_matches = []
    for topic in answer.topics:
        rows = [position[s.id] for s in topic.statements]
        topic_vector = weighted_mean(s_matrix[rows], s_weights[rows])
        own_scores = own_matrix @ topic_vector
        own_hit = _hit(own, own_scores.argmax(), own_scores.max())
        source_hit = None
        delta = None
        if has_sources:
            source_scores = source_matrix @ topic_vector
            source_hit = _hit(sources, source_scores.argmax(), source_scores.max())
            delta = own_hit.score - source_hit.score
        counts = dict(Counter(statuses[i] for i in rows if statuses[i]))
        topic_matches.append(TopicMatch(topic, own_hit, source_hit, delta, counts))

    def page_rows(group: Sequence[Chunk], matrix: np.ndarray, sim: np.ndarray, origin: str) -> list[tuple]:
        by_url: dict[str, list[int]] = {}
        for i, chunk in enumerate(group):
            by_url.setdefault(chunk.url, []).append(i)
        rows = []
        for url, indices in by_url.items():
            page_vector = weighted_mean(matrix[indices], _words(group, indices))
            per_statement = sim[:, indices].max(axis=1)
            if has_sources:
                covered = sum(delta_status(float(per_statement[i] - source_best[i]), covered_delta, weak_delta)
                              == STATUS_COVERED for i in range(len(statements)))
            else:
                mine = set(indices)
                covered = sum(int(own_arg[i]) in mine for i in range(len(statements)))
            rows.append((url, origin, float(page_vector @ answer_vector), float(per_statement.max()), int(covered)))
        return rows

    rows = page_rows(own, own_matrix, sim_own, OWN)
    if has_sources:
        rows += page_rows(sources, source_matrix, sim_source, SOURCE)
    rows.sort(key=lambda row: -row[2])
    pages = [PageRow(rank, url, origin, score, best, covered, len(statements))
             for rank, (url, origin, score, best, covered) in enumerate(rows, start=1)]

    return MatchResult(pages, topic_matches, statement_matches, has_sources,
                       DELTA_LABELS if has_sources else QUARTILE_LABELS)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_match.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aam/match.py tests/test_match.py
git commit -m "Add cosine matching for pages, topics and statements

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Chunk-embedding export and import

**Files:**
- Create: `aam/ingest/chunks_file.py`
- Test: `tests/test_chunks_file.py`

**Interfaces:**
- Consumes: `aam.models.Chunk`, `OWN`; `aam.text.count_words`; `aam.ingest.tables.read_table`.
- Produces:
  - `ChunkFileError(ValueError)`.
  - `is_chunks_table(columns: Sequence[str]) -> bool` (true when `chunk_index` and `embedding_0` are present).
  - `chunks_to_csv(chunks: Sequence[Chunk], *, provider: str, model: str, target_words: int) -> bytes` (UTF-8 CSV, columns `url, chunk_index, text, words, provider, model, target_words, embedding_0..N`, vectors rounded to 6 decimals, chunks without vector skipped).
  - `chunks_from_frame(df, *, provider: str, model: str) -> tuple[list[Chunk], int]` and `read_chunks_file(source, filename=None, *, provider: str, model: str) -> tuple[list[Chunk], int]`, both returning `(chunks with vectors and origin OWN, stored target size)`.
  - `check_dimension(chunks: Sequence[Chunk], dimension: int) -> None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_chunks_file.py`:

```python
import io

import numpy as np
import pandas as pd
import pytest
from pytest import approx

from aam.ingest.chunks_file import (ChunkFileError, check_dimension, chunks_from_frame, chunks_to_csv,
                                    is_chunks_table, read_chunks_file)
from aam.models import OWN, Chunk


def sample_chunks():
    return [
        Chunk("https://me.de/a", OWN, 0, "Erster Chunk, mit Komma; und Semikolon", 6,
              np.array([0.1234567, -0.5, 0.25], dtype=np.float32)),
        Chunk("https://me.de/a", OWN, 1, 'Zweiter "Chunk" mit Anführungszeichen', 4,
              np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        Chunk("https://me.de/b", OWN, 0, "Ohne Vektor", 2),
    ]


def exported(**overrides):
    params = {"provider": "openai", "model": "text-embedding-3-small", "target_words": 27, **overrides}
    return chunks_to_csv(sample_chunks(), **params)


def test_roundtrip_keeps_text_order_vectors_and_target():
    chunks, target = read_chunks_file(io.BytesIO(exported()), "chunks_embeddings.csv",
                                      provider="openai", model="text-embedding-3-small")
    assert target == 27
    assert [(c.url, c.index, c.origin) for c in chunks] == [("https://me.de/a", 0, OWN), ("https://me.de/a", 1, OWN)]
    assert chunks[0].text == "Erster Chunk, mit Komma; und Semikolon"
    assert chunks[1].text == 'Zweiter "Chunk" mit Anführungszeichen'
    assert [c.words for c in chunks] == [6, 4]
    assert chunks[0].vector.tolist() == approx([0.123457, -0.5, 0.25], abs=1e-6)
    assert chunks[0].vector.dtype == np.float32


def test_export_columns():
    header = exported().decode("utf-8").split("\n", 1)[0].strip()
    assert header == "url,chunk_index,text,words,provider,model,target_words,embedding_0,embedding_1,embedding_2"


def test_is_chunks_table():
    assert is_chunks_table(["url", "chunk_index", "text", "embedding_0"]) is True
    assert is_chunks_table(["url", "text"]) is False
    assert is_chunks_table(["Address", "embedding_0"]) is False


def test_other_model_is_rejected_with_both_names():
    with pytest.raises(ChunkFileError, match="openai / text-embedding-3-small erzeugt, eingestellt ist openai / other"):
        read_chunks_file(io.BytesIO(exported()), "c.csv", provider="openai", model="other")


def test_other_provider_is_rejected():
    with pytest.raises(ChunkFileError, match="eingestellt ist gemini"):
        read_chunks_file(io.BytesIO(exported()), "c.csv", provider="gemini", model="text-embedding-3-small")


def test_table_that_is_not_an_export_is_rejected():
    df = pd.DataFrame({"url": ["u"], "chunk_index": ["0"], "embedding_0": ["0.1"]})
    with pytest.raises(ChunkFileError, match="kein Chunk-Embedding-Export.*text, words, provider, model, target_words"):
        chunks_from_frame(df, provider="openai", model="m")


def test_invalid_numbers_are_rejected():
    text = exported().decode("utf-8").replace("-0.5", "kaputt")
    with pytest.raises(ChunkFileError, match="ungültige Zahlen"):
        read_chunks_file(io.BytesIO(text.encode("utf-8")), "c.csv", provider="openai", model="text-embedding-3-small")


def test_empty_export_is_rejected():
    header = exported().decode("utf-8").split("\n", 1)[0] + "\n"
    with pytest.raises(ChunkFileError, match="keine Zeilen"):
        read_chunks_file(io.BytesIO(header.encode("utf-8")), "c.csv", provider="openai", model="text-embedding-3-small")


def test_export_without_any_vector_raises():
    with pytest.raises(ChunkFileError, match="Keine Chunks mit Embedding"):
        chunks_to_csv([Chunk("u", OWN, 0, "x", 1)], provider="p", model="m", target_words=20)


def test_check_dimension():
    chunks = sample_chunks()[:2]
    check_dimension(chunks, 3)
    with pytest.raises(ChunkFileError, match="Dimension 3.*Dimension 1536"):
        check_dimension(chunks, 1536)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_chunks_file.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'aam.ingest.chunks_file'`.

- [ ] **Step 3: Implement `aam/ingest/chunks_file.py`**

```python
"""Chunk-embedding export: embed a domain once, reuse it for further keywords."""

from __future__ import annotations

import re
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from aam.ingest.tables import read_table
from aam.models import OWN, Chunk
from aam.text import count_words

META_COLUMNS = ("url", "chunk_index", "text", "words", "provider", "model", "target_words")
_EMBEDDING_COLUMN = re.compile(r"embedding_(\d+)")


class ChunkFileError(ValueError):
    """The uploaded chunk-embedding export cannot be used with the current settings."""


def is_chunks_table(columns: Sequence[str]) -> bool:
    names = {str(c) for c in columns}
    return "chunk_index" in names and "embedding_0" in names


def chunks_to_csv(chunks: Sequence[Chunk], *, provider: str, model: str, target_words: int) -> bytes:
    rows = [c for c in chunks if c.vector is not None]
    if not rows:
        raise ChunkFileError("Keine Chunks mit Embedding vorhanden.")
    meta = pd.DataFrame({
        "url": [c.url for c in rows],
        "chunk_index": [c.index for c in rows],
        "text": [c.text for c in rows],
        "words": [c.words for c in rows],
        "provider": provider,
        "model": model,
        "target_words": target_words,
    })
    matrix = np.round(np.vstack([c.vector for c in rows]).astype(np.float64), 6)
    vectors = pd.DataFrame(matrix, columns=[f"embedding_{i}" for i in range(matrix.shape[1])])
    return pd.concat([meta, vectors], axis=1).to_csv(index=False).encode("utf-8")


def chunks_from_frame(df: pd.DataFrame, *, provider: str, model: str) -> tuple[list[Chunk], int]:
    columns = [str(c) for c in df.columns]
    missing = [c for c in META_COLUMNS if c not in columns]
    embedding_columns = sorted((c for c in columns if _EMBEDDING_COLUMN.fullmatch(c)),
                               key=lambda c: int(c.split("_")[1]))
    if missing or not embedding_columns:
        missing_text = ", ".join(missing) if missing else "embedding_0"
        raise ChunkFileError(
            f"Die Datei ist kein Chunk-Embedding-Export dieses Tools. Fehlende Spalten: {missing_text}")
    if df.empty:
        raise ChunkFileError("Der Chunk-Embedding-Export enthält keine Zeilen.")
    found = {(str(p).strip(), str(m).strip()) for p, m in zip(df["provider"], df["model"])}
    if found != {(provider, model)}:
        described = ", ".join(f"{p} / {m}" for p, m in sorted(found))
        raise ChunkFileError(
            f"Der Export wurde mit {described} erzeugt, eingestellt ist {provider} / {model}. "
            "Bitte Anbieter und Modell in der Seitenleiste anpassen oder den Content neu einbetten.")
    try:
        matrix = df[embedding_columns].to_numpy(dtype=str).astype(np.float32)
        target = int(float(df["target_words"].iloc[0]))
        indices = [int(float(v)) for v in df["chunk_index"]]
    except ValueError as exc:
        raise ChunkFileError("Der Chunk-Embedding-Export enthält ungültige Zahlen.") from exc
    chunks = []
    for row, (url, text) in enumerate(zip(df["url"], df["text"])):
        text = str(text)
        chunks.append(Chunk(str(url).strip(), OWN, indices[row], text, count_words(text), matrix[row]))
    return chunks, target


def read_chunks_file(source, filename: Optional[str] = None, *, provider: str, model: str) -> tuple[list[Chunk], int]:
    try:
        df = read_table(source, filename)
    except ValueError as exc:
        if "leer" in str(exc):
            raise ChunkFileError("Der Chunk-Embedding-Export enthält keine Zeilen.") from exc
        raise
    return chunks_from_frame(df, provider=provider, model=model)


def check_dimension(chunks: Sequence[Chunk], dimension: int) -> None:
    """The uploaded vectors must have the dimension the current model produces."""
    if chunks and len(chunks[0].vector) != dimension:
        raise ChunkFileError(
            f"Die hochgeladenen Chunk-Embeddings haben Dimension {len(chunks[0].vector)}, "
            f"das eingestellte Modell liefert Dimension {dimension}.")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_chunks_file.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aam/ingest/chunks_file.py tests/test_chunks_file.py
git commit -m "Add chunk-embedding export and import for reuse across keywords

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Pipeline

**Files:**
- Create: `aam/pipeline.py`
- Modify: `tests/fakes.py` (append `sample_result`)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `segment_answer` (Task 2), `chunk_documents`, `stats_for` (Task 3), `url_key` (Task 4), `match` (Task 5), `check_dimension` (Task 6), `embed_cached` (Task 1), models.
- Produces:
  - `aam.pipeline.Prepared` dataclass: `answer`, `own_chunks`, `source_chunks`, `own_stats`, `source_stats` (None without sources), `failed: tuple[FailedUrl, ...]`, `warnings: tuple[str, ...]`, `prechunked: bool`, property `texts_to_embed -> int`.
  - `combine_documents(table_docs, fetched_docs) -> list[Document]` (table rows win on the same `url_key`).
  - `prepare(keyword, answer_text, *, own_docs=(), source_docs=(), own_chunks=None, stored_target=None, failed=(), target_override=None, embed=None) -> Prepared`. Raises `ValueError` (German).
  - `analyze(prepared, provider, cache, *, covered_delta=0.05, weak_delta=0.15, progress=None, today=None) -> Result`.
  - `rematch(result, *, covered_delta, weak_delta) -> Result`.
  - `tests.fakes.sample_result(with_sources=True) -> Result`.

- [ ] **Step 1: Write the failing tests**

`tests/test_pipeline.py`:

```python
import io
from datetime import date

import pytest
from pytest import approx

from aam.cache import MemoryCache
from aam.ingest.chunks_file import ChunkFileError, chunks_to_csv, read_chunks_file
from aam.match import QUARTILE_LABELS
from aam.models import OWN, SOURCE, Document, FailedUrl
from aam.pipeline import analyze, combine_documents, prepare, rematch
from tests.fakes import ANSWER, OWN_DOCS, SOURCE_DOCS, FakeProvider, sample_result

KEYWORD = "dachsanierung kosten"
COVERED = "Mit Dämmung steigen die Kosten auf 200 bis 300 Euro pro Quadratmeter."
GAP = "Die KfW fördert die Dachdämmung mit zinsgünstigen Krediten."


def by_text(result):
    return {m.statement.text: m for m in result.match.statements}


def test_prepare_segments_and_chunks_with_the_derived_target():
    prepared = prepare(KEYWORD, ANSWER, own_docs=OWN_DOCS, source_docs=SOURCE_DOCS)
    assert prepared.answer.target_words == 15
    assert (prepared.own_stats.documents, prepared.own_stats.chunks, prepared.own_stats.merged) == (2, 3, 2)
    assert prepared.source_stats.chunks == 2
    assert prepared.prechunked is False and prepared.warnings == () and prepared.failed == ()
    assert prepared.texts_to_embed == 7  # 5 statements + 2 chunks with other text; 3 chunks repeat a statement


def test_target_override_changes_the_chunking():
    prepared = prepare(KEYWORD, ANSWER, own_docs=OWN_DOCS, target_override=60)
    assert prepared.answer.target_words == 60
    assert prepared.own_stats.chunks == 2


def test_analyze_finds_coverage_and_gaps_against_sources():
    result = sample_result()
    covered, gap = by_text(result)[COVERED], by_text(result)[GAP]
    assert covered.status == "abgedeckt"
    assert covered.own.url == "https://me.de/kosten" and covered.own.score == approx(1.0, abs=1e-5)
    assert gap.status == "Lücke" and gap.delta < -0.15
    assert gap.source.url == "https://quelle.de/foerderung"
    own_pages = [p for p in result.match.pages if p.origin == OWN]
    assert own_pages[0].url == "https://me.de/kosten"
    meta = result.meta
    assert (meta.keyword, meta.date, meta.provider, meta.model) == (KEYWORD, "2026-10-03", "fake", "fake-model")
    assert (meta.dimension, meta.target_words, meta.method, meta.prechunked) == (64, 15, "structure", False)
    assert (meta.covered_delta, meta.weak_delta) == (0.05, 0.15)
    assert len(result.statement_vectors) == 5


def test_run_without_sources_uses_quartiles_and_own_pages_only():
    result = sample_result(with_sources=False)
    assert result.match.has_sources is False
    assert result.source_chunks == [] and result.source_stats is None
    assert all(m.status in QUARTILE_LABELS for m in result.match.statements)
    assert all(p.origin == OWN for p in result.match.pages)


def test_all_sources_failed_behaves_like_no_sources_and_lists_them():
    failed = [FailedUrl("https://quelle.de/x", SOURCE, "HTTP 403")]
    prepared = prepare(KEYWORD, ANSWER, own_docs=OWN_DOCS, source_docs=[], failed=failed)
    result = analyze(prepared, FakeProvider(), None)
    assert result.match.has_sources is False
    assert result.meta.failed == tuple(failed)


def test_second_analysis_reuses_cached_embeddings():
    provider, cache = FakeProvider(), MemoryCache()
    prepared = prepare(KEYWORD, ANSWER, own_docs=OWN_DOCS, source_docs=SOURCE_DOCS)
    seen = []
    analyze(prepared, provider, cache, progress=lambda done, total: seen.append((done, total)))
    assert provider.calls == 7 and seen[-1] == (7, 7)
    analyze(prepared, provider, cache, weak_delta=0.3)
    assert provider.calls == 7


def test_rematch_applies_new_thresholds_without_embedding():
    result = sample_result()
    loose = rematch(result, covered_delta=2.0, weak_delta=2.0)
    assert {m.status for m in loose.match.statements} == {"abgedeckt"}
    assert (loose.meta.covered_delta, loose.meta.weak_delta) == (2.0, 2.0)
    assert by_text(result)[GAP].status == "Lücke"  # the original result is untouched


def prechunked(first, **overrides):
    data = chunks_to_csv(first.own_chunks, provider="fake", model="fake-model", target_words=15)
    chunks, target = read_chunks_file(io.BytesIO(data), "c.csv", provider="fake", model="fake-model")
    params = {"own_chunks": chunks, "stored_target": target, "source_docs": SOURCE_DOCS, **overrides}
    return prepare(KEYWORD, ANSWER, **params)


def test_reusing_exported_chunks_gives_the_same_results_and_embeds_less():
    first = sample_result()
    prepared = prechunked(first)
    assert prepared.prechunked is True and prepared.warnings == ()
    assert (prepared.own_stats.documents, prepared.own_stats.chunks) == (2, 3)
    provider = FakeProvider()
    second = analyze(prepared, provider, None)
    assert provider.calls == 5  # only the statements; the source chunks repeat two of them
    assert second.meta.prechunked is True
    for a, b in zip(first.match.statements, second.match.statements):
        assert (a.own.url, a.status) == (b.own.url, b.status)
        assert a.own.score == approx(b.own.score, abs=1e-4)
    for a, b in zip(first.match.pages, second.match.pages):
        assert a.url == b.url and a.page_score == approx(b.page_score, abs=1e-4)


def test_exported_chunks_with_another_dimension_are_rejected():
    prepared = prechunked(sample_result())
    with pytest.raises(ChunkFileError, match="Dimension 64.*Dimension 32"):
        analyze(prepared, FakeProvider(dim=32), None)


def test_stored_target_far_from_the_current_one_warns():
    first = sample_result()
    assert prechunked(first, stored_target=20).warnings == ()
    warnings = prechunked(first, stored_target=40).warnings
    assert len(warnings) == 1 and "Zielgröße 40" in warnings[0] and "15" in warnings[0]


def test_combine_documents_prefers_table_rows():
    table = [Document("https://me.de/a", "frog", OWN)]
    fetched = [Document("http://me.de/a/", "fetched", OWN), Document("https://me.de/b", "b", OWN)]
    assert [d.text for d in combine_documents(table, fetched)] == ["frog", "b"]


@pytest.mark.parametrize("keyword,answer,docs,message", [
    ("  ", ANSWER, OWN_DOCS, "Keyword"),
    (KEYWORD, " \n", OWN_DOCS, "AI-Antwort"),
    (KEYWORD, ANSWER, [], "Kein eigener Content"),
    (KEYWORD, ANSWER, [Document("https://me.de/leer", "  ", OWN)], "Kein eigener Content"),
])
def test_prepare_validates_inputs(keyword, answer, docs, message):
    with pytest.raises(ValueError, match=message):
        prepare(keyword, answer, own_docs=docs)


class FlakyProvider(FakeProvider):
    """Returns no vector for texts that mention the team, like a provider rejecting one input."""

    def embed(self, texts):
        return [None if "Team" in text else vector for text, vector in zip(texts, super().embed(texts))]


def test_chunks_without_embedding_are_dropped_with_a_warning():
    prepared = prepare(KEYWORD, ANSWER, own_docs=OWN_DOCS)
    result = analyze(prepared, FlakyProvider(), None)
    assert {c.url for c in result.own_chunks} == {"https://me.de/kosten"}
    assert any(w.startswith("1 Chunk") for w in result.meta.warnings)


def test_statement_without_embedding_aborts():
    class NoStatements(FakeProvider):
        def embed(self, texts):
            return [None if "KfW" in text else vector for text, vector in zip(texts, super().embed(texts))]

    with pytest.raises(ValueError, match="Aussage der AI-Antwort"):
        analyze(prepare(KEYWORD, ANSWER, own_docs=OWN_DOCS), NoStatements(), None)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_pipeline.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'aam.pipeline'`.

- [ ] **Step 3: Implement `aam/pipeline.py`**

```python
"""One run in two phases: prepare (segment and chunk, free) and analyze (embed and match)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import date
from typing import Callable, Optional, Sequence

from aam.chunk import chunk_documents, stats_for
from aam.embeddings import EmbeddingProvider, embed_cached
from aam.ingest.chunks_file import check_dimension
from aam.ingest.urls import url_key
from aam.match import match
from aam.models import Chunk, ChunkStats, Document, FailedUrl, Result, RunMeta, SegmentedAnswer
from aam.segment import Embed, segment_answer

MAX_TARGET_DEVIATION = 0.5


@dataclass
class Prepared:
    answer: SegmentedAnswer
    own_chunks: list
    source_chunks: list
    own_stats: ChunkStats
    source_stats: Optional[ChunkStats]
    failed: tuple
    warnings: tuple
    prechunked: bool

    @property
    def texts_to_embed(self) -> int:
        """Distinct texts the next analysis sends to the provider at most."""
        texts = {s.text for s in self.answer.statements}
        texts.update(c.text for c in self.own_chunks + self.source_chunks if c.vector is None)
        return len(texts)


def combine_documents(table_docs: Sequence[Document], fetched_docs: Sequence[Document]) -> list[Document]:
    """Documents from an uploaded table win over fetched ones for the same URL."""
    docs = list(table_docs)
    seen = {url_key(d.url) for d in docs}
    for doc in fetched_docs:
        key = url_key(doc.url)
        if key not in seen:
            seen.add(key)
            docs.append(doc)
    return docs


def prepare(keyword: str, answer_text: str, *, own_docs: Sequence[Document] = (),
            source_docs: Sequence[Document] = (), own_chunks: Optional[Sequence[Chunk]] = None,
            stored_target: Optional[int] = None, failed: Sequence[FailedUrl] = (),
            target_override: Optional[int] = None, embed: Optional[Embed] = None) -> Prepared:
    if not keyword.strip():
        raise ValueError("Bitte ein Keyword oder einen Prompt eintragen.")
    if not answer_text.strip():
        raise ValueError("Bitte die AI-Antwort einfügen.")
    answer = segment_answer(keyword, answer_text, target_override=target_override, embed=embed)
    warnings: list[str] = []
    if own_chunks is not None:
        chunks = list(own_chunks)
        own_stats = stats_for(chunks, documents=len({c.url for c in chunks}))
        target = answer.target_words
        if stored_target and abs(stored_target - target) / target > MAX_TARGET_DEVIATION:
            warnings.append(
                f"Die hochgeladenen Chunks wurden mit Zielgröße {stored_target} Wörter geschnitten, "
                f"die aktuelle AI-Antwort ergibt {target}. Die Scores sind dadurch nur eingeschränkt vergleichbar.")
    else:
        chunks, own_stats = chunk_documents(own_docs, answer.target_words)
    if not chunks:
        raise ValueError("Kein eigener Content mit Text vorhanden. Bitte Export oder URL-Liste prüfen.")
    source_chunks: list[Chunk] = []
    source_stats = None
    if source_docs:
        source_chunks, source_stats = chunk_documents(source_docs, answer.target_words)
    return Prepared(answer, chunks, source_chunks, own_stats, source_stats, tuple(failed), tuple(warnings),
                    own_chunks is not None)


def analyze(prepared: Prepared, provider: EmbeddingProvider, cache, *, covered_delta: float = 0.05,
            weak_delta: float = 0.15, progress: Optional[Callable[[int, int], None]] = None,
            today: Optional[date] = None) -> Result:
    statements = prepared.answer.statements
    pending = [c for c in prepared.own_chunks + prepared.source_chunks if c.vector is None]
    texts = [s.text for s in statements] + [c.text for c in pending]
    vectors = embed_cached(provider, texts, cache, progress=progress)
    statement_vectors = vectors[:len(statements)]
    if any(v is None for v in statement_vectors):
        raise ValueError("Mindestens eine Aussage der AI-Antwort konnte nicht eingebettet werden.")
    for chunk, vector in zip(pending, vectors[len(statements):]):
        chunk.vector = vector
    dimension = len(statement_vectors[0])
    if prepared.prechunked:
        check_dimension(prepared.own_chunks, dimension)

    own = [c for c in prepared.own_chunks if c.vector is not None]
    sources = [c for c in prepared.source_chunks if c.vector is not None]
    warnings = list(prepared.warnings)
    dropped = len(prepared.own_chunks) + len(prepared.source_chunks) - len(own) - len(sources)
    if dropped:
        if dropped == 1:
            warnings.append("1 Chunk konnte nicht eingebettet werden und fehlt in der Auswertung.")
        else:
            warnings.append(f"{dropped} Chunks konnten nicht eingebettet werden und fehlen in der Auswertung.")
    matched = match(prepared.answer, statement_vectors, own + sources,
                    covered_delta=covered_delta, weak_delta=weak_delta)
    meta = RunMeta(
        keyword=prepared.answer.keyword,
        date=(today or date.today()).isoformat(),
        provider=provider.name,
        model=provider.model,
        dimension=dimension,
        target_words=prepared.answer.target_words,
        median_words=prepared.answer.median_words,
        method=prepared.answer.method,
        covered_delta=covered_delta,
        weak_delta=weak_delta,
        failed=tuple(prepared.failed),
        warnings=tuple(warnings),
        prechunked=prepared.prechunked,
    )
    return Result(meta, matched, prepared.answer, own, sources, prepared.own_stats, prepared.source_stats,
                  list(statement_vectors))


def rematch(result: Result, *, covered_delta: float, weak_delta: float) -> Result:
    """New thresholds on an existing result; no embedding call."""
    matched = match(result.answer, result.statement_vectors, result.own_chunks + result.source_chunks,
                    covered_delta=covered_delta, weak_delta=weak_delta)
    meta = dataclasses.replace(result.meta, covered_delta=covered_delta, weak_delta=weak_delta)
    return dataclasses.replace(result, meta=meta, match=matched)
```

Append to `tests/fakes.py`:

```python


def sample_result(with_sources: bool = True):
    """A complete run over the sample data with the fake provider."""
    from datetime import date

    from aam.pipeline import analyze, prepare

    prepared = prepare("dachsanierung kosten", ANSWER, own_docs=OWN_DOCS,
                       source_docs=SOURCE_DOCS if with_sources else [])
    return analyze(prepared, FakeProvider(), None, today=date(2026, 10, 3))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_pipeline.py -q`
Expected: PASS.

- [ ] **Step 5: Run the whole suite and commit**

Run: `.venv/bin/pytest -q` — expected: PASS.

```bash
git add aam/pipeline.py tests/fakes.py tests/test_pipeline.py
git commit -m "Add pipeline with prepare, analyze and rematch

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Report (tables, summary, protocol, Excel)

**Files:**
- Create: `aam/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `aam.models.Result`, `SegmentedAnswer`, `ChunkStats`, `FailedUrl`, `OWN`, `SOURCE`; `aam.match.STATUS_COVERED`, `STATUS_GAP`; `tests.fakes.sample_result`.
- Produces:
  - `KEYWORD_COLUMN = "Keyword/Prompt"`, `SHEETS = ("Seiten", "Themen", "Aussagen", "Chunks", "Protokoll")`.
  - `pages_frame(result)`, `topics_frame(result)`, `statements_frame(result)`, `chunks_frame(result)`, `protocol_frame(result)` → `pandas.DataFrame`, each with `Keyword/Prompt` as first column.
  - `answer_frame(answer: SegmentedAnswer) -> DataFrame` with columns `Thema, Aussage, Wortzahl` (UI preview, no keyword column).
  - `protocol_lines(answer, own_stats, source_stats, failed, *, prechunked=False) -> list[str]`.
  - `summary(result) -> str`.
  - `to_excel(result) -> bytes`.

- [ ] **Step 1: Write the failing tests**

`tests/test_report.py`:

```python
import io

import pandas as pd

from aam.match import QUARTILE_LABELS
from aam.models import SOURCE, FailedUrl
from aam.report import (KEYWORD_COLUMN, SHEETS, answer_frame, chunks_frame, pages_frame, protocol_frame,
                        protocol_lines, statements_frame, summary, to_excel, topics_frame)
from tests.fakes import sample_result

KEYWORD = "dachsanierung kosten"


def test_excel_has_five_sheets_and_each_starts_with_the_keyword():
    sheets = pd.read_excel(io.BytesIO(to_excel(sample_result())), sheet_name=None)
    assert tuple(sheets) == SHEETS == ("Seiten", "Themen", "Aussagen", "Chunks", "Protokoll")
    for name, df in sheets.items():
        assert df.columns[0] == KEYWORD_COLUMN == "Keyword/Prompt", name
        assert set(df[KEYWORD_COLUMN]) == {KEYWORD}, name


def test_pages_frame_with_sources():
    df = pages_frame(sample_result())
    assert list(df.columns) == ["Keyword/Prompt", "Rang", "URL", "Herkunft", "Seitenscore", "Bester Chunk Score",
                                "Aussagen abgedeckt", "Aussagen gesamt"]
    assert list(df["Rang"]) == [1, 2, 3]
    assert set(df["Herkunft"]) == {"eigen", "Quelle"}
    assert set(df["Aussagen gesamt"]) == {5}
    assert all(round(v, 4) == v for v in df["Seitenscore"])


def test_topics_frame_with_sources():
    df = topics_frame(sample_result())
    assert list(df.columns) == ["Keyword/Prompt", "Thema", "Anzahl Aussagen", "Beste eigene URL",
                                "Bester eigener Chunk", "Score", "Beste Quell-URL", "Bester Quell-Chunk",
                                "Score Quelle", "Delta", "abgedeckt", "schwach", "Lücke"]
    assert list(df["Thema"]) == ["Einleitung", "Kosten im Überblick", "Förderung"]
    row = df[df["Thema"] == "Förderung"].iloc[0]
    assert (row["Anzahl Aussagen"], row["Lücke"], row["abgedeckt"]) == (2, 2, 0)
    assert row["Beste Quell-URL"] == "https://quelle.de/foerderung"


def test_statements_frame_with_sources():
    df = statements_frame(sample_result())
    assert list(df.columns) == ["Keyword/Prompt", "Thema", "Aussage (AI-Chunk)", "Wortzahl", "Beste eigene URL",
                                "Bester eigener Chunk", "Score", "Beste Quell-URL", "Bester Quell-Chunk",
                                "Score Quelle", "Delta", "Status"]
    assert len(df) == 5
    row = df[df["Aussage (AI-Chunk)"] == "Das BAFA zahlt Zuschüsse von bis zu 20 Prozent."].iloc[0]
    assert (row["Thema"], row["Wortzahl"], row["Status"]) == ("Förderung", 9, "Lücke")
    assert abs(row["Delta"] - (row["Score"] - row["Score Quelle"])) < 2e-4


def test_frames_without_sources_drop_the_source_columns():
    result = sample_result(with_sources=False)
    pages, topics, statements = pages_frame(result), topics_frame(result), statements_frame(result)
    assert "Aussagen mit bestem Chunk" in pages.columns and "Aussagen abgedeckt" not in pages.columns
    assert set(pages["Herkunft"]) == {"eigen"}
    for df in (topics, statements):
        assert not {"Beste Quell-URL", "Bester Quell-Chunk", "Score Quelle", "Delta"} & set(df.columns)
    assert list(topics.columns[-4:]) == list(QUARTILE_LABELS)
    assert statements.columns[-1] == "Status"


def test_chunks_frame_lists_answer_statements_own_and_source_chunks():
    df = chunks_frame(sample_result())
    assert list(df.columns) == ["Keyword/Prompt", "Herkunft", "URL", "Chunk-Index", "Text", "Wortzahl"]
    assert df["Herkunft"].value_counts().to_dict() == {"AI-Antwort": 5, "eigen": 3, "Quelle": 2}
    assert "embedding_0" not in df.columns


def test_protocol_lines_describe_answer_chunking_and_failures():
    result = sample_result()
    failed = [FailedUrl("https://quelle.de/x", SOURCE, "HTTP 403")]
    lines = protocol_lines(result.answer, result.own_stats, result.source_stats, failed)
    assert lines[0] == ("AI-Antwort: 3 Themen, 5 Aussagen, Median 12 Wörter → Zielgröße 15 Wörter "
                        "(Segmentierung: Struktur)")
    assert lines[1].startswith("Eigener Content: 2 Seiten → 3 Chunks, Median ")
    assert "2 Absätze zusammengelegt, 0 Absätze geteilt" in lines[1]
    assert lines[2].startswith("Quellen: 1 Seiten → 2 Chunks")
    assert lines[3] == "Nicht geladen (Quelle): https://quelle.de/x (HTTP 403)"


def test_protocol_lines_without_sources_and_with_prechunked_content():
    result = sample_result(with_sources=False)
    lines = protocol_lines(result.answer, result.own_stats, None, [], prechunked=True)
    assert lines[1].startswith("Eigener Content (aus Chunk-Embedding-Export): ")
    assert lines[2].startswith("Quellen: keine")


def test_protocol_frame_contains_the_run_settings():
    df = protocol_frame(sample_result())
    assert list(df.columns) == ["Keyword/Prompt", "Angabe", "Wert"]
    values = dict(zip(df["Angabe"], df["Wert"]))
    assert values["Datum"] == "2026-10-03"
    assert values["Embedding-Anbieter"] == "fake" and values["Modell"] == "fake-model"
    assert values["Dimension"] == 64 and values["Zielgröße (Wörter)"] == 15
    assert values["Segmentierung"] == "Struktur"
    assert values["Schwelle abgedeckt"] == 0.05 and values["Schwelle schwach"] == 0.15
    assert any(a == "Chunking" for a in df["Angabe"])


def test_summary_with_sources_names_top_page_and_gap_topics():
    text = summary(sample_result())
    assert text.startswith("Relevanteste eigene Seite: https://me.de/kosten (Seitenscore 0,")
    assert "von 3)" in text
    assert "Lücken: 2 Aussagen im Thema „Förderung“" in text


def test_summary_without_sources_explains_the_missing_benchmark():
    text = summary(sample_result(with_sources=False))
    assert text.startswith("Relevanteste eigene Seite: https://me.de/kosten")
    assert "keinen Benchmark" in text


def test_answer_frame_shows_every_ai_chunk():
    df = answer_frame(sample_result().answer)
    assert list(df.columns) == ["Thema", "Aussage", "Wortzahl"]
    assert len(df) == 5 and df.iloc[3]["Thema"] == "Förderung" and df.iloc[3]["Wortzahl"] == 8
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_report.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'aam.report'`.

- [ ] **Step 3: Implement `aam/report.py`**

```python
"""Turn a Result into tables, a short summary, the chunking protocol and an Excel file."""

from __future__ import annotations

import io
from typing import Optional, Sequence

import pandas as pd

from aam.match import STATUS_COVERED, STATUS_GAP
from aam.models import OWN, SOURCE, ChunkStats, FailedUrl, Result, SegmentedAnswer

KEYWORD_COLUMN = "Keyword/Prompt"
SHEETS = ("Seiten", "Themen", "Aussagen", "Chunks", "Protokoll")
ORIGIN_LABELS = {OWN: "eigen", SOURCE: "Quelle"}
METHOD_LABELS = {"structure": "Struktur", "semantic": "semantisch"}
_SOURCE_COLUMNS = ["Beste Quell-URL", "Bester Quell-Chunk", "Score Quelle", "Delta"]


def _frame(rows: list, columns: list[str], keyword: str) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=columns)
    df.insert(0, KEYWORD_COLUMN, keyword)
    return df


def _score(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value), 4)


def _de(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def pages_frame(result: Result) -> pd.DataFrame:
    covered = "Aussagen abgedeckt" if result.match.has_sources else "Aussagen mit bestem Chunk"
    rows = [(p.rank, p.url, ORIGIN_LABELS[p.origin], _score(p.page_score), _score(p.best_chunk_score),
             p.covered, p.total) for p in result.match.pages]
    columns = ["Rang", "URL", "Herkunft", "Seitenscore", "Bester Chunk Score", covered, "Aussagen gesamt"]
    return _frame(rows, columns, result.meta.keyword)


def _hit_cells(match_row, has_sources: bool) -> list:
    cells = [match_row.own.url, match_row.own.text, _score(match_row.own.score)]
    if has_sources:
        cells += [match_row.source.url, match_row.source.text, _score(match_row.source.score),
                  _score(match_row.delta)]
    return cells


def topics_frame(result: Result) -> pd.DataFrame:
    has_sources = result.match.has_sources
    labels = list(result.match.status_labels)
    rows = [[t.topic.label, len(t.topic.statements)] + _hit_cells(t, has_sources)
            + [t.counts.get(label, 0) for label in labels] for t in result.match.topics]
    columns = (["Thema", "Anzahl Aussagen", "Beste eigene URL", "Bester eigener Chunk", "Score"]
               + (_SOURCE_COLUMNS if has_sources else []) + labels)
    return _frame(rows, columns, result.meta.keyword)


def statements_frame(result: Result) -> pd.DataFrame:
    has_sources = result.match.has_sources
    rows = [[m.topic_label, m.statement.text, m.statement.words] + _hit_cells(m, has_sources) + [m.status]
            for m in result.match.statements]
    columns = (["Thema", "Aussage (AI-Chunk)", "Wortzahl", "Beste eigene URL", "Bester eigener Chunk", "Score"]
               + (_SOURCE_COLUMNS if has_sources else []) + ["Status"])
    return _frame(rows, columns, result.meta.keyword)


def chunks_frame(result: Result) -> pd.DataFrame:
    rows = [("AI-Antwort", "", s.id, s.text, s.words) for s in result.answer.statements]
    rows += [(ORIGIN_LABELS[c.origin], c.url, c.index, c.text, c.words)
             for c in list(result.own_chunks) + list(result.source_chunks)]
    return _frame(rows, ["Herkunft", "URL", "Chunk-Index", "Text", "Wortzahl"], result.meta.keyword)


def answer_frame(answer: SegmentedAnswer) -> pd.DataFrame:
    """Every AI chunk with its topic, shown before any embedding happens."""
    rows = [(topic.label, s.text, s.words) for topic in answer.topics for s in topic.statements]
    return pd.DataFrame(rows, columns=["Thema", "Aussage", "Wortzahl"])


def _stats_line(label: str, stats: ChunkStats) -> str:
    line = (f"{label}: {stats.documents} Seiten → {stats.chunks} Chunks, Median {stats.median_words:.0f} Wörter "
            f"(p10: {stats.p10_words:.0f}, p90: {stats.p90_words:.0f}), "
            f"{stats.merged} Absätze zusammengelegt, {stats.split} Absätze geteilt")
    if stats.empty_urls:
        line += f", {len(stats.empty_urls)} Seiten ohne Text"
    return line


def protocol_lines(answer: SegmentedAnswer, own_stats: ChunkStats, source_stats: Optional[ChunkStats],
                   failed: Sequence[FailedUrl], *, prechunked: bool = False) -> list[str]:
    lines = [
        f"AI-Antwort: {len(answer.topics)} Themen, {len(answer.statements)} Aussagen, "
        f"Median {answer.median_words:.0f} Wörter → Zielgröße {answer.target_words} Wörter "
        f"(Segmentierung: {METHOD_LABELS[answer.method]})",
        _stats_line("Eigener Content (aus Chunk-Embedding-Export)" if prechunked else "Eigener Content", own_stats),
    ]
    if source_stats is None:
        lines.append("Quellen: keine. Der Status zeigt die relative Stärke innerhalb des eigenen Contents.")
    else:
        lines.append(_stats_line("Quellen", source_stats))
    lines += [f"Nicht geladen ({ORIGIN_LABELS[f.origin]}): {f.url} ({f.reason})" for f in failed]
    return lines


def protocol_frame(result: Result) -> pd.DataFrame:
    meta = result.meta
    rows = [
        ("Datum", meta.date),
        ("Embedding-Anbieter", meta.provider),
        ("Modell", meta.model),
        ("Dimension", meta.dimension),
        ("Zielgröße (Wörter)", meta.target_words),
        ("Median Aussagenlänge (Wörter)", meta.median_words),
        ("Segmentierung", METHOD_LABELS[meta.method]),
        ("Schwelle abgedeckt", meta.covered_delta),
        ("Schwelle schwach", meta.weak_delta),
    ]
    rows += [("Chunking", line) for line in protocol_lines(
        result.answer, result.own_stats, result.source_stats, meta.failed, prechunked=meta.prechunked)]
    rows += [("Hinweis", warning) for warning in meta.warnings]
    return _frame(rows, ["Angabe", "Wert"], meta.keyword)


def summary(result: Result) -> str:
    matched = result.match
    top = next(p for p in matched.pages if p.origin == OWN)
    text = (f"Relevanteste eigene Seite: {top.url} (Seitenscore {_de(top.page_score)}, "
            f"Rang {top.rank} von {len(matched.pages)}).")
    if not matched.has_sources:
        return (f"{text} Sie stellt bei {top.covered} von {top.total} Aussagen den besten Chunk der Domain. "
                "Ohne Quellen gibt es keinen Benchmark: der Status zeigt nur die relative Stärke "
                "innerhalb des eigenen Contents.")
    covered = sum(m.status == STATUS_COVERED for m in matched.statements)
    text += (f" Über die ganze Domain sind {covered} von {top.total} Aussagen abgedeckt, "
             f"diese Seite allein deckt {top.covered} ab.")
    gaps = [(t.topic.label, t.counts[STATUS_GAP]) for t in matched.topics if t.counts.get(STATUS_GAP)]
    if not gaps:
        return f"{text} Keine Lücken."
    parts = [f"{count} {'Aussage' if count == 1 else 'Aussagen'} im Thema „{label}“" for label, count in gaps]
    return f"{text} Lücken: {', '.join(parts)}."


def to_excel(result: Result) -> bytes:
    frames = (pages_frame(result), topics_frame(result), statements_frame(result), chunks_frame(result),
              protocol_frame(result))
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in zip(SHEETS, frames):
            frame.to_excel(writer, sheet_name=name, index=False)
    return buffer.getvalue()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_report.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aam/report.py tests/test_report.py
git commit -m "Add report tables, summary, protocol and Excel export

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Streamlit app, examples, README

**Files:**
- Create: `app.py`, `examples/beispiel-ai-antwort.txt`, `examples/beispiel-content.csv`, `examples/beispiel-quellen.csv`, `README.md`
- Test: `tests/test_app.py`, `tests/test_examples.py`

**Interfaces:**
- Consumes: `aam.embeddings` (`PROVIDERS`, `DEFAULT_EMBED_MODELS`, `make_provider`, `embed_cached`, `EmbeddingError`), `aam.extract.fetch_documents`, `aam.pipeline` (`prepare`, `analyze`, `rematch`, `combine_documents`), `aam.report` (all frame functions, `protocol_lines`, `summary`, `to_excel`), `aam.ingest.chunks_file` (`is_chunks_table`, `chunks_from_frame`, `chunks_to_csv`), `aam.ingest.content.documents_from_frame`, `aam.ingest.tables` (`read_table_info`, `describe_table_info`), `aam.ingest.urls` (`parse_urls`, `url_key`), `aam.cache.MemoryCache`, `aam.envfile.load_env_file`.
- Produces: the app. Widget keys used by tests: `keyword`, `answer`, `source_urls`, `own_upload`, `own_urls`, `source_upload`, `provider`, `target_override`, `covered_delta`, `weak_delta`, `download_excel`, `download_chunks`. Session keys: `prepared`, `prepared_sig`, `result`, `result_sig`, `cache`, `fetched`. Button labels: `Segmentieren und chunken`, `Analyse starten`.

The app must call `embeddings.make_provider(...)` and `extract.fetch_documents(...)` through their modules (not via `from ... import`), because the tests replace these two attributes.

- [ ] **Step 1: Create the example files**

`examples/beispiel-ai-antwort.txt`:

```text
Eine Dachsanierung kostet je nach Umfang zwischen 100 und 300 Euro pro Quadratmeter. Entscheidend sind Material, Dachform und der Zustand des Dachstuhls.

Kosten im Überblick
Eine Neueindeckung ohne Dämmung liegt bei 100 bis 150 Euro pro Quadratmeter.
Mit Dämmung steigen die Kosten auf 200 bis 300 Euro pro Quadratmeter.
Für das Gerüst kommen etwa 10 bis 15 Euro pro Quadratmeter Fassadenfläche hinzu.

Förderung
Die KfW fördert die Dachdämmung mit zinsgünstigen Krediten.
Das BAFA zahlt Zuschüsse von bis zu 20 Prozent der förderfähigen Kosten.
Voraussetzung ist, dass ein Energieberater die Maßnahme begleitet.

Dauer
Die Arbeiten dauern bei einem Einfamilienhaus meist ein bis zwei Wochen.
Bei schlechtem Wetter oder Schäden am Dachstuhl verlängert sich die Bauzeit.
```

`examples/beispiel-content.csv`:

```text
Address,Extract Main Content 1
https://www.beispiel-dachdecker.de/ratgeber/dachsanierung-kosten,"Dachsanierung: Mit diesen Kosten müssen Sie rechnen
Eine Neueindeckung ohne Dämmung kostet bei uns zwischen 100 und 150 Euro pro Quadratmeter Dachfläche.
Wird das Dach zusätzlich gedämmt, steigen die Kosten auf 200 bis 300 Euro pro Quadratmeter.
Der genaue Preis hängt vom gewählten Material, der Dachform und dem Zustand des Dachstuhls ab. Tonziegel sind teurer als Betondachsteine, halten aber länger.
Wie lange dauert eine Dachsanierung?
Bei einem Einfamilienhaus dauern die Arbeiten in der Regel ein bis zwei Wochen. Schlechtes Wetter kann die Bauzeit verlängern."
https://www.beispiel-dachdecker.de/ratgeber/dachdaemmung,"Dachdämmung: Aufsparren, Zwischensparren oder Untersparren
Die Aufsparrendämmung wird von außen auf den Sparren verlegt und vermeidet Wärmebrücken.
Die Zwischensparrendämmung ist günstiger, weil das Dach dafür nicht neu eingedeckt werden muss.
Welche Dämmung sich eignet, hängt vom Zustand des Dachs und vom geplanten Ausbau ab."
https://www.beispiel-dachdecker.de/ueber-uns,"Über uns
Wir sind ein Dachdeckerbetrieb aus Köln und seit 20 Jahren für private und gewerbliche Kunden tätig.
Unser Team besteht aus zwölf Gesellen und zwei Meistern. Wir bilden regelmäßig aus."
```

`examples/beispiel-quellen.csv`:

```text
Address,Extract Main Content 1
https://www.beispiel-energieportal.de/foerderung-dach,"Förderung für die Dachsanierung
Die KfW fördert die Dachdämmung über zinsgünstige Kredite im Programm für energetische Sanierung.
Das BAFA zahlt einen Zuschuss von bis zu 20 Prozent der förderfähigen Kosten.
Voraussetzung für beide Programme ist, dass ein Energieberater die Maßnahme plant und begleitet."
https://www.beispiel-bauportal.de/dach/kosten,"Was kostet ein neues Dach?
Eine Dachsanierung kostet je nach Umfang zwischen 100 und 300 Euro pro Quadratmeter.
Für das Gerüst kommen etwa 10 bis 15 Euro pro Quadratmeter Fassadenfläche hinzu.
Bei Schäden am Dachstuhl oder schlechtem Wetter verlängert sich die Bauzeit deutlich."
```

- [ ] **Step 2: Write the failing tests**

`tests/test_examples.py`:

```python
from pathlib import Path

from aam.ingest.content import read_content_table
from aam.models import SOURCE
from aam.pipeline import analyze, prepare
from aam.report import to_excel
from tests.fakes import FakeProvider

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_examples_run_through_the_whole_pipeline():
    answer = (EXAMPLES / "beispiel-ai-antwort.txt").read_text(encoding="utf-8")
    own, own_info = read_content_table(EXAMPLES / "beispiel-content.csv")
    sources, _ = read_content_table(EXAMPLES / "beispiel-quellen.csv", origin=SOURCE)
    assert own_info.documents == 3 and len(sources) == 2
    prepared = prepare("dachsanierung kosten", answer, own_docs=own, source_docs=sources)
    assert [t.label for t in prepared.answer.topics] == ["Einleitung", "Kosten im Überblick", "Förderung", "Dauer"]
    result = analyze(prepared, FakeProvider(), None)
    assert result.match.has_sources is True
    statuses = {m.statement.text: m.status for m in result.match.statements}
    assert statuses["Die KfW fördert die Dachdämmung mit zinsgünstigen Krediten."] == "Lücke"
    assert len(to_excel(result)) > 0
```

`tests/test_app.py`:

```python
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from aam.embeddings import EmbeddingError
from aam.ingest.chunks_file import chunks_to_csv
from aam.models import OWN, SOURCE, FailedUrl
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


def test_chunking_step_shows_protocol_ai_chunks_and_embedding_count(backend):
    at = chunked(source_file=True)
    assert not at.exception and not at.error
    protocol = at.code[0].value
    assert "AI-Antwort: 3 Themen, 5 Aussagen" in protocol and "Zielgröße 15 Wörter" in protocol
    assert "Eigener Content: 2 Seiten → 3 Chunks" in protocol
    assert "Quellen: 1 Seiten → 2 Chunks" in protocol
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


def test_threshold_change_rematches_without_new_embeddings(backend):
    at = analyzed(source_file=True)
    calls = backend.calls
    at.sidebar.slider(key="weak_delta").set_value(0.5).run()
    assert not at.exception
    assert at.session_state["result"].meta.weak_delta == 0.5
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
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_app.py tests/test_examples.py -q`
Expected: `tests/test_examples.py` passes already (it only needs the example files); `tests/test_app.py` FAILS because `app.py` does not exist.

- [ ] **Step 4: Implement `app.py`**

```python
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
from aam.models import OWN, SOURCE

UPLOAD_TYPES = ["csv", "tsv", "txt", "xlsx", "xlsm"]
KEY_ENV = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}
RUN_ERRORS = (ValueError, embeddings.EmbeddingError)
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

st.set_page_config(page_title="AI Answer Matcher", layout="wide")
load_env_file(Path(__file__).resolve().parent / ".env")
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
    signature = (file.name, file.size, provider_name, model)
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
            parsed["note"] = (f"{describe_table_info(info)} · URL-Spalte „{found.url_column}“ · "
                              f"Content-Spalte „{found.content_column}“ · {found.documents} Seiten")
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
        with httpx.Client() as client:
            docs, failed = extract.fetch_documents(
                todo, origin, client,
                progress=lambda done, total: bar.progress(done / total, text=f"Lade URL {done} von {total} …"))
        bar.empty()
        for doc in docs:
            state["fetched"][(origin, url_key(doc.url))] = doc
    found = [state["fetched"][(origin, url_key(u))] for u in wanted if (origin, url_key(u)) in state["fetched"]]
    return found, list(failed)


def cached_bytes(slot: str, key, build):
    """Build a download once per result instead of on every rerun."""
    entry = state.get(slot)
    if not entry or entry[0] != key:
        entry = (key, build())
        state[slot] = entry
    return entry[1]


def show_result(result) -> None:
    st.subheader("Ergebnis")
    st.markdown(report.summary(result))
    for warning in result.meta.warnings:
        st.warning(warning)
    tab_pages, tab_topics, tab_statements = st.tabs(["Seiten", "Themen", "Aussagen"])
    with tab_pages:
        st.caption("Ganze AI-Antwort gegen ganze Seite. Eigene Seiten und Quellen in einem Ranking.")
        st.dataframe(report.pages_frame(result), width="stretch", hide_index=True)
    with tab_topics:
        st.caption("Bester Chunk der eigenen Domain pro Themengebiet der AI-Antwort.")
        st.dataframe(report.topics_frame(result), width="stretch", hide_index=True)
    with tab_statements:
        st.caption("Bester Chunk der eigenen Domain pro einzelner Aussage (AI-Chunk).")
        st.dataframe(report.statements_frame(result), width="stretch", hide_index=True)

    slug = re.sub(r"[^a-z0-9]+", "-", result.meta.keyword.lower()).strip("-")[:40] or "lauf"
    signature = state["result_sig"]
    excel = cached_bytes("excel", (signature, result.meta.covered_delta, result.meta.weak_delta),
                         lambda: report.to_excel(result))
    left, right = st.columns(2)
    left.download_button("Excel herunterladen", data=excel, mime=XLSX_MIME, key="download_excel",
                         file_name=f"ai-answer-matcher_{slug}_{result.meta.date}.xlsx")
    if not result.meta.prechunked:
        chunks_csv = cached_bytes("chunks_csv", signature, lambda: chunks_file.chunks_to_csv(
            result.own_chunks, provider=provider_name, model=model, target_words=result.meta.target_words))
        right.download_button(
            "Chunk-Embeddings herunterladen", data=chunks_csv, mime="text/csv", key="download_chunks",
            file_name=f"chunks_embeddings_{result.meta.date}.csv",
            help="Beim nächsten Keyword statt des Frog-Exports hochladen. "
                 "Dann wird nur noch die AI-Antwort eingebettet.")


# ---------------------------------------------------------------- page

st.title("AI Answer Matcher")
st.caption("Wie nah ist dein Content an der Antwort von AI Overviews oder AI Mode? "
           "Vergleich per Embeddings und Kosinus-Ähnlichkeit, auf Seiten- und auf Passagen-Ebene.")

st.header("1. AI-Antwort")
keyword = st.text_input("Keyword oder Prompt", key="keyword", placeholder="z. B. dachsanierung kosten")
answer_text = st.text_area("AI-Antwort (Text aus AI Overview oder AI Mode einfügen)", key="answer", height=260)
source_urls = parse_urls(st.text_area(
    "Quell-URLs der AI-Antwort (optional, eine pro Zeile)", key="source_urls", height=110,
    help="Die zitierten Quellen dienen als Benchmark. Ohne Quellen läuft die Analyse trotzdem."))

st.header("2. Eigener Content")
own_file = st.file_uploader(
    "Screaming-Frog-Export mit URL und Main Content, oder Chunk-Embedding-Export aus einem früheren Lauf",
    type=UPLOAD_TYPES, key="own_upload")
own_parsed = parse_upload(own_file, OWN, "own_parsed")
show_upload_note(own_parsed)
own_urls = parse_urls(st.text_area(
    "Eigene URLs (optional, eine pro Zeile). Das Tool lädt den Content selbst.", key="own_urls", height=110))
source_file = st.file_uploader(
    "Optional: Content der Quell-URLs als Datei (gleiches Format wie der Frog-Export)",
    type=UPLOAD_TYPES, key="source_upload")
source_parsed = parse_upload(source_file, SOURCE, "source_parsed")
show_upload_note(source_parsed)

prechunked = bool(own_parsed and own_parsed["chunks"] is not None)
if prechunked and own_urls:
    st.info("Mit einem Chunk-Embedding-Export werden die eigenen URLs nicht zusätzlich geladen.")
upload_error = any(p is not None and p["error"] for p in (own_parsed, source_parsed))
has_own = prechunked or bool(own_parsed and own_parsed["docs"]) or bool(own_urls)
ready = bool(keyword.strip() and answer_text.strip() and has_own and not upload_error)


def input_signature() -> tuple:
    """Everything the chunking depends on; a change makes the prepared state stale."""
    return (keyword.strip(), answer_text.strip(), tuple(source_urls), tuple(own_urls),
            own_parsed["sig"] if own_parsed else None, source_parsed["sig"] if source_parsed else None,
            int(target_override), provider_name, model)


st.header("3. Chunking prüfen")
if not ready:
    st.caption("Es fehlen noch Keyword, AI-Antwort oder eigener Content.")
if st.button("Segmentieren und chunken", disabled=not ready, type="primary"):
    try:
        table_own = own_parsed["docs"] if own_parsed else []
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
        )
        state["prepared_sig"] = input_signature()
        state.pop("result", None)
    except RUN_ERRORS as exc:
        state.pop("prepared", None)
        st.error(str(exc))

prepared = state.get("prepared")
fresh = prepared is not None and state.get("prepared_sig") == input_signature()
if prepared is not None and not fresh:
    st.warning("Die Eingaben haben sich seit dem letzten Chunking geändert. "
               "Bitte erneut segmentieren und chunken.")
if fresh:
    st.code("\n".join(report.protocol_lines(
        prepared.answer, prepared.own_stats, prepared.source_stats, prepared.failed,
        prechunked=prepared.prechunked)), language=None)
    for warning in prepared.warnings:
        st.warning(warning)
    st.markdown("**AI-Chunks:** so wurde die Antwort in Themen und Aussagen zerlegt")
    st.dataframe(report.answer_frame(prepared.answer), width="stretch", hide_index=True)
    with st.expander("Beispiel-Chunks aus dem eigenen Content"):
        for chunk in prepared.own_chunks[:3]:
            st.markdown(f"**{chunk.url}** · Chunk {chunk.index} · {chunk.words} Wörter")
            st.write(chunk.text)
    st.caption(f"Im nächsten Schritt werden bis zu {prepared.texts_to_embed} Texte eingebettet.")

st.header("4. Analysieren")
if st.button("Analyse starten", disabled=not fresh, type="primary"):
    try:
        provider = make_provider()
        bar = st.progress(0.0, text="Embeddings werden erstellt …")
        state["result"] = pipeline.analyze(
            prepared, provider, state["cache"], covered_delta=covered_delta, weak_delta=weak_delta,
            progress=lambda done, total: bar.progress(done / max(total, 1),
                                                      text=f"Embeddings: {done} von {total} Texten"))
        bar.empty()
        state["result_sig"] = state["prepared_sig"]
    except RUN_ERRORS as exc:
        state.pop("result", None)
        st.error(f"Analyse fehlgeschlagen: {exc}")

result = state.get("result")
if result is not None and fresh and state.get("result_sig") == state.get("prepared_sig"):
    if (result.meta.covered_delta, result.meta.weak_delta) != (covered_delta, weak_delta):
        result = state["result"] = pipeline.rematch(result, covered_delta=covered_delta, weak_delta=weak_delta)
    show_result(result)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_app.py tests/test_examples.py -q`
Expected: PASS.

Notes for debugging: `AppTest` reruns the whole script on every `.run()`. If an element lookup fails, print `[e.value for e in at.error]` and `at.exception` first.

- [ ] **Step 6: Write `README.md`**

````markdown
# AI Answer Matcher

Wie nah ist dein Content an der Antwort, die AI Overviews oder AI Mode zu einem Keyword geben?

Das Tool vergleicht eine eingefügte AI-Antwort per Embeddings und Kosinus-Ähnlichkeit mit deinem Content. Auf zwei Ebenen:

- **Seite gegen Antwort:** Welche Seite ist insgesamt am relevantesten für die ganze AI-Antwort?
- **Chunk gegen Chunk:** Welche Passage deiner Domain passt am besten zu jedem Themengebiet und zu jeder einzelnen Aussage der AI-Antwort?

Die von der AI zitierten Quellen dienen optional als Benchmark. Dann siehst du pro Aussage, wie weit dein bester Absatz hinter dem besten Absatz der Quellen liegt.

## Die Idee dahinter

Klassisches Ranking ist ein Wettbewerb zwischen Seiten. AI Overviews und AI Mode sind ein Wettbewerb zwischen Passagen: Die Antwort wird aus einzelnen Aussagen zusammengesetzt, und jede Aussage stammt aus einer Passage irgendeiner Quelle. Deine Seite muss also nicht als Ganzes die beste sein. Sie muss für einen Teilaspekt die beste Passage liefern.

Deshalb zerlegt das Tool die AI-Antwort in Themen und Aussagen, leitet daraus die Chunk-Größe ab und schneidet deinen Content in Passagen derselben Größenordnung.

## Was du brauchst

- Python 3.11 oder neuer
- Einen API-Key für OpenAI oder Gemini, oder ein lokales [Ollama](https://ollama.com) mit einem Embedding-Modell
- Deinen Content als Screaming-Frog-Export mit URL und Main Content, oder eine Liste von URLs

## Einrichtung

Mit uv (empfohlen, holt Python 3.11 selbst):

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/streamlit run app.py
```

Mit vorhandenem Python 3.11:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Den API-Key trägst du in der Seitenleiste ein. Alternativ legst du neben `app.py` eine Datei `.env` an:

```text
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
```

## Ablauf

1. **AI-Antwort:** Keyword oder Prompt eintragen, den Antworttext aus AI Overview oder AI Mode einfügen. Optional die zitierten Quell-URLs, eine pro Zeile.
2. **Eigener Content:** Screaming-Frog-Export hochladen (Spalten für URL und Main Content, CSV oder Excel) oder eigene URLs eintragen, die das Tool selbst lädt. Beides lässt sich kombinieren, bei gleicher URL gewinnt der Export.
3. **Chunking prüfen:** Das Tool zeigt, wie es die AI-Antwort in Themen und Aussagen zerlegt hat, welche Chunk-Größe daraus folgt und wie viele Chunks aus deinem Content entstanden sind. Bis hierhin kostet nichts.
4. **Analysieren:** Embeddings werden erstellt, danach erscheinen Zusammenfassung, drei Tabellen und die Downloads.

Zum Ausprobieren liegen Beispieldateien in `examples/`.

## Eingaben im Detail

**Screaming-Frog-Export.** Erkannt werden URL-Spalten mit den Namen `Address`, `URL`, `Adresse`, `Page`, `Seite` und Content-Spalten wie `Extract Main Content 1`, `Main Content`, `Content`, `Text`, `Seiteninhalt`. Wichtig: Der Extraktor sollte Absätze als Zeilenumbrüche liefern, denn daran schneidet das Tool die Chunks.

**Quell-URLs.** Werden per HTTP geladen, der Main Content kommt aus trafilatura. Seiten, die sich nicht laden lassen, stehen im Protokoll. Für solche Fälle kannst du den Content der Quellen auch als Datei im selben Format hochladen.

**Ohne Quell-URLs** läuft die Analyse ebenfalls. Der Status zeigt dann nur die relative Stärke innerhalb deines eigenen Contents.

## Ergebnisse

| Tabelle | Inhalt |
|---|---|
| Seiten | Ranking aller Seiten (eigene und Quellen) nach Ähnlichkeit zur ganzen AI-Antwort |
| Themen | Bester Chunk deiner Domain pro Themengebiet, im Vergleich zum besten Quell-Chunk |
| Aussagen | Bester Chunk deiner Domain pro Aussage, mit Abstand zur besten Quelle und Status |

Status mit Quellen: **abgedeckt** (dein Chunk liegt höchstens 0,05 unter dem besten Quell-Chunk), **schwach** (bis 0,15 darunter), **Lücke** (mehr als 0,15 darunter). Beide Schwellen sind in der Seitenleiste einstellbar.

Das Excel enthält zusätzlich alle Chunks und ein Protokoll mit Datum, Modell, Chunk-Größe und nicht geladenen URLs. In jedem Blatt steht das Keyword in der ersten Spalte.

## Domain einmal einbetten, für viele Keywords nutzen

Nach einer Analyse kannst du die **Chunk-Embeddings herunterladen**. Beim nächsten Keyword lädst du diese Datei statt des Frog-Exports hoch. Dann wird nur noch die AI-Antwort (und ggf. die Quellen) eingebettet. Anbieter und Modell müssen dieselben sein wie beim Export.

## Was die Zahlen bedeuten und was nicht

- **Ähnlichkeit ist notwendig, nicht hinreichend.** Ein hoher Score heißt nicht, dass Google dich zitiert. Rankings, Crawlbarkeit und Autorität entscheiden mit.
- **Nicht auf den Score optimieren.** Die AI-Antwort ist aus den Quellen entstanden, die Quellen scoren deshalb zwangsläufig hoch. Wer die Antwort nachschreibt, liefert keinen Grund, zitiert zu werden. Nützlich sind die Lücken: Aussagen, zu denen dir eine eigenständige Passage fehlt.
- **Momentaufnahme.** AI-Antworten ändern sich täglich und je nach Nutzer. Das Datum steht im Protokoll.
- **Absolute Werte sind modellabhängig.** Vergleiche nur Läufe mit demselben Embedding-Modell.

## Tests

```bash
uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/pytest -q
```

Die Tests laufen ohne Netz und ohne API-Key.

## Geplant

Abruf von AI Overview und AI Mode über SerpApi und eine Validierungsstudie über mehrere Keywords (zitierte gegen nicht zitierte Seiten).
````

- [ ] **Step 7: Run the whole suite and commit**

Run: `.venv/bin/pytest -q` — expected: PASS.

```bash
git add app.py examples README.md tests/test_app.py tests/test_examples.py
git commit -m "Add Streamlit app, example data and README

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: End-to-end check

**Files:** none (verification only; fix what it uncovers in the owning module with a test).

- [ ] **Step 1: Full test run**

Run: `.venv/bin/pytest -q`
Expected: PASS, zero failures, zero errors, no network access.

- [ ] **Step 2: Confirm the package stays free of Streamlit**

Run: `grep -rn "streamlit" aam/ || echo "clean"`
Expected: `clean`.

- [ ] **Step 3: Start the app once**

```bash
.venv/bin/streamlit run app.py --server.headless true --server.port 8765 &
sleep 6
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8765/
kill %1
```

Expected: `200`.

- [ ] **Step 4: Commit any fixes**

```bash
git status --short
git add -A && git commit -m "Fix issues found in the end-to-end check

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

Skip the commit when nothing changed.
