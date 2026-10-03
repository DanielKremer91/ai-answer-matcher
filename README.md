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
3. **Chunking prüfen:** Das Tool zeigt, wie es die AI-Antwort in Themen und Aussagen zerlegt hat, welche Chunk-Größe daraus folgt und wie viele Chunks aus deinem Content entstanden sind. Bis hierhin entstehen keine Embedding-Kosten, außer die Antwort hat keine erkennbare Struktur: dann werden ihre Sätze schon hier eingebettet, um Themengrenzen zu finden.
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
