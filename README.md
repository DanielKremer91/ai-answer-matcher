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

Unter Windows (PowerShell):

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\streamlit run app.py
```

Mit vorhandenem Python 3.11:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Unter Windows:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
.venv\Scripts\streamlit run app.py
```

Die App ist nur vom eigenen Rechner aus erreichbar (`localhost`, siehe `.streamlit/config.toml`), nicht aus dem Netzwerk.

Den API-Key trägst du in der Seitenleiste ein. Alternativ legst du neben `app.py` eine Datei `.env` an:

```text
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
```

**Ollama (lokal, ohne API-Key).** Das Modell einmal laden und in der Seitenleiste den Anbieter „ollama“ wählen. Als Modellname steht dort `bge-m3` voreingestellt, ein mehrsprachiges Modell, das kein Aufgaben-Präfix braucht:

```bash
ollama pull bge-m3
```

## Ablauf

1. **AI-Antwort:** Keyword oder Prompt eintragen, den Antworttext aus AI Overview oder AI Mode einfügen. Optional die zitierten Quell-URLs, eine pro Zeile.
2. **Eigener Content:** Screaming-Frog-Export hochladen (Spalten für URL und Main Content, CSV oder Excel) oder eigene URLs eintragen, die das Tool selbst lädt. Beides lässt sich kombinieren, bei gleicher URL gewinnt der Export.
3. **Chunking prüfen:** Das Tool zeigt, wie es die AI-Antwort in Themen und Aussagen zerlegt hat, welche Chunk-Größe daraus folgt und wie viele Chunks aus deinem Content entstanden sind. Bis hierhin entstehen keine Embedding-Kosten, außer die Antwort hat keine erkennbare Struktur: dann werden ihre Sätze schon hier eingebettet, um Themengrenzen zu finden.
4. **Analysieren:** Embeddings werden erstellt, danach erscheinen Zusammenfassung, drei Tabellen und die Downloads.

Zum Ausprobieren liegen Beispieldateien in `examples/`.

## Screaming-Frog-Export erstellen

1. In Screaming Frog unter *Configuration → Custom → Custom Extraction* (XPath oder CSS-Selektor) oder *Custom JavaScript* einen Extraktor anlegen, der den Hauptinhalt der Seite liefert, ohne Navigation, Footer und Seitenleisten. Der Extraktor soll jeden Absatz als eigene Zeile ausgeben, denn daran schneidet das Tool die Chunks.
2. Den Extraktor so benennen, dass die Spalte im Export „Extract Main Content 1“ heißt oder „Main Content“ enthält.
3. Die Domain crawlen und den Tab mit der Custom Extraction als CSV oder Excel exportieren (Spalten `Address` und die Content-Spalte).
4. Nur den thematisch relevanten Teil der Domain exportieren, zum Beispiel einen Ratgeber-Bereich. Das hält den Export klein und verfälscht die Auswertung nicht (siehe unten).

## Eingaben im Detail

**Screaming-Frog-Export.** Erkannt werden URL-Spalten mit den Namen `Address`, `URL`, `Adresse`, `Page`, `Seite` und Content-Spalten wie `Extract Main Content 1`, `Main Content`, `Content`, `Text`, `Seiteninhalt`. Wichtig: Der Extraktor sollte Absätze als Zeilenumbrüche liefern, denn daran schneidet das Tool die Chunks.

**AI-Antwort einfügen.**

- Entferne Zeilen, die nicht zur Antwort gehören: Hinweise wie „KI-Antworten können Fehler enthalten“, Titel der Quellenkarten und Domainnamen.
- Trage den API-Key ein, bevor du Schritt 3 startest. Hat die Antwort keine erkennbare Struktur, zerlegt das Tool sie nach Bedeutung und braucht dafür schon in Schritt 3 das Embedding-Modell.

**Quell-URLs.** Werden per HTTP geladen, der Main Content kommt aus trafilatura. Seiten, die sich nicht laden lassen, stehen im Protokoll. Für solche Fälle kannst du den Content der Quellen auch als Datei im selben Format hochladen.

**Ohne Quell-URLs** läuft die Analyse ebenfalls. Der Status zeigt dann nur die relative Stärke innerhalb deines eigenen Contents. Für diesen Status braucht es mindestens 4 Aussagen in der AI-Antwort, sonst bleibt er leer.

## Datenschutz

Mit OpenAI oder Gemini gehen die Texte der AI-Antwort, deines Contents und der Quellen an den jeweiligen Anbieter. Mit Ollama bleibt alles auf deinem Rechner. Die geladenen URLs ruft das Tool direkt von deinem Rechner ab.

## Grenzen

- **URL-Liste:** gedacht für eine Handvoll Seiten. Die URLs werden nacheinander geladen, mit 20 Sekunden Zeitlimit pro Seite. Für eine ganze Domain nimm den Frog-Export.
- **Chunk-Embedding-Export:** rund 15 KB pro Chunk bei 1.536 Dimensionen, 5.000 Chunks sind etwa 70 MB. Das Upload-Limit liegt bei 1.000 MB.
- **Abgebrochene Analyse** (zum Beispiel wegen eines Rate-Limits): Ein zweiter Klick auf „Analyse starten“ macht dort weiter, wo der erste aufgehört hat, denn bereits eingebettete Texte liegen im Cache der Sitzung.

## Ergebnisse

| Tabelle | Inhalt |
|---|---|
| Seiten | Ranking aller Seiten (eigene und Quellen) nach Ähnlichkeit zur ganzen AI-Antwort |
| Themen | Bester Chunk deiner Domain pro Themengebiet, im Vergleich zum besten Quell-Chunk |
| Aussagen | Bester Chunk deiner Domain pro Aussage, mit Abstand zur besten Quelle und Status |
| Aussage × Seite | Jede Aussage gegen jede Seite: bester Chunk der Seite und Relevanz der Seite als Ganzes |

Im Tab „Aussage × Seite" stehen pro Aussage und Seite zwei Werte:

- **Bester Chunk Score:** Gibt es auf dieser Seite eine Passage, die zur Aussage passt?
- **Seiten-Relevanz:** Passt die Seite als Ganzes zum Thema der Aussage? Der Wert vergleicht die Aussage mit dem Durchschnitt aller Passagen der Seite. Er taugt zum Sortieren deiner Seiten, nicht als absolute Zahl: Seiten mit vielen Themen schneiden schlechter ab.

Hoher Chunk Score bei niedriger Seiten-Relevanz heißt: Die passende Passage steht auf einer Seite, die insgesamt etwas anderes behandelt. Pro Aussage erscheinen die zehn besten eigenen Seiten und alle Quellen. Bei bis zu 15 Seiten zeigt das Tool zusätzlich eine Matrix.

Status mit Quellen: **abgedeckt** (dein Chunk liegt höchstens 0,05 unter dem besten Quell-Chunk), **schwach** (bis 0,15 darunter), **Lücke** (mehr als 0,15 darunter). Beide Schwellen sind in der Seitenleiste einstellbar.

Das Excel enthält zusätzlich alle Chunks und ein Protokoll mit Datum, Modell, Chunk-Größe und nicht geladenen URLs. In jedem Blatt steht das Keyword in der ersten Spalte.

## Domain einmal einbetten, für viele Keywords nutzen

Nach einer Analyse kannst du die **Chunk-Embeddings herunterladen**. Beim nächsten Keyword lädst du diese Datei statt des Frog-Exports hoch. Dann wird nur noch die AI-Antwort (und ggf. die Quellen) eingebettet. Anbieter und Modell müssen dieselben sein wie beim Export. Auch die Quellen werden dann mit der Chunk-Größe des Exports geschnitten, damit eigene Chunks und Quell-Chunks vergleichbar bleiben. Das steht im Protokoll.

## Was die Zahlen bedeuten und was nicht

- **Ähnlichkeit ist notwendig, nicht hinreichend.** Ein hoher Score heißt nicht, dass Google dich zitiert. Rankings, Crawlbarkeit und Autorität entscheiden mit.
- **Nicht auf den Score optimieren.** Die AI-Antwort ist aus den Quellen entstanden, die Quellen scoren deshalb zwangsläufig hoch. Wer die Antwort nachschreibt, liefert keinen Grund, zitiert zu werden. Nützlich sind die Lücken: Aussagen, zu denen dir eine eigenständige Passage fehlt.
- **Der Vergleich ist größenabhängig.** Der beste eigene Chunk ist das Maximum über alle Chunks der Domain, der beste Quell-Chunk das Maximum über wenige Seiten. Je größer dein Export, desto leichter wird „abgedeckt“ erreicht. Beschränke den Export deshalb auf relevante Seiten.
- **Gleiche Extraktion für einen fairen Benchmark.** Eigener Content aus dem Frog-Extraktor und Quellen aus der automatischen Extraktion (trafilatura) unterscheiden sich in dem, was als Boilerplate gilt. Für einen fairen Vergleich lade den Content der Quellen mit demselben Frog-Setup und lade ihn als Datei hoch.
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

## Lizenz

MIT, siehe [LICENSE](LICENSE).
