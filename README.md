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
2. **Eigener Content:** Screaming-Frog-Export hochladen (Spalten für URL und Main Content, CSV oder Excel) oder eigene URLs eintragen, die das Tool selbst lädt. Beides lässt sich kombinieren, bei gleicher URL gewinnt der Export. Für einen schnellen Test kannst du den Text einer Seite auch direkt einfügen, ein Absatz pro Zeile. Dasselbe gilt für den Content einer Quelle (Feld „Quell-Content als Text einfügen“) und im Gap-Modus für den Wettbewerber.
3. **Chunking prüfen:** Das Tool zeigt, wie es die AI-Antwort in Themen und Aussagen zerlegt hat, welche Chunk-Größe daraus folgt und wie viele Chunks aus deinem Content entstanden sind. Bis hierhin entstehen keine Embedding-Kosten, außer die Antwort hat keine erkennbare Struktur: dann werden ihre Sätze schon hier eingebettet, um Themengrenzen zu finden.
4. **Analysieren:** Embeddings werden erstellt, danach erscheinen Zusammenfassung, vier Tabellen und die Downloads.

Die Chunk-Zielgröße ist der Median der Aussagenlänge mal dem **Chunk-Faktor** (Standard 1,5): Website-Text braucht für denselben Sachverhalt mehr Wörter als eine verdichtete AI-Aussage, deshalb dürfen deine Chunks etwas größer sein als die Aussagen. Bei Median 14 ergibt das eine Zielgröße von 21 Wörtern; Absätze bis zur doppelten Zielgröße bleiben ganz. Zwei Schalter in der Seitenleiste bestimmen, wie dein Content geschnitten wird. **Kurze Absätze zusammenlegen** verbindet Zeilen unter der halben Zielgröße, etwa Überschriften, mit dem folgenden Absatz. **Lange Absätze teilen** schneidet Absätze über der doppelten Zielgröße an Satzgrenzen in Fenster, die einzeln eingebettet werden; das beste Fenster zählt. Beides ist standardmäßig an. Schaltest du das Teilen ab, bleibt ein langer Absatz ein Vektor, und eine einzelne Aussage darin geht im Durchschnitt unter: In einem Test lag der Kosinus-Wert eines wörtlich enthaltenen Satzes bei 0,80 in einem 36-Wörter-Absatz, bei 0,56 in einem 291-Wörter-Absatz und bei 0,48 in einem Absatz ohne den Satz. Schritt 3 listet alle geteilten Absätze auf.

Zum Ausprobieren liegen Beispieldateien in `examples/`.

## Drei Modi

Ganz oben wählst du den Modus. Anbieter, Modell und API-Key in der Seitenleiste gelten für alle drei. Die Chunking-Schalter und die Schwellen gibt es nur im ersten Modus. Jeder Modus hat seine eigenen Eingaben, ein Wechsel setzt nichts zurück: Texte, Einstellungen, Ergebnisse und hochgeladene Dateien bleiben erhalten. Eine hochgeladene Datei zeigt nach dem Wechsel „Zuletzt geladen: …“ und gilt weiter, bis du eine andere wählst oder sie mit „Datei entfernen“ löschst.

**AI-Antwort gegen Content.** Der oben beschriebene Ablauf: eine AI-Antwort gegen deine Domain, Chunk gegen Chunk.

**Absatz gegen Text.** Für den schnellen Einzelfall: Du fügst eine Aussage oder einen Absatz der AI-Antwort und deinen Text ein und klickst auf „Vergleichen“. Ist dein Text höchstens viermal so lang wie die Aussage, gilt ein Kosinus-Wert zwischen den beiden ganzen Texten. Ist er länger, schneidet das Tool ihn in Fenster von der Länge der Aussage (und in doppelt so lange) und zeigt zwei Werte: den **Ganztext-Wert**, den die Länge nach unten zieht, und das **beste Fenster** mit seinem Text. Der Fensterwert zeigt, ob die Aussage an einer Stelle vorkommt.

**Content-Gap zwischen Seiten.** Du gibst eine oder mehrere Wettbewerber-Seiten und deinen eigenen Content an (URLs, Frog-Export oder bei deinem Content eingefügter Text). Das Tool schneidet jede Seite an ihren Überschriften in Abschnitte und sucht zu jedem Abschnitt des Wettbewerbers den ähnlichsten Abschnitt deines Contents. Die Tabelle ist nach Kosinus absteigend sortiert. Die Zeilen am Ende sind die Kandidaten für Lücken; das Tool schreibt kein „fehlt“ hinein, die Wertung bleibt bei dir. Dazu gibt es den Ganztext-Wert Seite gegen Seite, ein Protokoll, die Texte beider Abschnitte zum Nachlesen und ein Excel mit den Blättern „Abschnitte“ und „Protokoll“. Mit einer optional eingefügten AI-Antwort steht in jeder Zeile zusätzlich die nächste Aussage der Antwort mit ihrem Kosinus. Die Spalte „Vergleich“ nennt, woher der Wert kommt: „Ganztext“, „Fenster aus Wettbewerber-Abschnitt“ oder „Fenster aus eigenem Abschnitt“. Bei Paaren mit starkem Längenunterschied zählt der bessere von Ganztext- und Fensterwert. Die Überschrift eines Abschnitts wird mit eingebettet.

Für den Gap-Modus braucht der Main Content die Überschriften als Markierung (`## H2`, `### H3`). Das Snippet `examples/screaming-frog-main-content.js` für Screaming Frogs Custom JavaScript liefert genau das: ein Absatz pro Zeile, Überschriften mit `##`. Seiten, die das Tool selbst lädt, bringen ihre Überschriften mit. Schon eine einzige H2 teilt die Seite in die Einleitung und diesen Abschnitt. Ohne Überschriften hat eine Seite nur einen Abschnitt, verglichen wird dann die ganze Seite.


Mit AI-Antwort bekommt jede Zeile der Gap-Tabelle zusätzlich die nächste Aussage der Antwort, deren Kosinus und eine **Priorität**: Kosinus zur AI-Aussage minus dein Kosinus. Hoch heißt: Der Abschnitt des Wettbewerbers liegt nah an etwas, das die AI-Antwort sagt, und du hast dazu kaum etwas. Die Sortierung „Priorität“ stellt diese Lücken nach oben.

## Screaming-Frog-Export erstellen

1. In Screaming Frog unter *Configuration → Custom → Custom JavaScript* ein Snippet anlegen und den Code aus `examples/screaming-frog-main-content.js` einfügen. Er liefert den Hauptinhalt der Seite ohne Navigation, Footer und Seitenleisten, jeden Absatz als eigene Zeile und die Überschriften mit `##`, `###` usw. markiert; die Einrichtung steht im Kopf der Datei. Das Tool schneidet daran die Chunks und im Gap-Modus die Abschnitte.

   **Vorsicht mit XPath:** Ein Custom Extraction mit XPath und „Extract Text“ verliert Überschriften und Zeilenumbrüche, weil es nur den zusammenhängenden Text der Knoten liefert. Der Export enthält dann keine Absätze und keine Abschnitte, im Gap-Modus ist jede Seite ein einziger Abschnitt.
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

- **Gap-Modus mit vielen eigenen Abschnitten:** Passt ein Abschnitt viel länger als der andere, schneidet das Tool Fenster. Bei bis zu 40 eigenen Abschnitten gilt das für jedes Paar mit mehr als vierfachem Längenunterschied. Bei einer ganzen Domain bekommen nur die besten Kandidaten Fenster: pro Wettbewerber-Abschnitt die fünf eigenen Abschnitte mit dem höchsten Ganztext-Wert und die fünf besten unter den Paaren mit starkem Längenunterschied. Ein eigener Abschnitt, der die Passage enthält, aber weder so noch so unter den Besten liegt, wird übersehen.
- **URL-Liste:** gedacht für eine Handvoll Seiten. Die URLs werden nacheinander geladen, mit 20 Sekunden Zeitlimit pro Seite. Für eine ganze Domain nimm den Frog-Export.
- **Chunk-Embedding-Export:** rund 15 KB pro Chunk bei 1.536 Dimensionen, 5.000 Chunks sind etwa 70 MB. Das Upload-Limit liegt bei 1.000 MB.
- **Abgebrochene Analyse** (zum Beispiel wegen eines Rate-Limits): Ein zweiter Klick auf „Analyse starten“ macht dort weiter, wo der erste aufgehört hat, denn bereits eingebettete Texte liegen im Cache der Sitzung.

## Ergebnisse

| Tabelle | Inhalt |
|---|---|
| Seiten | Ranking aller Seiten (eigene und Quellen) nach Ähnlichkeit zur ganzen AI-Antwort |
| Themen | Bester Chunk deiner Domain pro Themengebiet, im Vergleich zum besten Quell-Chunk |
| Aussagen | Bester Chunk deiner Domain pro Aussage, mit Abstand zur besten Quelle, der Nähe zum Quell-Chunk (Cosinus der beiden besten Chunks, nur mit Quell-URLs) und Status |
| Aussage × Seite | Jede Aussage gegen jede Seite: bester Chunk der Seite und Relevanz der Seite als Ganzes |

Der Seitenvektor für „Seiten" und für die Seiten-Relevanz ist das Embedding des kompletten Seitentexts, solange der Text in das Modell passt (OpenAI und bge-m3 etwa 4.000 bis 5.000 Wörter, Gemini etwa 1.200). Längere Seiten und Seiten aus einem Chunk-Embedding-Export bekommen stattdessen den wortgewichteten Durchschnitt ihrer Chunk-Vektoren. Das Protokoll zählt beide Fälle.

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
