# AI Answer Matcher – Design Spec

Datum: 2026-10-03
Status: Entwurf zur Freigabe
Kontext: Tool für einen SEOKomm-Vortrag. Es wird als offenes GitHub-Repo an die Zuhörer weitergegeben. Schwesterprojekt des Broken Link Matcher (https://github.com/DanielKremer91/broken-link-matcher), dessen Muster und Provider-Schicht übernommen werden.

## 1. Ziel

Das Tool beantwortet die Frage: Wie nah ist mein Content an der Antwort, die AI Overviews oder AI Mode zu einem Keyword oder Prompt geben? Es vergleicht auf Embedding-Basis mit Kosinus-Ähnlichkeit, auf zwei Ebenen:

- **Seite gegen Antwort:** Welche Seite ist insgesamt am relevantesten für die ganze AI-Antwort?
- **Chunk gegen Chunk:** Welche Passage meiner Domain passt am besten zu jedem Themengebiet und zu jeder einzelnen Aussage der AI-Antwort?

Die von der AI zitierten Quellen dienen optional als Benchmark. Der Nutzen liegt in Lücken ("diese Aussage fehlt mir") und im relativen Abstand zu den Quellen, nicht im absoluten Score.

Nicht-Ziele:

- Keine Prognose, ob eine Seite zitiert wird. Ähnlichkeit ist notwendig, nicht hinreichend.
- Keine LLM-Aufrufe außer Embeddings. Segmentierung und Chunking sind deterministisch.
- Kein Speichern zwischen Sitzungen außer dem bewusst exportierten Chunk-Embedding-Export.
- Keine Verarbeitung mehrerer AI-Antworten in einem Lauf der Oberfläche.
- Keine Übernahme von Seiten-Embeddings aus Screaming Frog (gemittelte Vektoren taugen nicht für Chunk-Vergleiche).
- Keine 2x2-Diagnose-Tabelle pro URL.

## 2. Ausbaustufen

| Stufe | Inhalt |
|---|---|
| 1 | Paket `aam`, Streamlit-Oberfläche, manuelle Eingabe von Keyword, AI-Antwort und Quell-URLs, alle Auswertungen, Excel-Export, Wiederverwendung von Chunk-Embeddings |
| 2 | SerpApi-Abruf in der Oberfläche (füllt die Eingabefelder) und CLI-Befehl `study` für die Validierungsstudie |

Stufe 1 ist ohne Stufe 2 vollständig benutzbar. Stufe 2 bekommt einen eigenen Implementierungsplan. Diese Spec beschreibt beide, damit die Modulgrenzen von Anfang an passen.

## 3. Architektur

Ein Python-Paket `aam` mit getrennten Modulen und eine Streamlit-Datei `app.py`, die nur Eingaben einsammelt, Module aufruft und Ergebnisse anzeigt. Kein Modul in `aam` importiert Streamlit. Alle HTTP-Zugriffe laufen über `httpx`, damit Tests sie mocken können. Python 3.11 oder neuer, Umgebung mit `uv` wie im Broken Link Matcher. Oberfläche und Meldungen sind deutsch.

| Modul | Aufgabe | Hängt ab von |
|---|---|---|
| `aam.models` | Dataclasses (siehe 3.1) | – |
| `aam.ingest.tables` | CSV/Excel robust lesen (UTF-8, UTF-16, Tab, Semikolon), Spalten finden | pandas |
| `aam.ingest.content` | Frog-Export mit URL und Main Content → `Document`-Liste | tables |
| `aam.ingest.chunks_file` | Chunk-Embedding-Export dieses Tools einlesen und prüfen | tables |
| `aam.ingest.urls` | URL-Textfeld parsen, normalisieren, deduplizieren | – |
| `aam.extract` | URL laden, Main Content mit trafilatura extrahieren, Fehlerliste | httpx, trafilatura |
| `aam.segment` | AI-Antwort → Themen und Aussagen, Zielgröße ableiten | embeddings (nur Fallback) |
| `aam.chunk` | `Document` → Chunks um die Zielgröße, Statistik | – |
| `aam.embeddings` | Provider OpenAI, Gemini, Ollama, Batching, Cache nach Textinhalt (aus `blm` übernommen) | httpx |
| `aam.match` | Kosinus-Matrizen, Seiten-Ranking, Themen- und Aussagen-Treffer, Status | numpy |
| `aam.report` | Ergebnis → DataFrames und Excel | pandas, openpyxl |
| `aam.pipeline` | Orchestriert einen Lauf in zwei Phasen (`prepare`, `analyze`) | alle |
| `aam.serpapi` (Stufe 2) | AI Mode und AI Overview abrufen → Antworttext, Quell-URLs, organische URLs | httpx |
| `aam.study` (Stufe 2) | Validierungsstudie über mehrere Keywords | pipeline, serpapi |

### 3.1 Datenmodell

- `Document`: `url`, `text`, `origin` (`"own"` oder `"source"`).
- `Statement`: `id`, `topic_id`, `text`, `words`.
- `Topic`: `id`, `label`, `statements`.
- `SegmentedAnswer`: `keyword`, `topics`, `target_words`, `method` (`"structure"` oder `"semantic"`).
- `Chunk`: `id`, `url`, `origin`, `index`, `text`, `words`, `vector` (optional).
- `ChunkStats`: Anzahl Dokumente, Anzahl Chunks, Median, p10, p90 der Wortzahl, Anzahl zusammengelegter und geteilter Absätze.
- `RunMeta`: Keyword, Datum, Provider, Modell, Dimension, Zielgröße, Schwellen, nicht geladene URLs mit Grund.
- `Result`: Seiten-Ranking, Themen-Treffer, Aussagen-Treffer, Chunks, Statistiken, `RunMeta`.

## 4. Eingaben

1. **Keyword/Prompt** (Pflicht, Freitext). Steht als erste Spalte in jedem Excel-Blatt.
2. **AI-Antwort** (Pflicht, Textfeld). Zeilenumbrüche bleiben erhalten.
3. **Quell-URLs** (optional, Textfeld, eine URL pro Zeile). Ohne Quell-URLs läuft das Tool vollständig, siehe 7.4.
4. **Eigener Content** (Pflicht, mindestens einer der Wege):
   - Frog-Export (CSV oder Excel) mit URL-Spalte (`url`, `address`, `adresse`, `page`, `seite`) und Content-Spalte (`extract main content 1`, `main content`, `content`, `text`, `body text`, `seiteninhalt`, `inhalt`). Die Oberfläche zeigt, welche Spalten erkannt wurden.
   - URL-Liste im Textfeld. Das Tool lädt und extrahiert selbst.
   - Beides kombiniert. Bei gleicher URL gewinnt der Frog-Export.
   - Alternativ: Chunk-Embedding-Export aus einem früheren Lauf (siehe 8). Ersetzt die beiden anderen Wege.
5. **Quell-Content als Datei** (optional). Frog-Export für die Quell-URLs im selben Format. Ersetzt das Laden für die enthaltenen URLs.

## 5. Segmentierung der AI-Antwort (`aam.segment`)

Zwei Ebenen: Thema und Aussage.

**Strukturweg (Standard).** Die Antwort wird zeilenweise gelesen. Eine neue Themengrenze beginnt bei:

- einer Überschriftenzeile: höchstens 10 Wörter, kein Satzendezeichen am Ende, gefolgt von weiterem Text; Markdown-Präfixe (`#`, `**…**`) werden entfernt;
- einer Zeile mit fettem oder kurzem Einleitungswort vor einem Doppelpunkt (höchstens 5 Wörter vor dem Doppelpunkt, danach Text). Das Einleitungswort wird das Themenlabel, der Rest die erste Aussage;
- einer Leerzeile zwischen zwei Blöcken, wenn der folgende Block nicht mit einem Aufzählungszeichen an eine Liste anschließt. Diese Regel gilt nur, wenn die Antwort weder Überschriften noch Einleitungswörter enthält. Gibt es solche Marker, bestimmen allein sie die Themen, sonst würde ein zweiter Absatz unter einer Überschrift aus seinem Thema gerissen.

Eine Überschriftenzeile ohne folgende Aussage (etwa kurze Listenpunkte ohne Satzzeichen) geht nicht verloren: ihr Text wird zur Aussage des vorangehenden Themas.

Text vor der ersten erkannten Grenze bildet das Thema "Einleitung". Aufzählungszeichen und Nummerierungen werden entfernt.

**Semantischer Fallback.** Wenn der Strukturweg nur ein Thema findet und die Antwort mindestens 6 Sätze hat: Sätze einzeln einbetten, Kosinus benachbarter Sätze berechnen, Grenze dort, wo der Wert unter (Mittelwert − 1 Standardabweichung) fällt. Label ist der erste Satz des Abschnitts, auf 8 Wörter gekürzt. Bei weniger als 6 Sätzen bleibt es bei einem Thema.

**Aussagen.** Innerhalb eines Themas ist jede nichtleere Zeile eine Aussage. Zeilen über 60 Wörter werden an Satzgrenzen geteilt. Satztrennung per regulärem Ausdruck mit Schutz für gängige deutsche Abkürzungen (`z. B.`, `ca.`, `bzw.`, `Nr.`, `usw.`, Ordinalzahlen).

**Zielgröße.** Median der Wortzahl aller Aussagen, begrenzt auf 15 bis 80. Der Nutzer kann sie überschreiben.

## 6. Chunking (`aam.chunk`)

Eine Funktion für eigenen Content und Quellen:

1. Dokument an Zeilenumbrüchen in Absätze teilen, leere Zeilen verwerfen.
2. Absätze über dem Doppelten der Zielgröße an Satzgrenzen in Fenster teilen: Sätze sammeln, bis die Zielgröße erreicht ist; das nächste Fenster beginnt mit dem letzten Satz des vorigen (ein Satz Überlappung). Ein einzelner Satz über dem Doppelten der Zielgröße bleibt ein eigener Chunk.
3. Absätze unter der Hälfte der Zielgröße mit dem folgenden Absatz zusammenlegen, solange das Ergebnis das Doppelte der Zielgröße nicht überschreitet. Der letzte Absatz eines Dokuments wird mit dem vorigen zusammengelegt.
4. Dokumente ohne Text liefern keine Chunks und erscheinen im Protokoll.

`ChunkStats` zählt Dokumente, Chunks, zusammengelegte und geteilte Absätze und die Verteilung der Wortzahl.

**Seitenvektor.** Längengewichteter Mittelwert der Chunk-Vektoren einer URL (Gewicht = Wortzahl), danach normalisiert. Der Antwortvektor entsteht genauso aus den Aussagen-Vektoren. Dadurch gibt es keine Token-Limit-Probleme und keine zusätzlichen Embedding-Aufrufe.

## 7. Matching (`aam.match`)

Alle Vektoren werden L2-normalisiert. Symmetrischer Vergleich: alle Texte laufen ohne Query- oder Passage-Präfix durch dasselbe Modell (bei Gemini `task_type=SEMANTIC_SIMILARITY`).

### 7.1 Seiten-Ranking

Pro URL (eigen und Quelle gemischt): Seitenscore = Kosinus(Seitenvektor, Antwortvektor); bester Chunk = höchster Kosinus irgendeines Chunks der URL zu irgendeiner Aussage; Anzahl abgedeckter Aussagen (Status "abgedeckt" nach 7.3, bezogen auf den besten Chunk dieser URL je Aussage). Sortiert nach Seitenscore, Rang über alle URLs.

### 7.2 Themen und Aussagen

Pro Aussage: bester eigener Chunk (URL, Text, Score) über die gesamte Domain; bester Quell-Chunk (URL, Text, Score); Delta = eigener Score − Quell-Score.

Pro Thema: Themenvektor = längengewichteter Mittelwert seiner Aussagen-Vektoren. Bester eigener Chunk und bester Quell-Chunk gegen den Themenvektor, Delta, dazu die Anzahl Aussagen und wie viele davon abgedeckt, schwach oder Lücke sind.

### 7.2a Aussage gegen Seite

Pro Aussage und Seite (eigen und Quelle): bester Chunk dieser Seite zur Aussage (Text, Score) und Seiten-Relevanz = Kosinus(Aussage, Seitenvektor). Der beste Chunk beantwortet, ob die Seite eine passende Passage hat. Die Seiten-Relevanz beantwortet, ob die Seite als Ganzes zum Thema der Aussage passt; sie ist ein relatives Signal, weil der gemittelte Seitenvektor bei Seiten mit vielen Themen verdünnt. Pro Aussage werden die zehn besten eigenen Seiten nach Chunk-Score und alle Quellen ausgegeben, mit Rang. Der Vergleich bleibt symmetrisch: Das Modell bettet nie eine ganze Seite als einen Text ein.

### 7.3 Status mit Quellen

| Status | Bedingung |
|---|---|
| abgedeckt | Delta ≥ −0,05 |
| schwach | −0,15 ≤ Delta < −0,05 |
| Lücke | Delta < −0,15 |

Beide Schwellen sind einstellbar. Sie beziehen sich auf den Abstand zur besten Quelle, nicht auf absolute Kosinus-Werte, weil diese modellabhängig sind.

### 7.4 Status ohne Quellen

Wenn keine Quelle geladen werden konnte oder keine angegeben wurde: Spalten zu Quelle und Delta entfallen. Der Status kommt aus der Verteilung der besten eigenen Scores über alle Aussagen: oberstes Viertel "stark", bis Median "über Durchschnitt", bis unteres Viertel "unter Durchschnitt", darunter "schwach". Bei weniger als 4 Aussagen wird kein Status vergeben und ein Hinweis angezeigt. Das Seiten-Ranking enthält dann nur eigene URLs, und "Aussagen abgedeckt" zählt die Aussagen, bei denen die URL den besten eigenen Chunk der Domain stellt.

## 8. Wiederverwendung von Chunk-Embeddings

Der Lauf exportiert neben dem Excel eine Datei `chunks_embeddings.csv` für den eigenen Content: `url`, `chunk_index`, `text`, `words`, `provider`, `model`, `target_words`, `embedding_0..N`.

Beim Hochladen dieser Datei als eigener Content:

- Provider, Modell und Dimension müssen zur aktuellen Einstellung passen. Sonst Abbruch mit klarer Meldung, welche Werte in der Datei stehen.
- Die Chunks werden nicht neu geschnitten. Weicht die gespeicherte Zielgröße um mehr als 50 Prozent von der Zielgröße der aktuellen AI-Antwort ab, erscheint ein Warnhinweis, der Lauf geht weiter.
- Eingebettet werden nur noch AI-Antwort und Quellen.
- Die Quellen werden in diesem Fall mit der gespeicherten Zielgröße geschnitten, nicht mit der Zielgröße der aktuellen AI-Antwort. So haben eigene Chunks und Quell-Chunks dieselbe Granularität, und das Delta wird nicht durch unterschiedliche Chunk-Größen verzerrt.

## 9. Oberfläche (`app.py`)

Seitenleiste: Embedding-Provider und Modell (Default OpenAI `text-embedding-3-small`), API-Key aus Eingabefeld oder Umgebungsvariable (nie gespeichert), Zielgröße überschreiben, Schwellen "abgedeckt" und "schwach". In Stufe 2 zusätzlich der SerpApi-Key.

Hauptbereich in vier Schritten, jeder wird aktiv, wenn der vorige vollständig ist:

1. **AI-Antwort:** Keyword/Prompt, Antworttext, Quell-URLs (optional).
2. **Eigener Content:** Upload Frog-Export oder Chunk-Embedding-Export, Textfeld URL-Liste, optional Upload Quell-Content.
3. **Chunking prüfen:** Button "Segmentieren und chunken". Lädt URLs, segmentiert, chunkt. Zeigt:
   - das Protokoll: Themen, Aussagen, Median und Zielgröße der AI-Antwort; Seiten, Chunks, Median, p10, p90, zusammengelegte und geteilte Absätze für eigenen Content und für Quellen; nicht geladene URLs mit Grund; Segmentierungsmethode;
   - die AI-Chunks vollständig als Tabelle: Thema, Aussage, Wortzahl;
   - drei Beispiel-Chunks des eigenen Contents;
   - die Anzahl Texte, die im nächsten Schritt eingebettet werden.
   Die Zielgröße kann hier geändert und neu gechunkt werden, ohne Embedding-Kosten. Einzige Ausnahme: Greift der semantische Fallback der Segmentierung, werden in diesem Schritt die Sätze der AI-Antwort eingebettet (wenige Dutzend Texte, danach im Cache).
4. **Analysieren:** Embedding mit Fortschrittsbalken, Matching, dann Zusammenfassung, vier Tabellen (Seiten, Themen, Aussagen, Aussage × Seite, letztere bei bis zu 15 Seiten zusätzlich als Matrix) und zwei Downloads (Excel, Chunk-Embedding-Export). Der Chunk-Embedding-Export entfällt, wenn der eigene Content selbst aus einem solchen Export kam. Geänderte Schwellen werden ohne neues Embedding auf das vorhandene Ergebnis angewendet.

Zusammenfassung in zwei Sätzen: relevanteste eigene Seite mit Seitenscore und Rang, Anzahl abgedeckter Aussagen, Themen mit Lücken.

Ein Embedding-Cache pro Sitzung (Schlüssel: Provider, Modell, Text) verhindert doppelte Aufrufe bei Wiederholung mit anderen Schwellen.

## 10. Excel-Export (`aam.report`)

Jedes Blatt hat `Keyword/Prompt` als erste Spalte.

| Blatt | Spalten |
|---|---|
| Seiten | Rang, URL, Herkunft, Seitenscore, Bester Chunk Score, Aussagen abgedeckt, Aussagen gesamt |
| Themen | Thema, Anzahl Aussagen, Beste eigene URL, Bester eigener Chunk, Score, Beste Quell-URL, Bester Quell-Chunk, Score Quelle, Delta, abgedeckt / schwach / Lücke |
| Aussagen | Thema, Aussage (AI-Chunk), Wortzahl, Beste eigene URL, Bester eigener Chunk, Score, Beste Quell-URL, Bester Quell-Chunk, Score Quelle, Delta, Status |
| Aussage x Seite | Thema, Aussage (AI-Chunk), Rang, URL, Herkunft, Bester Chunk Score, Seiten-Relevanz, Bester Chunk |
| Chunks | Herkunft, URL, Chunk-Index, Text, Wortzahl (AI-Aussagen, eigene Chunks und Quell-Chunks; ohne Vektoren) |
| Protokoll | Datum, Provider, Modell, Dimension, Zielgröße, Segmentierungsmethode, Schwellen, Chunk-Statistiken, nicht geladene URLs mit Grund |

Ohne Quellen entfallen die Quell-Spalten und Delta, und die Spalte "Aussagen abgedeckt" im Blatt Seiten heißt "Aussagen mit bestem Chunk". Scores mit vier Nachkommastellen.

## 11. Fehlerbehandlung

- Fehler erscheinen als deutsche Meldung im jeweiligen Schritt, nie als Stacktrace.
- Nicht ladbare URLs (Timeout 20 s, Statuscode, leerer Main Content) brechen den Lauf nicht ab. Sie stehen im Protokoll mit Grund.
- Fehlende Spalten: Meldung nennt die gefundenen Spaltennamen und die erwarteten.
- Embedding-Fehler: bis zu 3 Versuche mit wachsender Wartezeit, danach Abbruch mit der Meldung des Anbieters.
- Kein eigener Content mit Text: Abbruch vor dem Embedding.
- AI-Antwort leer oder ohne erkennbare Aussage: Abbruch in Schritt 3.

## 12. Stufe 2: SerpApi und Studie

**Abruf in der Oberfläche.** Mit SerpApi-Key erscheint in Schritt 1: Keyword, Land, Sprache, Auswahl AI Mode oder AI Overview, Button "Abrufen". Das Ergebnis füllt Keyword, Antworttext und Quell-URLs. Der Nutzer kann alles korrigieren, danach läuft der Ablauf unverändert.

- AI Mode: `engine=google_ai_mode`, Antwort aus `text_blocks`, Quellen aus `references[].link`. Überschriften- und Listenblöcke werden so in Text mit Zeilenumbrüchen übersetzt, dass der Strukturweg der Segmentierung sie erkennt.
- AI Overview: zuerst `engine=google`, daraus `ai_overview.page_token`, sofort danach `engine=google_ai_overview` (das Token läuft nach einer Minute ab). Liefert die Suche die Overview direkt mit, entfällt der zweite Aufruf. Gibt es keine Overview, erscheint eine Meldung.
- Stand 2026-10-03: 250 freie Suchen pro Monat, kein Pay-per-Call, kleinster Tarif 25 $ für 1.000 Suchen. Die genaue Feldstruktur der Antworten wird beim Bau von Stufe 2 gegen echte Antworten geprüft und als Fixture gespeichert.

**Studie.** `python -m aam.cli study keywords.csv --mode overview --country de --lang de`. Pro Keyword: Antwort, zitierte Quellen und organische Top 10 abrufen, alle URLs laden, mit denselben Modulen segmentieren, chunken und einbetten, pro URL Seitenscore, besten Chunk und abgedeckte Aussagen berechnen, Label "zitiert" oder "nicht zitiert". Rohantworten werden als JSON auf Platte gecacht. Ausgabe: Excel mit Blatt "URLs" (eine Zeile pro Keyword und URL) und Blatt "Auswertung" (pro Metrik Median zitiert und nicht zitiert, AUC, Anteil der Keywords, bei denen die beste zitierte Seite vorne liegt).

## 13. Tests

pytest, ohne Netz. Ein deterministischer Fake-Embedder (Vektor aus Wort-Hashes) ersetzt die Provider. HTTP wird über `httpx.MockTransport` gemockt.

Fixtures: AI-Antworten mit Überschriften, mit Einleitungswort und Doppelpunkt, mit Bullets, als reiner Fließtext; Frog-Exporte in UTF-8 mit Komma und UTF-16 mit Tab; eine HTML-Seite für die Extraktion; ein Chunk-Embedding-Export.

Pflichtfälle:

- Segmentierung: jede Antwortform liefert die erwarteten Themen, Labels und Aussagen; der Fallback greift nur bei einem Thema und mindestens 6 Sätzen; Abkürzungen trennen keine Sätze.
- Zielgröße: Median korrekt, Begrenzung auf 15 und 80, Überschreiben wirkt.
- Chunking: kurze Absätze werden zusammengelegt, lange geteilt mit einem Satz Überlappung, Statistik stimmt, leere Dokumente liefern keine Chunks.
- Matching: bekannte Vektoren ergeben erwartete Scores, Ränge, Deltas und Status an beiden Schwellen.
- Lauf ohne Quell-URLs: läuft durch, Quell-Spalten fehlen, Status aus Perzentilen, bei weniger als 4 Aussagen kein Status.
- Lauf, bei dem alle Quell-URLs scheitern: verhält sich wie ohne Quellen und nennt die URLs im Protokoll.
- Chunk-Embedding-Export: Export und Wiedereinlesen ergeben identische Ergebnisse; falsches Modell oder falsche Dimension werden abgelehnt; abweichende Zielgröße warnt.
- Excel: sechs Blätter, Keyword in der ersten Spalte jedes Blatts.
- Frog bei doppelter URL vor URL-Liste.

## 14. Repo

Struktur: `aam/`, `app.py`, `tests/`, `docs/`, `examples/` (anonymisierte Beispieldaten), `requirements.txt`, `requirements-dev.txt`, `README.md` (deutsch, mit Einrichtung über `uv` und über `pip`).
