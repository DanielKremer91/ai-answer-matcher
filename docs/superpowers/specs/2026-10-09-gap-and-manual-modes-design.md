# AI Answer Matcher – Ergänzung: Gap-Modus, manueller Abgleich, Abschnitte

Datum: 2026-10-09
Status: Entwurf zur Umsetzung
Ergänzt: 2026-10-03-ai-answer-matcher-design.md (Stufe 1). Alles dort Beschriebene bleibt gültig, soweit es hier nicht geändert wird.

## 1. Ziel

Drei Erweiterungen, die aus der Nutzung von Stufe 1 entstanden sind:

1. **Manueller Abgleich:** ein kurzer Text (eine Aussage der AI-Antwort) gegen einen beliebigen eingefügten Text, kurz oder lang, mit verlässlichem Wert.
2. **Content-Gap zwischen Seiten:** Ausgangspunkt ist eine Wettbewerber-Seite. Für jeden ihrer Abschnitte wird der ähnlichste Abschnitt der eigenen Seite (oder Domain) gesucht. Absteigend sortiert zeigt die Liste, welche Abschnitte des Wettbewerbers bei dir fehlen. Optional mit Bezug zur AI-Antwort.
3. **Abschnitte aus dem Layout:** Überschriften werden erkannt (`##`-Markierung aus trafilatura oder aus dem Frog-Snippet), ein Abschnitt ist alles von einer Überschrift bis zur nächsten derselben Ebene.

Nicht-Ziele: keine Schwelle für „fehlt" (die sortierte Liste mit Werten reicht), kein Retrieval-Modus (Query/Passage-Präfixe), keine neue Anbieter-Schicht.

## 2. Grundregel: Zerlegung folgt der Frage

- „Steht diese Aussage im Text?" → der lange Text wird in Fenster geteilt, das beste Fenster zählt (Stufe 1, manueller Abgleich).
- „Behandelt dieser Abschnitt dasselbe Thema?" → Abschnitt gegen Abschnitt, als Ganzes eingebettet (Gap-Modus). Fenster nur, wenn ein Abschnitt mehr als das Vierfache des anderen hat.
- Ganztext-Werte entstehen aus dem Embedding des ganzen Texts, wenn er in das Modell passt, sonst aus dem Durchschnitt der Teile (bereits in Stufe 1 umgesetzt, gilt auch für Abschnitte).

## 3. Abschnitte (`aam.sections`)

Eingabe ist der Main Content einer Seite, ein Absatz pro Zeile. Überschriften sind Zeilen der Form `# Titel` bis `###### Titel`. Sie stammen aus trafilatura (Markdown-Ausgabe) oder aus dem Frog-Snippet `examples/screaming-frog-main-content.js`.

- **Abschnittsebene:** die kleinste Überschriftennummer größer als 1, die auf der Seite mindestens einmal vorkommt (also H2 vor H3 vor H4). Gibt es keine solche Ebene, ist die ganze Seite ein Abschnitt mit dem Titel der H1, sonst „Gesamte Seite".
- **Abschnitt:** alles von einer Überschrift der Abschnittsebene bis zur nächsten Überschrift derselben oder einer höheren Ebene. Tiefer liegende Überschriften bleiben als Textzeilen (ohne Markierung) im Abschnitt. Tabellenzeilen und Listen gehören zum Abschnitt.
- **Einleitung:** Text vor der ersten Abschnittsüberschrift wird ein Abschnitt mit dem Titel der H1, sonst „Einleitung". Leere Abschnitte (Überschrift ohne Text) entfallen.
- `Section(url, origin, index, title, level, text, words)`.

Für den AI-Modus (Stufe 1) ändert sich nur: Markierungen `#` am Zeilenanfang werden beim Chunking entfernt, die Überschrift bleibt als kurze Zeile erhalten und wird wie bisher mit dem folgenden Absatz zusammengelegt.

`aam.extract.extract_text` liefert ab jetzt die Markdown-Ausgabe von trafilatura, damit Überschriften markiert ankommen. Fettdruck-Markierungen (`**`) werden entfernt.

## 4. Vergleich zweier Texte (`aam.compare`)

Eine Funktion für den manuellen Abgleich und für Abschnittspaare:

```
compare_texts(short, long, provider, cache) -> Comparison
```

- Ist der längere Text höchstens viermal so lang wie der kürzere: ein Kosinus-Wert zwischen beiden Ganztext-Vektoren. `windowed = False`.
- Sonst: Zielgröße = Wortzahl des kürzeren Texts, begrenzt auf 15 bis 80. Der längere Text wird mit `aam.chunk` in Fenster der Zielgröße und zusätzlich in Fenster der doppelten Zielgröße geteilt. Jedes Fenster wird eingebettet, der beste Kosinus über beide Fenstergrößen zählt. Ergebnis: bester Wert, bestes Fenster (Text, Größe), dazu der Ganztext-Wert (Embedding des ganzen langen Texts, wenn er passt, sonst Durchschnitt der Fenster). `windowed = True`.
- `Comparison(score, best_window, window_words, whole_score, windowed, short_words, long_words)`.

Ganztext-Vektoren nutzen dieselbe Grenze wie in Stufe 1 (`fits_whole_text`); passt ein Text nicht, ist sein Vektor der Durchschnitt seiner Fenster.

## 5. Manueller Abgleich (Modus „Absatz gegen Text")

Eingaben: Feld A „Aussage oder Absatz der AI-Antwort", Feld B „Dein Text". Anbieter aus der Seitenleiste.

Ausgabe: der Wert nach Abschnitt 4. Bei `windowed`: der Ganztext-Wert, der beste Fensterwert, der Fenstertext und der Hinweis „Dein Text ist {n}-mal so lang wie die Aussage. Der Ganztext-Wert wird durch die Länge gedrückt, der Fensterwert ist der verlässlichere." Ohne Fenster: ein Wert mit Satz „Beide Texte sind ähnlich lang, der Wert gilt direkt."

Keine Datei, keine Excel. Ein Button „Vergleichen".

## 6. Content-Gap zwischen Seiten (Modus „Content-Gap")

Eingaben:

- **Wettbewerber:** eine oder mehrere URLs (Textfeld) oder ein Frog-Export (Tabelle mit URL und Main Content). Jede Seite wird einzeln ausgewertet.
- **Eigener Content:** eine URL oder mehrere (Textfeld), ein Frog-Export, oder eingefügter Text. Mit mehreren eigenen Seiten wird über alle gesucht; die Tabelle nennt die URL des Treffers.
- **AI-Antwort:** optional. Mit Antwort bekommt jede Wettbewerber-Zeile zusätzlich die nächste Aussage der Antwort und deren Kosinus.
- Keine Schwelle. Fenster-Regel und Anbieter aus Abschnitt 4 und der Seitenleiste.

Ablauf: Seiten laden oder lesen, Abschnitte bilden (Abschnitt 3), Abschnittsvektoren einbetten, jeden Wettbewerber-Abschnitt mit jedem eigenen Abschnitt vergleichen (Abschnitt 4), pro Wettbewerber-Abschnitt den besten eigenen Abschnitt behalten.

Ausgabe pro Wettbewerber-Seite:

- **Ganztext-Wert** Seite gegen Seite (bei mehreren eigenen Seiten: die beste).
- **Protokoll:** Abschnittsebene (z. B. „H2"), Anzahl Abschnitte und Wörter je Seite, Hinweis wenn eine Seite nur einen Abschnitt hat.
- **Tabelle,** absteigend nach Kosinus: Abschnitt des Wettbewerbers (Titel, Wörter), bester eigener Abschnitt (URL, Titel, Wörter), Kosinus, bestes Fenster (falls gefenstert), optional nächste AI-Aussage mit Kosinus. Dazu die Texte beider Abschnitte zum Nachlesen.
- **Excel:** ein Blatt „Abschnitte" (die Tabelle mit Keyword/Prompt als erste Spalte, leer ohne AI-Antwort) und ein Blatt „Protokoll".

Die untersten Zeilen der Tabelle sind die Lücken. Das Tool schreibt kein „fehlt" in die Tabelle; die Oberfläche erklärt die Lesart. Der Lesehinweis (`gap_summary`, für die erste Wettbewerber-Seite) nennt die Spanne und die Leserichtung: „{url}: {n} Abschnitte, Kosinus von {min} bis {max}. Die untere Hälfte der Liste (Kosinus unter {Median}) zuerst prüfen, dort stehen die Kandidaten für Lücken." (Kommadezimalen, zwei Stellen; bei einem einzigen Abschnitt: dessen Hinweis oder „{url}: ein Abschnitt, Kosinus {x} gegen {eigene URL}."; ohne Zeilen leer).

**Fenster-Kandidaten.** Bis `WINDOW_ALL_UP_TO = 40` eigenen Abschnitten bekommt jedes Paar über dem Längenverhältnis Fenster. Darüber besteht die Kandidatenmenge pro Wettbewerber-Abschnitt aus den fünf besten eigenen Abschnitten nach Ganztext-Kosinus und den fünf besten unter den Paaren mit Längenverhältnis über 4 (sonst fällt ein langer eigener Abschnitt, der die Passage enthält, durch die Vorauswahl).

**Wert eines gefensterten Paars.** `Kosinus = max(Ganztext-Wert, bester Fensterwert)`. `GapRow.method` und die Spalte „Vergleich" (nach „Kosinus") nennen die Herkunft: „Ganztext", „Fenster aus Wettbewerber-Abschnitt" (Fenster aus dem längeren Wettbewerber-Text) oder „Fenster aus eigenem Abschnitt". „Bestes Fenster" ist nur gefüllt, wenn das Fenster gewonnen hat.

**Eingebetteter Text.** Eingebettet und gefenstert wird `Titel\nText` (`Section.embed_text`, auch für die Einleitung mit ihrem Titel); angezeigt wird der Text ohne Titel. So findet ein FAQ-Abschnitt, dessen Überschrift die Frage ist, seinen Gegenpart.

**Fenster.** Vor dem Fenstern werden Zeilen unter der halben Zielgröße mit der folgenden Zeile verbunden (die letzte mit der vorigen), Trennzeilen von Markdown-Tabellen (`|---|---|`) entfallen; `extract_text` entfernt sie schon beim Lesen.

**Zu große Blöcke (Abschnitte, Abschnitt 3).** Ein Block, der tiefere Überschriften enthält und länger als das Doppelte des mittleren (unteren Median) Blocks ist, wird zusätzlich an der kleinsten tieferen Überschriftenebene geteilt; die neuen Abschnitte tragen diese Ebene. Ein Newsletter-Kasten als einzige H2 schluckt so nicht mehr die H3-Abschnitte des Artikels.

Der Hinweis „Die Seite hat nur einen Abschnitt." erscheint, sobald eine Seite nach dem Schneiden genau einen Abschnitt hat; „Verglichen wird die ganze Seite." folgt nur, wenn sie keine Abschnittsebene hat.

## 7. Stufe-1-Ergänzung: Nähe zum Quell-Chunk

Der Tab „Aussagen" und das Excel-Blatt „Aussagen" bekommen die Spalte „Nähe zum Quell-Chunk": Kosinus zwischen dem besten eigenen Chunk und dem besten Quell-Chunk derselben Aussage. Nur mit Quellen.

## 8. Oberfläche

Ganz oben ein Modus-Schalter: „AI-Antwort gegen Content" (Stufe 1, unverändert), „Absatz gegen Text", „Content-Gap zwischen Seiten". Die Seitenleiste gilt für alle Modi; Chunking-Schalter und Schwellen erscheinen nur im ersten Modus. Jeder Modus hat einen Erklärkasten. Session-State je Modus getrennt, damit ein Wechsel nichts zerstört.

## 9. Tests

pytest ohne Netz, Fake-Embedder. Pflichtfälle: Abschnittsebene H2/H3/keine, Einleitung, tiefere Überschriften als Text, Markierungen beim Chunking entfernt; `compare_texts` mit und ohne Fenster, Verhältnis genau 4, beide Fenstergrößen, Ganztext-Rückfall bei Überlänge; Gap-Modus mit einer und mehreren eigenen Seiten, mit und ohne AI-Antwort, Sortierung, Excel mit Keyword-Spalte; manueller Modus in der App; Modus-Wechsel ohne Fehler; Stufe-1-Tests unverändert grün.
