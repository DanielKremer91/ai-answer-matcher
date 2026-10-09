// =============================================================================
// Screaming Frog Custom JavaScript: Main Content mit Überschriften-Markierung
// =============================================================================
//
// Gibt den Main Content einer Seite als Text zurück, ein Absatz pro Zeile.
// Überschriften werden markiert (## für H2, ### für H3 usw.), damit der
// AI Answer Matcher Abschnitte erkennen kann. Tabellenzeilen werden zu einer
// Zeile mit " | " zwischen den Zellen, Listenpunkte zu je einer Zeile.
//
// Einrichtung im Screaming Frog:
// - Configuration > Spider > Rendering: JavaScript
// - Configuration > Custom > Custom JavaScript: neues Snippet, Typ "Extraction",
//   Name "Main Content", diesen Code einfügen
// - Nach dem Crawl: Tab "Custom JavaScript" exportieren (CSV oder Excel).
//   Die Spalte heißt dann "Main Content 1" und wird vom Tool erkannt.
//
// Für eine bestimmte Website lässt sich unten ein genauerer Selektor für den
// Content-Bereich eintragen, z. B. '.article-body'.
// =============================================================================

// Bereich, der als Main Content gilt. Der erste Treffer wird verwendet,
// ohne Treffer die ganze Seite.
const CONTENT_SELECTORS = ['main', 'article', '[role="main"]', '#content', '.content'];

// Elemente, die innerhalb des Main Contents ignoriert werden
const EXCLUDE_SELECTOR = [
  'script', 'style', 'noscript', 'template', 'svg', 'iframe', 'video', 'audio', 'form',
  'nav', 'header', 'footer', 'aside',
  '[role="navigation"]', '[role="banner"]', '[role="contentinfo"]', '[role="complementary"]',
  '[aria-hidden="true"]',
  '[id*="cookie" i]', '[class*="cookie" i]', '[id*="consent" i]', '[class*="consent" i]',
  '[class*="breadcrumb" i]', '[class*="newsletter" i]', '[class*="share" i]'
].join(',');

// Elemente, nach denen ein Zeilenumbruch folgt
const BLOCK_TAGS = /^(P|DIV|SECTION|ARTICLE|LI|UL|OL|DL|DT|DD|TR|TABLE|BLOCKQUOTE|FIGCAPTION|PRE|DETAILS|SUMMARY)$/;
const HEADING_TAGS = /^H([1-6])$/;

function findContentRoot() {
  for (const selector of CONTENT_SELECTORS) {
    const el = document.querySelector(selector);
    if (el && el.textContent.trim()) return el;
  }
  return document.body;
}

// Text rekursiv sammeln; ausgeschlossene Elemente überspringen.
// Überschriften bekommen eine eigene Zeile mit #-Markierung nach Ebene.
function collectText(node) {
  let text = '';
  node.childNodes.forEach(child => {
    if (child.nodeType === Node.TEXT_NODE) {
      text += child.nodeValue;
    } else if (child.nodeType === Node.ELEMENT_NODE) {
      if (child.matches(EXCLUDE_SELECTOR)) return;
      if (child.tagName === 'BR') { text += '\n'; return; }
      const heading = HEADING_TAGS.exec(child.tagName);
      if (heading) {
        const title = child.textContent.replace(/\s+/g, ' ').trim();
        if (title) text += '\n' + '#'.repeat(Number(heading[1])) + ' ' + title + '\n';
        return;
      }
      text += collectText(child);
      if (child.tagName === 'TD' || child.tagName === 'TH') text += ' | ';
      else if (BLOCK_TAGS.test(child.tagName)) text += '\n';
    }
  });
  return text;
}

function extractMainContent() {
  return collectText(findContentRoot())
    .split('\n')
    .map(line => line.replace(/\s+/g, ' ').replace(/(\s*\|\s*)+$/, '').trim())
    .filter(line => line.replace(/[\s|#]/g, ''))
    .join('\n');
}

return seoSpider.data(extractMainContent());
