"""Main content by walking the DOM in reading order.

Why not only trafilatura: it guesses the main text by heuristics and, on real pages, dropped collapsed
FAQ accordions (OBI) and all heading markers (zooplus). The walk here takes the main container, drops
boilerplate by tag, role and class name, and emits one line per block in document order: headings as
`## Title`, list items and paragraphs as lines, table rows as `| a | b |`. Collapsed elements stay in,
since an accordion's text is in the HTML even when the browser hides it.
"""

from __future__ import annotations

import re
from typing import Optional, Union

from lxml import html as lxml_html

HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
# Elements whose content never belongs to the main text.
DROP_TAGS = {"script", "style", "noscript", "template", "svg", "iframe", "form", "button", "input", "select",
             "textarea", "label", "nav", "header", "footer", "aside", "menu", "dialog", "video", "audio",
             "canvas", "map", "object", "embed", "picture", "img", "source", "link", "meta", "head"}
DROP_ROLES = {"navigation", "banner", "contentinfo", "complementary", "dialog", "search", "menu", "menubar",
              "toolbar", "tablist", "alert", "alertdialog", "status"}
# Class or id fragments of boilerplate blocks inside the main container. Word boundaries keep "author"
# from matching "authority", the fragments are matched against each class token and the id.
_BOILERPLATE_WORDS = (
    "nav", "navbar", "navigation", "menu", "breadcrumb", "breadcrumbs", "footer", "sidebar", "cookie", "consent",
    "newsletter", "share", "sharing", "social", "socials", "related", "teaser", "teasers", "recommend",
    "recommendation", "recommendations", "author", "authors", "autor", "autoren", "redaktion", "byline",
    "comment", "comments", "promo", "promotion", "advert", "advertisement", "ad", "ads", "banner", "modal",
    "popup", "toc", "tableofcontents", "inhaltsverzeichnis", "pagination", "pager", "rating", "ratings",
    "review", "reviews", "tags", "taglist", "cta", "widget", "skip", "login", "search", "cart", "basket",
    "wishlist", "gallery", "carousel", "slider", "video", "player", "sticky", "overlay", "tooltip",
    "feedback", "survey", "club", "shop", "product", "products", "price", "prices", "offer", "offers",
    "card", "cards", "similar", "similiar", "more", "weitere", "empfehlungen", "verwandte", "ähnliche",
)
_BOILERPLATE = re.compile(r"(?:^|[-_ :])(" + "|".join(_BOILERPLATE_WORDS) + r")(?:$|[-_ :])", re.IGNORECASE)
_TEASER_TITLES = re.compile(r"^(das könnte (dich|sie) auch interessieren|weitere (beiträge|artikel|ratgeber)\b.*|"
                            r"ähnliche (beiträge|artikel)|mehr zum thema|passende (produkte|artikel)|"
                            r"(related|similar) (articles|posts)|you might also like|empfehlungen( für dich)?|"
                            r"beliebte (beiträge|artikel)|aktuelle (beiträge|artikel)|das könnte dir auch gefallen)\s*:?$",
                            re.IGNORECASE)
_TOC_TITLES = re.compile(r"^(inhalt|inhalte|inhaltsverzeichnis|inhaltsübersicht|übersicht|table of contents|"
                         r"contents|in diesem artikel|das erwartet dich|auf einen blick)\s*:?$", re.IGNORECASE)
_CARD_CHILDREN = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "div", "section", "article", "figure"}
_LEAF_TAGS = {"p", "li", "dt", "dd", "figcaption", "summary", "pre", "caption", "address", "legend"}
_SPACE = re.compile(r"\s+")
_INVISIBLE = re.compile(r"[\u00ad\u200b\u200c\u200d\u2060\ufeff]")  # soft hyphen, zero-width characters
_CHARSET = re.compile(rb"""charset\s*=\s*["']?\s*([\w.-]+)""", re.IGNORECASE)
_MIN_CONTAINER_WORDS = 50
_MIN_SHARE_OF_BODY = 0.2  # a <main> with less than this share of the body text is not the content


def _norm(text: Optional[str]) -> str:
    return _SPACE.sub(" ", _INVISIBLE.sub("", text or "").replace("\xa0", " ")).strip()


_BLOCKISH = {"p", "div", "li", "ul", "ol", "dl", "dt", "dd", "br", "tr", "td", "th", "table", "section", "article",
             "blockquote", "figure", "figcaption", "summary", "details", "pre", "hr", "h1", "h2", "h3", "h4", "h5", "h6"}


def _raw(el) -> str:
    """Text of an element with the source whitespace intact. Inline children join without extra spaces
    (so `m<sup>2</sup>` stays `m2`), block children are separated by a space."""
    parts = [el.text or ""]
    for child in el:
        if isinstance(child.tag, str):
            inner = _raw(child)
            parts.append(f" {inner} " if child.tag.lower() in _BLOCKISH else inner)
        parts.append(child.tail or "")
    return "".join(parts)


def _text(el) -> str:
    return _norm(_raw(el))


def _words(el) -> int:
    return len(_text(el).split())


def _tokens(el) -> list[str]:
    tokens = (el.get("class") or "").split()
    if el.get("id"):
        tokens.append(el.get("id"))
    return tokens


def _is_boilerplate(el) -> bool:
    if not isinstance(el.tag, str):
        return True  # comments, processing instructions
    tag = el.tag.lower()
    if tag == "header":
        return not any(child.tag.lower() in HEADINGS for child in el.iter() if isinstance(child.tag, str))
    if tag == "button":  # an accordion's question lives in a button; a short control ("Menü") does not count
        return len(_text(el).split()) < 4
    if tag in DROP_TAGS:
        return True
    if tag == "a" and any(isinstance(c.tag, str) and c.tag.lower() in _CARD_CHILDREN for c in el):
        return True  # a link wrapping headings or paragraphs is a teaser card
    role = (el.get("role") or "").lower()
    if role in DROP_ROLES:
        return True
    if tag in HEADINGS or tag in _LEAF_TAGS:
        return False  # a heading or paragraph with class "teaser-title" stays; its container decides
    return any(_BOILERPLATE.search(token) for token in _tokens(el))


def _is_link_list(el) -> bool:
    """A table of contents: a list whose items are nothing but links, mostly to anchors on the same page.
    Lists of links to other pages stay, inside the main container they are usually content."""
    items = [child for child in el if isinstance(child.tag, str) and child.tag.lower() == "li"]
    if len(items) < 2:
        return False
    anchors = 0
    for item in items:
        links = item.findall(".//a")
        if not links:
            return False
        if _norm(" ".join(_text(a) for a in links)) != _text(item):
            return False
        anchors += any((a.get("href") or "").startswith("#") for a in links)
    return anchors * 2 >= len(items)


def _container(doc):
    body = doc.find("body")
    if body is None:
        body = doc
    body_words = max(_words(body), 1)
    candidates = doc.xpath("//main") + doc.xpath("//*[@role='main']")
    articles = sorted(doc.xpath("//article"), key=_words, reverse=True)
    if articles:
        candidates.append(articles[0])
    for xpath in ("//*[@id='content']", "//*[@id='main']", "//*[@id='main-content']", "//*[@class='content']"):
        candidates += doc.xpath(xpath)
    for candidate in candidates:
        words = _words(candidate)
        if words >= _MIN_CONTAINER_WORDS and words / body_words >= _MIN_SHARE_OF_BODY:
            return candidate
    return body


_MAX_BOLD_HEADING_WORDS = 10
_SENTENCE_END = re.compile(r"[.!?:;,]$")


def _bold_heading(el) -> Optional[str]:
    """A paragraph that is nothing but one short bold phrase works as a heading (OBI marks its sections
    this way). Returns the phrase, or None."""
    children = [c for c in el if isinstance(c.tag, str)]
    if len(children) != 1 or children[0].tag.lower() not in ("strong", "b") or _norm(el.text) or _norm(children[0].tail):
        return None
    title = _text(children[0])
    if not title or len(title.split()) > _MAX_BOLD_HEADING_WORDS or _SENTENCE_END.search(title):
        return None
    return title


class _Walker:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self._pending: list[str] = []
        self._level = 1  # level of the last heading, for bold paragraphs that act as headings

    def _flush(self) -> None:
        text = _norm(" ".join(self._pending))
        self._pending = []
        if text:
            self.lines.append(text)

    def _emit(self, text: str) -> None:
        self._flush()
        text = _norm(text)
        if text:
            self.lines.append(text)

    def _table(self, table) -> None:
        self._flush()
        for row in table.iter("tr"):
            cells = [_text(cell) for cell in row if isinstance(cell.tag, str) and cell.tag.lower() in ("td", "th")]
            if any(cells):
                self.lines.append("| " + " | ".join(cells) + " |")

    def walk(self, el) -> None:
        if _is_boilerplate(el):
            return
        tag = el.tag.lower()
        if tag in HEADINGS:
            title = _text(el)
            if title:
                self._level = HEADINGS[tag]
                self._emit("#" * self._level + " " + title)
            else:
                self._flush()
        elif tag == "p" and _bold_heading(el):
            self._emit("#" * min(6, self._level + 1) + " " + _bold_heading(el))
        elif tag == "table":
            self._table(el)
        elif tag in ("ul", "ol") and _is_link_list(el):
            self._flush()
        elif tag in _LEAF_TAGS:
            self._emit(_text(el))
        elif tag == "br":
            self._flush()
        else:
            self._pending.append(_norm(el.text))
            for child in el:
                if isinstance(child.tag, str) and child.tag.lower() in _INLINE:
                    if not _is_boilerplate(child):
                        self._pending.append(_text(child))
                else:
                    self.walk(child)
                self._pending.append(_norm(child.tail))
            if tag not in _INLINE:
                self._flush()


_INLINE = {"a", "span", "strong", "em", "b", "i", "u", "small", "sup", "sub", "mark", "abbr", "code", "time",
           "q", "cite", "dfn", "kbd", "s", "del", "ins", "var", "samp", "bdi", "bdo", "wbr", "font", "data",
           "output", "ruby", "rt", "rp", "big", "tt", "nobr"}


def _drop_teaser_sections(lines: list[str]) -> list[str]:
    """From a heading like "Das könnte dich auch interessieren" to the next heading of the same or a
    higher level everything is recommendations, not the page's content."""
    out: list[str] = []
    skipping = 0
    for line in lines:
        level = _heading_level(line)
        if skipping and (not level or level > skipping):
            continue
        skipping = 0
        if level and _TEASER_TITLES.match(line.lstrip("#").strip()):
            skipping = level
            continue
        out.append(line)
    return out


def _drop_empty_headings(lines: list[str]) -> list[str]:
    """A heading directly followed by a heading of the same or a higher level (or by nothing) has no
    content of its own, typically a table of contents whose link list was dropped."""
    out: list[str] = []
    for i, line in enumerate(lines):
        level = _heading_level(line)
        if level:
            following = lines[i + 1] if i + 1 < len(lines) else None
            if following is None or (_heading_level(following) and _heading_level(following) <= level):
                continue
            if _TOC_TITLES.match(line.lstrip("#").strip()):
                continue
        out.append(line)
    return out


def _heading_level(line: str) -> int:
    match = re.match(r"^(#{1,6})\s", line)
    return len(match.group(1)) if match else 0


def _dedupe(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for line in lines:
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


def extract_dom(html: Union[str, bytes]) -> str:
    """Main content as one block per line, headings marked with `#`; "" when nothing usable is found."""
    if isinstance(html, bytes):
        match = _CHARSET.search(html[:20000])
        encoding = match.group(1).decode("ascii", "ignore") if match else "utf-8"
        try:
            html = html.decode(encoding, errors="replace")
        except LookupError:
            html = html.decode("utf-8", errors="replace")
    html = re.sub(r"^\s*<\?xml[^>]*\?>", "", html)  # lxml rejects a declaration in a str
    if not html.strip():
        return ""
    # remove_comments keeps the text after a comment (Vue/Nuxt wrap headings in hydration comments);
    # removing comment nodes afterwards would drop that text with them.
    parser = lxml_html.HTMLParser(remove_comments=True, remove_pis=True)
    doc = lxml_html.document_fromstring(html, parser=parser)
    container = _container(doc)
    walker = _Walker()
    walker.walk(container)
    walker._flush()
    lines = walker.lines
    if not any(line.startswith("# ") for line in lines):  # the H1 often sits in a hero block outside <main>
        for h1 in doc.xpath("//h1"):
            title = _text(h1)
            if title:
                lines = [f"# {title}"] + [line for line in lines if line != title]
                break
    lines = _dedupe(_drop_empty_headings(_drop_teaser_sections(lines)))
    text = "\n".join(lines)
    return text if len(text.split()) >= _MIN_CONTAINER_WORDS else ""
