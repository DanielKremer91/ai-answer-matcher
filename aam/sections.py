"""Cut a page into sections along its headings (see spec 2026-10-09, section 3)."""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Optional

from aam.models import Document
from aam.text import count_words, normalize_space

HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")

INTRO_TITLE = "Einleitung"
WHOLE_PAGE_TITLE = "Gesamte Seite"
# A block with deeper headings is split further when it is longer than this many times the median block.
OVERSIZE_FACTOR = 2


@dataclass(frozen=True)
class Section:
    url: str
    origin: str
    index: int
    title: str
    level: int  # heading level of the section; 0 for the introduction / the whole page
    text: str  # section body, one paragraph per line, no heading markers
    words: int

    @property
    def embed_text(self) -> str:
        """The text that is embedded and windowed: the title carries the topic (an FAQ heading is the question)."""
        return f"{self.title}\n{self.text}"


def strip_heading_marker(line: str) -> str:
    """Title of a heading line; any other line is returned unchanged."""
    match = HEADING.match(line)
    return match.group(2) if match else line


def heading_level(line: str) -> Optional[int]:
    match = HEADING.match(line)
    return len(match.group(1)) if match else None


def _lines(text: str) -> list[str]:
    return [line for line in (raw.strip() for raw in text.split("\n")) if line]


def section_level(text: str) -> Optional[int]:
    """Smallest heading level above 1 that occurs at least once; None when there is none."""
    levels = {heading_level(line) for line in _lines(text)}
    found = [level for level in levels if level is not None and level > 1]
    return min(found) if found else None


def plain_text(doc_or_text) -> str:
    """The text without heading markers: trimmed lines, empty ones removed, joined with newlines."""
    text = doc_or_text if isinstance(doc_or_text, str) else doc_or_text.text
    return "\n".join(strip_heading_marker(line) for line in _lines(text))


def _first_h1(lines: list[str]) -> Optional[str]:
    for line in lines:
        if heading_level(line) == 1:
            return strip_heading_marker(line)
    return None


_Body = list[tuple[Optional[int], str]]  # (heading level or None, text without marker)
_Block = tuple[str, int, _Body]  # (title, level, body)


def _body_text(body: _Body) -> str:
    return "\n".join(normalize_space(text) for _, text in body)


def _split_oversized(blocks: list[_Block], level: int) -> list[_Block]:
    """Cut every block that holds deeper headings and is much longer than the median block at those headings."""
    sizes = [count_words(_body_text(body)) for _, _, body in blocks if _body_text(body).strip()]
    if not sizes:
        return blocks
    limit = OVERSIZE_FACTOR * statistics.median_low(sizes)
    result: list[_Block] = []
    for title, block_level, body in blocks:
        deeper = {lvl for lvl, _ in body if lvl is not None and lvl > level}
        if not deeper or count_words(_body_text(body)) <= limit:
            result.append((title, block_level, body))
            continue
        sub = min(deeper)
        result.append((title, block_level, []))
        for lvl, text in body:
            if lvl == sub:
                result.append((text, sub, []))
            else:
                result[-1][2].append((lvl, text))
    return result


def split_sections(doc: Document) -> list[Section]:
    lines = _lines(doc.text)
    if not lines:
        return []
    level = section_level(doc.text)
    if level is None:
        body = [strip_heading_marker(line) for line in lines]
        text = "\n".join(body)
        return [Section(doc.url, doc.origin, 0, _first_h1(lines) or WHOLE_PAGE_TITLE, 0, text, count_words(text))]

    # blocks: (title, level, body lines); the first block is the introduction. Only an H1 before the first
    # section heading names the introduction (and is not part of its text); any other H1 is a plain line.
    intro_title: Optional[str] = None
    blocks: list[_Block] = [(INTRO_TITLE, 0, [])]
    for line in lines:
        current = heading_level(line)
        if current == level:
            blocks.append((strip_heading_marker(line), level, []))
        elif current == 1 and len(blocks) == 1 and intro_title is None:
            intro_title = strip_heading_marker(line)
        else:
            blocks[-1][2].append((current, strip_heading_marker(line)))
    if intro_title:
        blocks[0] = (intro_title, 0, blocks[0][2])

    sections: list[Section] = []
    for title, block_level, body in _split_oversized(blocks, level):
        text = _body_text(body)
        if not text.strip():
            continue
        sections.append(Section(doc.url, doc.origin, len(sections), title, block_level, text, count_words(text)))
    return sections
