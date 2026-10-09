"""Cut a page into sections along its headings (see spec 2026-10-09, section 3)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from aam.models import Document
from aam.text import count_words, normalize_space

HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")

INTRO_TITLE = "Einleitung"
WHOLE_PAGE_TITLE = "Gesamte Seite"


@dataclass(frozen=True)
class Section:
    url: str
    origin: str
    index: int
    title: str
    level: int  # heading level of the section; 0 for the introduction / the whole page
    text: str  # section body, one paragraph per line, no heading markers
    words: int


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
    """Smallest heading level above 1 that occurs at least twice; None when there is none."""
    counts: dict[int, int] = {}
    for line in _lines(text):
        level = heading_level(line)
        if level is not None and level > 1:
            counts[level] = counts.get(level, 0) + 1
    repeated = [level for level, n in counts.items() if n >= 2]
    return min(repeated) if repeated else None


def _first_h1(lines: list[str]) -> Optional[str]:
    for line in lines:
        if heading_level(line) == 1:
            return strip_heading_marker(line)
    return None


def split_sections(doc: Document) -> list[Section]:
    lines = _lines(doc.text)
    if not lines:
        return []
    level = section_level(doc.text)
    if level is None:
        body = [strip_heading_marker(line) for line in lines]
        text = "\n".join(body)
        return [Section(doc.url, doc.origin, 0, _first_h1(lines) or WHOLE_PAGE_TITLE, 0, text, count_words(text))]

    # blocks: (title, level, body lines); the first block is the introduction
    blocks: list[tuple[str, int, list[str]]] = [(_first_h1(lines) or INTRO_TITLE, 0, [])]
    for line in lines:
        current = heading_level(line)
        if current == level:
            blocks.append((strip_heading_marker(line), level, []))
        elif current == 1:
            if len(blocks) > 1:  # a further H1 ends the open section and starts its own block
                blocks.append((strip_heading_marker(line), 1, []))
            # the first H1 is the title of the introduction and not part of its text
        else:
            blocks[-1][2].append(strip_heading_marker(line))

    sections: list[Section] = []
    for title, block_level, body in blocks:
        text = "\n".join(normalize_space(line) for line in body)
        if not text.strip():
            continue
        sections.append(Section(doc.url, doc.origin, len(sections), title, block_level, text, count_words(text)))
    return sections
