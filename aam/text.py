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


# A Markdown table separator row such as `|---|:---:|`, or a bare rule: no content.
SEPARATOR_LINE = re.compile(r"^\|?[\s:|-]+\|?$")


def is_separator_line(line: str) -> bool:
    return bool(SEPARATOR_LINE.match(line.strip()))


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
