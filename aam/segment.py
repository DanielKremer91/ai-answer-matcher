"""Split an AI answer into topics and statements and derive the chunk target size.

Structure first: headings, lead-ins ("Kosten: ...") and, when neither exists, blank lines.
If that yields a single topic and an embedding function is available, adjacent sentences
are compared and the answer is cut where their similarity drops.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from aam.models import SegmentedAnswer, Statement, Topic
from aam.text import count_words, normalize_space, short_label, split_sentences

MIN_TARGET = 15
MAX_TARGET = 80
MAX_HEADING_WORDS = 10
MAX_LEAD_WORDS = 5
MAX_STATEMENT_WORDS = 60
MIN_FALLBACK_SENTENCES = 6
INTRO_LABEL = "Einleitung"
_SIMILARITY_EPS = 1e-6

Embed = Callable[[list[str]], Sequence[Optional[np.ndarray]]]

_BULLET = re.compile(r"^\s*(?:[-*•–·▪◦]|\d{1,2}[.)])\s+")
_MD_HEADING = re.compile(r"^\s*#{1,6}\s+")
_WHOLE_BOLD = re.compile(r"^\s*(\*\*|__)(.+?)\1\s*:?\s*$")
_LEAD_IN = re.compile(r"^(?P<label>[^:.!?]{1,80}):\s+(?P<rest>\S.*)$")
_LABEL_ONLY = re.compile(r"^(?P<label>[^:.!?]{1,80}):$")  # "Trockenlegung & Abdichtung:" before sub-items
_SENTENCE_END = re.compile(r"[.!?:]$")


@dataclass
class _Line:
    text: str
    bullet: bool
    forced_heading: bool  # markdown heading or a line that is bold as a whole
    after_blank: bool


@dataclass
class _Draft:
    label: str
    texts: list[str] = field(default_factory=list)
    forced: bool = False
    heading: bool = False  # opened by a heading line; lead-ins inside it are statements, not topics
    prefix: str = ""  # label of a "Label:" line, carried into the statements that follow it


def _parse_lines(text: str) -> list[_Line]:
    lines: list[_Line] = []
    blank = False
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if not raw.strip():
            blank = True
            continue
        bullet = bool(_BULLET.match(raw))
        body = _BULLET.sub("", raw, count=1) if bullet else raw
        forced = bool(_MD_HEADING.match(body)) or bool(_WHOLE_BOLD.match(body))
        body = _MD_HEADING.sub("", body, count=1)
        body = normalize_space(body.replace("**", "").replace("__", ""))
        if not body:
            continue
        lines.append(_Line(body, bullet, forced, blank))
        blank = False
    return lines


def _heading_like(line: _Line) -> bool:
    """Short, unpunctuated, not a bullet and not a "Label: text" line (that is content, not a heading)."""
    return (not line.bullet and count_words(line.text) <= MAX_HEADING_WORDS
            and not _SENTENCE_END.search(line.text) and _lead_in(line) is None)


def _is_heading(line: _Line, following: Optional[_Line]) -> bool:
    """A heading needs text after it. Unless it is marked up explicitly, the next line must be
    running text (not itself heading-like), otherwise unpunctuated short lines become headings arbitrarily."""
    if following is None:
        return False
    if line.forced_heading:
        return True
    return _heading_like(line) and not _heading_like(following)


def _lead_in(line: _Line) -> Optional[re.Match]:
    match = _LEAD_IN.match(line.text)
    if match and count_words(match["label"]) <= MAX_LEAD_WORDS:
        return match
    return None


def _label_only(line: _Line) -> Optional[str]:
    match = _LABEL_ONLY.match(line.text)
    if match and count_words(match["label"]) <= MAX_LEAD_WORDS:
        return match["label"].strip()
    return None


def _structure_topics(lines: list[_Line]) -> list[_Draft]:
    following = [lines[i + 1] if i + 1 < len(lines) else None for i in range(len(lines))]
    explicit = any(_is_heading(line, following[i]) or _lead_in(line) for i, line in enumerate(lines))
    drafts: list[_Draft] = []
    for i, line in enumerate(lines):
        if _is_heading(line, following[i]):
            drafts.append(_Draft(line.text.rstrip(":").strip(), forced=line.forced_heading, heading=True))
            continue
        under_heading = bool(drafts) and drafts[-1].heading
        lead = _lead_in(line)
        if lead:
            if under_heading:  # "Sanierputz: Alter Putz wird ..." is a statement of the heading's topic
                drafts[-1].prefix = ""
                drafts[-1].texts.append(f"{lead['label'].strip()}: {lead['rest'].strip()}")
            else:
                drafts.append(_Draft(lead["label"].strip(), [lead["rest"].strip()]))
            continue
        label = _label_only(line)
        if label:
            if under_heading:  # "Trockenlegung & Abdichtung:" labels the sub-items that follow
                drafts[-1].prefix = label
            else:
                drafts.append(_Draft(label))
            continue
        if not drafts:
            drafts.append(_Draft(INTRO_LABEL))
        elif not explicit and line.after_blank and not line.bullet and drafts[-1].texts:
            drafts.append(_Draft(short_label(line.text)))
        text = line.text.rstrip(":").strip()
        drafts[-1].texts.append(f"{drafts[-1].prefix}: {text}" if drafts[-1].prefix else text)
    return _fold_empty(drafts)


def _fold_empty(drafts: list[_Draft]) -> list[_Draft]:
    """A heading without statements keeps its text: it becomes a statement of the topic before it."""
    out: list[_Draft] = []
    for draft in drafts:
        if draft.texts:
            out.append(draft)
        elif not draft.forced:
            if not out:
                out.append(_Draft(INTRO_LABEL))
            out[-1].texts.append(draft.label)
    return out


def _split_long(texts: list[str]) -> list[str]:
    out: list[str] = []
    for text in texts:
        if count_words(text) > MAX_STATEMENT_WORDS:
            out.extend(split_sentences(text))
        else:
            out.append(text)
    return out


def _semantic_topics(sentences: list[str], embed: Embed) -> Optional[list[_Draft]]:
    vectors = embed(sentences)
    if any(v is None for v in vectors):
        return None
    matrix = np.vstack(vectors).astype(np.float32)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    similarities = np.sum(matrix[:-1] * matrix[1:], axis=1)
    if float(similarities.std()) < _SIMILARITY_EPS:
        return None  # similarities barely vary: no meaningful topic change, keep the structural result
    threshold = float(similarities.mean() - similarities.std())
    groups = [[sentences[0]]]
    for sentence, similarity in zip(sentences[1:], similarities):
        if similarity < threshold - _SIMILARITY_EPS:
            groups.append([sentence])
        else:
            groups[-1].append(sentence)
    if len(groups) < 2:
        return None
    return [_Draft(short_label(group[0]), group) for group in groups]


def derive_target(word_counts: list[int]) -> tuple[int, float]:
    """Chunk target size (median, clamped to MIN_TARGET..MAX_TARGET) and the raw median."""
    median = float(statistics.median(word_counts))
    return int(min(MAX_TARGET, max(MIN_TARGET, round(median)))), median


def segment_answer(keyword: str, text: str, *, target_override: Optional[int] = None,
                   embed: Optional[Embed] = None) -> SegmentedAnswer:
    drafts = [_Draft(d.label, _split_long(d.texts)) for d in _structure_topics(_parse_lines(text))]
    if not drafts:
        raise ValueError("Die AI-Antwort enthält keine erkennbare Aussage.")
    method = "structure"
    if len(drafts) == 1 and embed is not None:
        sentences = [s for t in drafts[0].texts for s in split_sentences(t)]
        if len(sentences) >= MIN_FALLBACK_SENTENCES:
            semantic = _semantic_topics(sentences, embed)
            if semantic:
                drafts, method = semantic, "semantic"
    topics: list[Topic] = []
    next_id = 0
    for topic_id, draft in enumerate(drafts):
        statements = []
        for statement_text in draft.texts:
            statements.append(Statement(next_id, topic_id, statement_text, count_words(statement_text)))
            next_id += 1
        topics.append(Topic(topic_id, draft.label, tuple(statements)))
    derived, median = derive_target([s.words for t in topics for s in t.statements])
    target = int(target_override) if target_override else derived
    return SegmentedAnswer(keyword.strip(), tuple(topics), target, median, method)
