from datetime import date

import numpy as np
import pytest

from aam.gap import analyze_gap
from aam.gap_statements import (VERDICT_COVERED, VERDICT_DETAIL_GAPS, VERDICT_TOPIC_GAP, VERDICT_TOPIC_ONLY,
                                chunk_sections, section_verdicts, statement_level, verdict)
from aam.match import STATUS_COVERED, STATUS_GAP, STATUS_WEAK, unit_rows
from aam.models import OWN, SOURCE, Document
from aam.sections import split_sections
from tests.fakes import ANSWER, bow_vector


def sentence(prefix: str, n: int, shared: str = "") -> str:
    ws = [f"{prefix}{i}" for i in range(n)]
    if shared:
        ws = shared.split() + ws
    ws[0] = ws[0].capitalize()
    return " ".join(ws) + "."


def page(url, origin, *sections):
    return Document(url, "# Titel\n" + "\n".join(f"## {t}\n{b}" for t, b in sections), origin)


def embed(texts):
    return [bow_vector(t, 512) for t in texts]


def run(comps, owns, **kwargs):
    return analyze_gap(comps, owns, embed, "fake", "fake-model", today=date(2026, 10, 13), **kwargs)


def vectors_for(*chunk_lists):
    texts = list(dict.fromkeys(c.text for chunks in chunk_lists for c in chunks))
    matrix = unit_rows(np.vstack(embed(texts)).astype(np.float32))
    return dict(zip(texts, matrix))


# Three statements of one section share the topic words "kosten dach"; a fourth section has one statement.
A = sentence("alpha", 10, "kosten dach")
B = sentence("beta", 10, "kosten dach")
C = sentence("gamma", 10, "kosten dach")
D = sentence("delta", 12)


def test_chunk_sections_keeps_the_section_of_every_chunk():
    doc = page("https://w.de/", SOURCE, ("Kosten", "\n".join([A, B, C])), ("Garantie", D))
    chunks = chunk_sections(doc.url, SOURCE, split_sections(doc), 12)
    assert [c.section.title for c in chunks] == ["Kosten", "Kosten", "Kosten", "Garantie"]
    assert [c.text for c in chunks] == [A, B, C, D]


def test_verdict_labels():
    assert verdict([STATUS_GAP, STATUS_GAP, STATUS_COVERED]).verdict == VERDICT_TOPIC_GAP
    assert verdict([STATUS_GAP, STATUS_COVERED, STATUS_COVERED]).verdict == VERDICT_DETAIL_GAPS
    assert verdict([STATUS_WEAK, STATUS_WEAK, STATUS_COVERED]).verdict == VERDICT_TOPIC_ONLY
    assert verdict([STATUS_COVERED, STATUS_WEAK]).verdict == VERDICT_COVERED
    v = verdict([STATUS_GAP, STATUS_GAP, STATUS_WEAK, STATUS_COVERED])
    assert (v.total, v.covered, v.weak, v.gaps) == (4, 1, 1, 2)
    assert verdict([]).verdict == "" and verdict([]).total == 0


def test_verbatim_statement_is_covered_and_a_missing_statement_is_not():
    comp = page("https://w.de/", SOURCE, ("Kosten", "\n".join([A, B, C])))
    own = page("https://me.de/", OWN, ("Preise", "\n".join([A, sentence("other", 10)])))
    cc = chunk_sections(comp.url, SOURCE, split_sections(comp), 12)
    oc = chunk_sections(own.url, OWN, split_sections(own), 12)
    rows = statement_level(comp.url, cc, oc, vectors_for(cc, oc), covered_delta=0.05, weak_delta=0.15)
    by_text = {r.text: r for r in rows}
    assert by_text[A].status == STATUS_COVERED and by_text[A].own_text == A
    assert by_text[A].score == pytest.approx(1.0)
    assert by_text[A].ref_neighbours == pytest.approx(float(by_text[A].ref_neighbours)) and by_text[A].ref_neighbours < 1
    # B has no counterpart: the best own chunk is A, which only shares the topic words, exactly as B's
    # neighbours do -> not better than the neighbours, and A is more specific to itself than to B.
    assert by_text[B].status in (STATUS_WEAK, STATUS_GAP)
    assert by_text[B].delta < by_text[A].delta
    assert by_text[B].ref_specificity == pytest.approx(1.0)  # own chunk A is identical to competitor chunk A
    assert all(r.section.title == "Kosten" for r in rows)


def test_a_generic_own_chunk_close_to_everything_does_not_cover_anything():
    comp = page("https://w.de/", SOURCE, ("Kosten", "\n".join([A, B, C])))
    generic = "Kosten dach kosten dach kosten dach kosten dach kosten dach kosten dach."  # only the shared words
    own = page("https://me.de/", OWN, ("Preise", generic))
    cc = chunk_sections(comp.url, SOURCE, split_sections(comp), 12)
    oc = chunk_sections(own.url, OWN, split_sections(own), 12)
    rows = statement_level(comp.url, cc, oc, vectors_for(cc, oc), covered_delta=0.05, weak_delta=0.15)
    assert all(r.own_text == generic for r in rows)
    # the generic chunk sits equally close to every statement: specificity equals the score, so
    # delta <= 0 and nothing counts as the same statement with a margin; it is "topic only" at best
    assert all(r.status != STATUS_COVERED or r.delta <= 0.05 for r in rows)
    assert all(r.ref_specificity == pytest.approx(r.score, abs=1e-5) for r in rows)


def test_single_chunk_section_borrows_the_page_median_as_neighbour_reference():
    comp = page("https://w.de/", SOURCE, ("Kosten", "\n".join([A, B, C])), ("Garantie", D))
    own = page("https://me.de/", OWN, ("Preise", A))
    cc = chunk_sections(comp.url, SOURCE, split_sections(comp), 12)
    oc = chunk_sections(own.url, OWN, split_sections(own), 12)
    rows = statement_level(comp.url, cc, oc, vectors_for(cc, oc), covered_delta=0.05, weak_delta=0.15)
    d = next(r for r in rows if r.text == D)
    refs = sorted(r.ref_neighbours for r in rows if r.text != D)
    assert d.ref_neighbours == pytest.approx(refs[len(refs) // 2])  # median of the Kosten neighbours
    assert d.status == STATUS_GAP  # nothing in the own page resembles D


def test_page_with_one_chunk_has_no_neighbour_reference_and_uses_specificity_only():
    comp = page("https://w.de/", SOURCE, ("Garantie", D))
    own = page("https://me.de/", OWN, ("Preise", D))
    cc = chunk_sections(comp.url, SOURCE, split_sections(comp), 12)
    oc = chunk_sections(own.url, OWN, split_sections(own), 12)
    [row] = statement_level(comp.url, cc, oc, vectors_for(cc, oc), covered_delta=0.05, weak_delta=0.15)
    assert row.ref_neighbours is None and row.ref_specificity == 0.0
    assert row.status == STATUS_COVERED and row.delta == pytest.approx(1.0)


def test_overlapping_windows_are_no_neighbour_reference():
    """A long paragraph is cut into windows that share one sentence; such a sibling would look like a
    near-identical neighbour and push the reference up."""
    long = " ".join(sentence(f"w{i}", 8, "kosten dach") for i in range(8))
    comp = page("https://w.de/", SOURCE, ("Kosten", long))
    own = page("https://me.de/", OWN, ("Preise", sentence("other", 10)))
    cc = chunk_sections(comp.url, SOURCE, split_sections(comp), 15)
    assert len(cc) > 2
    oc = chunk_sections(own.url, OWN, split_sections(own), 15)
    vectors = vectors_for(cc, oc)
    rows = statement_level(comp.url, cc, oc, vectors, covered_delta=0.05, weak_delta=0.15)
    matrix = np.vstack([vectors[c.text] for c in cc])
    self_sim = matrix @ matrix.T
    for i, row in enumerate(rows):
        best_sibling = max(float(self_sim[i, k]) for k in range(len(cc)) if k != i)
        assert row.ref_neighbours <= best_sibling + 1e-6
    assert any(row.ref_neighbours < max(float(self_sim[i, k]) for k in range(len(cc)) if k != i) - 0.05
               for i, row in enumerate(rows))


def test_analyze_gap_carries_statements_verdicts_and_settings():
    comp = page("https://w.de/", SOURCE, ("Kosten", "\n".join([A, B, C])), ("Garantie", D))
    own = page("https://me.de/", OWN, ("Preise", "\n".join([A, B])))
    result = run([comp], [own], statement_target=12, covered_delta=0.04, weak_delta=0.2)
    assert (result.statement_target, result.covered_delta, result.weak_delta) == (12, 0.04, 0.2)
    assert [s.text for s in result.statements] == [A, B, C, D]
    by_title = {row.competitor.title: row for row in result.pages[0].rows}
    kosten, garantie = by_title["Kosten"].verdict, by_title["Garantie"].verdict
    assert kosten.total == 3 and kosten.covered >= 2
    assert garantie.total == 1 and garantie.gaps == 1 and garantie.verdict == VERDICT_TOPIC_GAP
    assert section_verdicts(result.statements)[("https://w.de/", by_title["Kosten"].competitor.index)] == kosten


def test_statements_get_the_nearest_ai_statement_and_a_priority():
    from aam.segment import segment_answer

    answer = segment_answer("kw", ANSWER)
    kfw = "Die KfW fördert die Dachdämmung mit zinsgünstigen Krediten."
    comp = page("https://w.de/", SOURCE, ("Förderung", "\n".join([kfw, sentence("other", 10)])))
    own = page("https://me.de/", OWN, ("Preise", sentence("mine", 10)))
    result = run([comp], [own], answer=answer, statement_target=12)
    row = next(s for s in result.statements if s.text == kfw)
    assert row.statement is not None and row.statement.text == kfw
    assert row.statement_score == pytest.approx(1.0)
    assert row.priority == pytest.approx(round(1.0 - row.score, 4))
    assert all(s.statement is None for s in run([comp], [own], statement_target=12).statements)
