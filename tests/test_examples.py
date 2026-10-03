from pathlib import Path

from aam.ingest.content import read_content_table
from aam.models import SOURCE
from aam.pipeline import analyze, prepare
from aam.report import to_excel
from tests.fakes import FakeProvider

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_examples_run_through_the_whole_pipeline():
    answer = (EXAMPLES / "beispiel-ai-antwort.txt").read_text(encoding="utf-8")
    own, own_info = read_content_table(EXAMPLES / "beispiel-content.csv")
    sources, _ = read_content_table(EXAMPLES / "beispiel-quellen.csv", origin=SOURCE)
    assert own_info.documents == 3 and len(sources) == 2
    prepared = prepare("dachsanierung kosten", answer, own_docs=own, source_docs=sources)
    assert [t.label for t in prepared.answer.topics] == ["Einleitung", "Kosten im Überblick", "Förderung", "Dauer"]
    result = analyze(prepared, FakeProvider(), None)
    assert result.match.has_sources is True
    statuses = {m.statement.text: m.status for m in result.match.statements}
    assert statuses["Die KfW fördert die Dachdämmung mit zinsgünstigen Krediten."] == "Lücke"
    assert len(to_excel(result)) > 0
