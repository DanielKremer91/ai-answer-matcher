from aam.models import OWN, SOURCE, SegmentedAnswer, Statement, Topic


def test_origin_constants():
    assert (OWN, SOURCE) == ("own", "source")


def test_statements_are_flattened_in_topic_order():
    s1 = Statement(0, 0, "a b", 2)
    s2 = Statement(1, 1, "c", 1)
    answer = SegmentedAnswer("kw", (Topic(0, "T1", (s1,)), Topic(1, "T2", (s2,))), 15, 1.5, "structure")
    assert answer.statements == [s1, s2]
