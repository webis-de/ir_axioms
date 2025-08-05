from ir_axioms.axiom import PROX1, PROX2, PROX3, PROX4, PROX5
from ir_axioms.model import TextQuery, TextDocument


def test_prox1() -> None:
    query = TextQuery(id="q1", text="blue car")
    document1 = TextDocument(id="d1", text="a blue car goes through the city")
    document2 = TextDocument(id="d2", text="through city blue goes car goes")

    axiom = PROX1()

    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1


def test_prox2() -> None:
    query = TextQuery(id="q1", text="q1 q2")
    document1 = TextDocument(id="d1", text="q1 x q2 y z a b c")
    document2 = TextDocument(id="d2", text="x y q1 q2")

    axiom = PROX2()

    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1


def test_prox3() -> None:
    query = TextQuery(id="q1", text="q1 q2")
    document1 = TextDocument(id="d1", text="a b c q1 d q2 e q1 q2")
    document2 = TextDocument(id="d2", text="a q2 b q1 q2")
    document3 = TextDocument(id="d3", text="q1 b q2")

    axiom = PROX3()

    # Document d2 contains the query phrase earlier.
    assert axiom.preference(query, document1, document2) == -1
    assert axiom.preference(query, document2, document1) == 1
    # Document d3 does not contain the query phrase.
    assert axiom.preference(query, document1, document3) == 1
    assert axiom.preference(query, document3, document1) == -1
    assert axiom.preference(query, document2, document3) == 1
    assert axiom.preference(query, document3, document2) == -1


def test_prox4() -> None:
    query = TextQuery(id="q1", text="q1 q2")
    document1 = TextDocument(id="d1", text="a b c q1 d q2 e q1")
    document2 = TextDocument(id="d2", text="a q2 b q2 q1")
    document3 = TextDocument(id="d3", text="a b c q1 d q2 e q2 f q1")
    document4 = TextDocument(id="d4", text="a b c d  q1 q2")
    document5 = TextDocument(id="d5", text="a b c q1 q1 q2")

    axiom = PROX4()

    # Document d2 contains a closer grouping.
    assert axiom.preference(query, document1, document2) == -1
    assert axiom.preference(query, document2, document1) == 1

    # Document d3 contains an equally close grouping more often.
    assert axiom.preference(query, document1, document3) == -1
    assert axiom.preference(query, document3, document1) == 1

    # Document d5 contains an additional zero-gap grouping
    # via repeated query terms.
    assert axiom.preference(query, document4, document5) == -1
    assert axiom.preference(query, document5, document4) == 1


def test_prox5() -> None:
    query = TextQuery(id="q1", text="q1 q2 q3")
    document1 = TextDocument(id="d1", text="q1 q2 q3")
    document2 = TextDocument(id="d2", text="q1 a q2 b c q3")

    axiom = PROX5()

    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1
