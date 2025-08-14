from ir_axioms.axiom import NOP, ORIG, ORACLE
from ir_axioms.model import Query, Document


def test_nop() -> None:
    query = Query(id="q1 q2 q3")
    document1 = Document(id="d1")
    document2 = Document(id="d2")

    axiom = NOP()

    assert axiom.preference(query, document1, document2) == 0
    assert axiom.preference(query, document2, document1) == 0


def test_original_rank() -> None:
    query = Query(id="q1 q2 q3")
    document1 = Document(id="d1", rank=1)
    document2 = Document(id="d2", rank=2)

    axiom = ORIG()

    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1


def test_original_score() -> None:
    query = Query(id="q1 q2 q3")
    document1 = Document(id="d1", score=2)
    document2 = Document(id="d2", score=1)

    axiom = ORIG()

    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1


def test_oracle() -> None:
    query = Query(id="q1 q2 q3")
    document1 = Document(id="d1", relevance=1)
    document2 = Document(id="d2", relevance=0)

    axiom = ORACLE()

    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1
