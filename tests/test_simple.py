from numpy import array

from ir_axioms.axiom import NOP, ORIG, ORACLE, DistanceAxiom, Axiom
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


def test_distance_int() -> None:
    axiom: Axiom[None, int] = DistanceAxiom()

    assert axiom.preference(None, 3, 1) == 2
    assert axiom.preference(None, 1, 3) == -2
    assert axiom.preference(None, 2, 2) == 0

    assert (
        axiom.preferences(None, [1, 3, 0])
        == array([[0, -2, 1], [2, 0, 3], [-1, -3, 0]])
    ).all()


def test_distance_float() -> None:
    axiom: Axiom[None, float] = DistanceAxiom()

    assert axiom.preference(None, 3.5, 1.0) == 2.5
    assert axiom.preference(None, 1.0, 3.5) == -2.5
    assert axiom.preference(None, 1.5, 1.5) == 0

    assert (
        axiom.preferences(None, [1.0, 3.0, 0.5])
        == array([[0, -2, 0.5], [2, 0, 2.5], [-0.5, -2.5, 0]])
    ).all()
