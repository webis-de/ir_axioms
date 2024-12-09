from axioms.axiom import NOP, ORIG, ORACLE
from axioms.model import Query, Document, RankedDocument, ScoredDocument, JudgedDocument


def test_nop():
    query = Query("q1 q2 q3")
    document1 = Document("d1")
    document2 = Document("d2")

    axiom = NOP()

    assert axiom.preference(query, document1, document2) == 0
    assert axiom.preference(query, document2, document1) == 0


def test_original_rank():
    query = Query("q1 q2 q3")
    document1 = RankedDocument("d1", 1)
    document2 = RankedDocument("d2", 2)

    axiom = ORIG()

    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1


def test_original_score():
    query = Query("q1 q2 q3")
    document1 = ScoredDocument("d1", 2)
    document2 = ScoredDocument("d2", 1)

    axiom = ORIG()

    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1


def test_oracle():
    query = Query("q1 q2 q3")
    document1 = JudgedDocument("d1", 1)
    document2 = JudgedDocument("d2", 0)

    axiom = ORACLE()

    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1
