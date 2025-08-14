from ir_axioms.axiom import LNC1, TF_LNC
from ir_axioms.model import Query, Document


def test_lnc1() -> None:
    query = Query(id="q1", text="q1 q2 q3")
    document1 = Document(id="d1", text="q1 q2 q3 w w w w w w w")
    document2 = Document(id="d2", text="q1 q2 q3 w w w w w w w w")

    axiom = LNC1()

    # Prefer the shorter document.
    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1


def test_tf_lnc() -> None:
    query = Query(id="q1", text="q1 q2 q3")
    document1 = Document(id="d1", text="q1 q1 q2 x y")
    document2 = Document(id="d2", text="q1 q2 x y")
    document3 = Document(id="d3", text="q1 q1 q1 x y")

    axiom = TF_LNC()

    # Prefer document with higher query term frequency
    # while document length without the query term is equal.
    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1
    assert axiom.preference(query, document1, document3) == 0
    assert axiom.preference(query, document3, document1) == 0
