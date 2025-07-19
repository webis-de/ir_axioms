from ir_axioms.axiom import LNC1, TF_LNC
from ir_axioms.model import TextQuery, TextDocument
from tests.util import inject_documents


def test_lnc1():
    query = TextQuery("q1", "q1 q2 q3")
    document1 = TextDocument("d1", "q1 q2 q3 w w w w w w w")
    document2 = TextDocument("d2", "q1 q2 q3 w w w w w w w w")

    inject_documents([document1, document2])

    axiom = LNC1()

    # Prefer the shorter document.
    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1


def test_tf_lnc():
    query = TextQuery("q1", "q1 q2 q3")
    document1 = TextDocument("d1", "q1 q1 q2 x y")
    document2 = TextDocument("d2", "q1 q2 x y")
    document3 = TextDocument("d3", "q1 q1 q1 x y")

    inject_documents([document1, document2])

    axiom = TF_LNC()

    # Prefer document with higher query term frequency
    # while document length without the query term is equal.
    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1
    assert axiom.preference(query, document1, document3) == 0
    assert axiom.preference(query, document3, document1) == 0
