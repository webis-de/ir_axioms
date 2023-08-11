from ir_axioms.axiom import LNC1, TF_LNC
from ir_axioms.model import Query, RankedTextDocument
from tests.unit.util import MemoryIndexContext


def test_lnc1():
    query = Query("q1 q2 q3")
    document1 = RankedTextDocument("d1", 2, 1, "q1 q2 q3 w w w w w w w")
    document2 = RankedTextDocument("d2", 1, 2, "q1 q2 q3 w w w w w w w w")
    context = MemoryIndexContext({document1, document2})

    axiom = LNC1()

    # Prefer the shorter document.
    assert axiom.preference(context, query, document1, document2) == 1
    assert axiom.preference(context, query, document2, document1) == -1


def test_tf_lnc():
    query = Query("q1 q2 q3")

    document1 = RankedTextDocument("d1", 3, 1, "q1 q1 q2 x y")
    document2 = RankedTextDocument("d2", 2, 2, "q1 q2 x y")
    document3 = RankedTextDocument("d3", 1, 3, "q1 q1 q1 x y")
    context = MemoryIndexContext({document1, document2, document3})

    axiom = TF_LNC()

    # Prefer document with higher query term frequency
    # while document length without the query term is equal.
    assert axiom.preference(context, query, document1, document2) == 1
    assert axiom.preference(context, query, document2, document1) == -1
    assert axiom.preference(context, query, document1, document3) == 0
    assert axiom.preference(context, query, document3, document1) == 0
