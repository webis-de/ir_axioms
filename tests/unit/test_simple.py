from ir_axioms.axiom import NopAxiom, OriginalAxiom
from ir_axioms.model import Query, RankedTextDocument
from tests.unit.util import MemoryIndexContext


def test_nop():
    query = Query("q1 q2 q3")
    document1 = RankedTextDocument("d1", 2, 1, "w1 w2 w3")
    document2 = RankedTextDocument("d2", 1, 2, "w1 w2 w3")
    context = MemoryIndexContext({document1, document2})

    axiom = NopAxiom()

    assert axiom.preference(context, query, document1, document2) == 0
    assert axiom.preference(context, query, document2, document1) == 0


def test_original():
    query = Query("q1 q2 q3")
    document1 = RankedTextDocument("d1", 2, 1, "w1 w2 w3")
    document2 = RankedTextDocument("d2", 1, 2, "w1 w2 w3")
    context = MemoryIndexContext({document1, document2})

    axiom = OriginalAxiom()

    assert axiom.preference(context, query, document1, document2) == 1
    assert axiom.preference(context, query, document2, document1) == -1
