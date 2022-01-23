from ir_axioms.axiom.term_frequency import TFC1
from ir_axioms.model import Query, RankedTextDocument
from tests.unit.util import MemoryRerankingContext


def test_tfc1():
    query = Query("w1 w2")
    document1 = RankedTextDocument("d1", 3, 1, "w1 w1 w2 w3")
    document2 = RankedTextDocument("d2", 2, 2, "w1 w2 w1 w1")
    document3 = RankedTextDocument("d2", 1, 3, "w1 w2 w1 w1")
    context = MemoryRerankingContext({document1, document2, document3})

    axiom = TFC1()

    assert axiom.preference(context, query, document1, document2) == -1
    assert axiom.preference(context, query, document2, document1) == 1

    assert axiom.preference(context, query, document1, document3) == -1
    assert axiom.preference(context, query, document3, document1) == 1

    assert axiom.preference(context, query, document2, document3) == 0
    assert axiom.preference(context, query, document3, document2) == 0
