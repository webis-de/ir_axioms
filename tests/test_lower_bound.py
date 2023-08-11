from ir_axioms.axiom import LB1
from ir_axioms.model import Query, RankedTextDocument
from tests.unit.util import MemoryIndexContext


def test_lb1():
    query = Query("test query words")
    document1 = RankedTextDocument(
        "d1", 1.00, 2,
        "test document that contains query words and phrases"
    )
    document2 = RankedTextDocument(
        "d2", 1.01, 1,
        "test document that contains words and phrases"
    )
    context = MemoryIndexContext({document2, document1})

    ax1 = LB1()

    # Prefer document that contains a query term ('query')
    # which the other document doesn't contain.
    assert ax1.preference(context, query, document1, document2) == 1
    assert ax1.preference(context, query, document2, document1) == -1
