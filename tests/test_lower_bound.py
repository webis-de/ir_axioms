from axioms.axiom import LB1
from axioms.model import Query, RankedTextDocument
from axioms.model.retrieval import set_index_context
from tests.unit.util import MemoryIndexContext


def test_lb1():
    query = Query("test query words")
    document1 = RankedTextDocument(
        "d1", 1.00, 2, "test document that contains query words and phrases"
    )
    document2 = RankedTextDocument(
        "d2", 1.01, 1, "test document that contains words and phrases"
    )
    context = MemoryIndexContext({document2, document1})
    set_index_context(context)

    ax1 = LB1

    # Prefer document that contains a query term ('query')
    # which the other document doesn't contain.
    assert ax1.preference(query, document1, document2) == 1
    assert ax1.preference(query, document2, document1) == -1
