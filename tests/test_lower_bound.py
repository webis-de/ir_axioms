from ir_axioms.axiom import LB1
from ir_axioms.model import TextQuery, ScoredTextDocument
from tests.util import inject_documents


def test_lb1():
    query = TextQuery("q1", "test query words")
    document1 = ScoredTextDocument(
        "d1", "test document that contains query words and phrases", 1.00
    )
    document2 = ScoredTextDocument(
        "d2", "test document that contains words and phrases", 1.01
    )

    inject_documents([document1, document2])

    ax1 = LB1()

    # Prefer document that contains a query term ('query')
    # which the other document doesn't contain.
    assert ax1.preference(query, document1, document2) == 1
    assert ax1.preference(query, document2, document1) == -1
