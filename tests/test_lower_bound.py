from ir_axioms.axiom import LB1
from ir_axioms.model import Query, Document


def test_lb1() -> None:
    query = Query(id="q1", text="test query words")
    document1 = Document(
        id="d1",
        text="test document that contains query words and phrases",
        score=1.00,
    )
    document2 = Document(
        id="d2",
        text="test document that contains words and phrases",
        score=1.01,
    )

    ax1 = LB1()

    # Prefer document that contains a query term ('query')
    # which the other document doesn't contain.
    assert ax1.preference(query, document1, document2) == 1
    assert ax1.preference(query, document2, document1) == -1
