from ir_axioms.axiom.term_frequency import TFC1
from ir_axioms.model import Query, RankedDocument
from tests.util import SimpleRerankingContext


def test_tfc1():
    query = Query("w1 w2")
    document1 = RankedDocument(
        "d1", "w1 w1 w2 w3",
        score=0.75, rank=1)
    document2 = RankedDocument(
        "d2", "w1 w2 w1 w1",
        score=0.5, rank=2
    )
    document3 = RankedDocument(
        "d2", "w1 w2 w1 w1",
        score=0.25, rank=3
    )
    context = SimpleRerankingContext([document1, document2, document3])

    axiom = TFC1()

    assert axiom.preference(context, query, document1, document2) == -1
    assert axiom.preference(context, query, document2, document1) == 1

    assert axiom.preference(context, query, document1, document3) == -1
    assert axiom.preference(context, query, document3, document1) == 1

    assert axiom.preference(context, query, document2, document3) == 0
    assert axiom.preference(context, query, document3, document2) == 0


if __name__ == '__main__':
    test_tfc1()