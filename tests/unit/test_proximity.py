from ir_axioms.axiom import PROX1, PROX2, PROX3, PROX4, PROX5
from ir_axioms.model import Query, RankedTextDocument
from tests.unit.util import MemoryIndexContext


def test_prox1():
    query = Query("blue car")
    document1 = RankedTextDocument(
        "d1", 2, 1,
        "a blue car goes through the city"
    )
    document2 = RankedTextDocument(
        "d2", 1, 2,
        "through city blue goes car goes"
    )
    context = MemoryIndexContext({document1, document2})

    axiom = PROX1()

    assert axiom.preference(context, query, document1, document2) == 1
    assert axiom.preference(context, query, document2, document1) == -1


def test_prox2():
    query = Query("q1 q2")
    document1 = RankedTextDocument("d1", 2, 1, "q1 x q2 y z a b c")
    document2 = RankedTextDocument("d2", 1, 2, "x y q1 q2")
    context = MemoryIndexContext({document1, document2})

    axiom = PROX2()

    assert axiom.preference(context, query, document1, document2) == 1
    assert axiom.preference(context, query, document2, document1) == -1


def test_prox3():
    query = Query("q1 q2")
    document1 = RankedTextDocument("d1", 3, 1, "a b c q1 d q2 e q1 q2")
    document2 = RankedTextDocument("d2", 2, 2, "a q2 b q1 q2")
    document3 = RankedTextDocument("d3", 1, 3, "q1 b q2")
    context = MemoryIndexContext({document1, document2, document3})

    axiom = PROX3()

    # Document d2 contains the query phrase earlier.
    assert axiom.preference(context, query, document1, document2) == -1
    assert axiom.preference(context, query, document2, document1) == 1
    # Document d3 does not contain the query phrase.
    assert axiom.preference(context, query, document1, document3) == 1
    assert axiom.preference(context, query, document3, document1) == -1
    assert axiom.preference(context, query, document2, document3) == 1
    assert axiom.preference(context, query, document3, document2) == -1


def test_prox4():
    query = Query("q1 q2")
    document1 = RankedTextDocument("d1", 5, 1, "a b c q1 d q2 e q1")
    document2 = RankedTextDocument("d2", 4, 2, "a q2 b q2 q1")
    document3 = RankedTextDocument("d3", 3, 3, "a b c q1 d q2 e q2 f q1")
    document4 = RankedTextDocument("d4", 2, 4, "a b c d  q1 q2")
    document5 = RankedTextDocument("d5", 1, 5, "a b c q1 q1 q2")
    context = MemoryIndexContext({
        document1, document2, document3, document4, document5
    })

    axiom = PROX4()

    # Document d2 contains a closer grouping.
    assert axiom.preference(context, query, document1, document2) == -1
    assert axiom.preference(context, query, document2, document1) == 1

    # Document d3 contains an equally close grouping more often.
    assert axiom.preference(context, query, document1, document3) == -1
    assert axiom.preference(context, query, document3, document1) == 1

    # Document d5 contains an additional zero-gap grouping
    # via repeated query terms.
    assert axiom.preference(context, query, document4, document5) == -1
    assert axiom.preference(context, query, document5, document4) == 1


def test_prox5():
    query = Query("q1 q2 q3")
    document1 = RankedTextDocument("d1", 2, 1, "q1 q2 q3")
    document2 = RankedTextDocument("d2", 1, 2, "q1 a q2 b c q3")
    context = MemoryIndexContext({document1, document2})

    axiom = PROX5()

    assert axiom.preference(context, query, document1, document2) == 1
    assert axiom.preference(context, query, document2, document1) == -1
