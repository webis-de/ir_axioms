from ir_axioms.axiom import STMC1, STMC2
from ir_axioms.model import Query, RankedTextDocument
from tests.unit.util import MemoryIndexContext


def test_stmc1():
    query = Query("blue car moves")
    document1 = RankedTextDocument(
        "d1", 2, 1,
        "blue auto goes through the city"
    )
    document2 = RankedTextDocument(
        "d2", 1, 2,
        "red airplane flies in the sky"
    )
    context = MemoryIndexContext({document1, document2})

    axiom = STMC1()

    # Document d1 contains a more similar term.
    assert axiom.preference(context, query, document1, document2) == 1
    assert axiom.preference(context, query, document2, document1) == -1


def test_stmc2():
    q = Query("q")
    document1 = RankedTextDocument("d1", 2, 1, "q")
    document2 = RankedTextDocument("d2", 1, 2, "t t t t")
    context = MemoryIndexContext({document1, document2})

    axiom = STMC2()

    # Document d1 contains an exact match.
    assert axiom.preference(context, q, document1, document2) == 1
    assert axiom.preference(context, q, document2, document1) == -1


def test_stmc2_equal():
    q = Query("dog breed")
    document1 = RankedTextDocument(
        "d1", 3, 1,
        "dog fire orange"
    )
    document2 = RankedTextDocument(
        "d2", 2, 2,
        "dog animal animal animal time key"
    )
    document3 = RankedTextDocument(
        "d3", 1, 3,
        "dog animal time key"
    )
    context = MemoryIndexContext({document1, document2, document3})

    axiom = STMC2()

    # Most similar query term 'dog' and non-query term 'animal'.
    # The document 2 non-query term frequency (0.5)
    # compared to the document 1 query term frequency (0.333)
    # is similar to the document 2 term set length (4)
    # compared to the document 1 term set length (3):
    # 1.333 ≈ 1.5
    assert axiom.preference(context, q, document1, document2) == 1
    assert axiom.preference(context, q, document2, document1) == -1

    # Most similar query term 'dog' and non-query term 'animal'.
    # The document 3 non-query term frequency (0.25)
    # compared to the document 1 query term frequency (0.333)
    # is not similar to the document 3 term set length (4)
    # compared to the document 1 term set length (3):
    # 1.333 != 0.75
    assert axiom.preference(context, q, document1, document3) == 0
    assert axiom.preference(context, q, document3, document1) == 0
