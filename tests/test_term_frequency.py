from axioms.axiom import TFC1, TFC3, M_TDC, ModifiedTdcAxiom
from axioms.model import Query, RankedTextDocument
from axioms.model.retrieval import set_index_context
from axioms.precondition import LenPrecondition
from tests.unit.util import MemoryIndexContext


def test_tfc1():
    query = Query("w1 w2")
    document1 = RankedTextDocument("d1", 3, 1, "w1 w1 w2 w3")
    document2 = RankedTextDocument("d2", 2, 2, "w1 w2 w1 w1")
    document3 = RankedTextDocument("d2", 1, 3, "w1 w2 w1 w1")
    context = MemoryIndexContext({document1, document2, document3})
    set_index_context(context)

    axiom = TFC1

    assert axiom.preference(query, document1, document2) == -1
    assert axiom.preference(query, document2, document1) == 1

    assert axiom.preference(query, document1, document3) == -1
    assert axiom.preference(query, document3, document1) == 1

    assert axiom.preference(query, document2, document3) == 0
    assert axiom.preference(query, document3, document2) == 0


def test_tfc3():
    query = Query("w1 w2 w3")
    document1 = RankedTextDocument("d1", 3, 1, "w1 w2 w2")
    document2 = RankedTextDocument("d2", 2, 2, "w2 w3 w1")
    document3 = RankedTextDocument("d3", 1, 3, "w3 w1 w1")
    context = MemoryIndexContext({document1, document2, document3})
    set_index_context(context)

    axiom = TFC3

    # Given query term pairs [(w2, w3)] that have approximately
    # the same term discriminator, i.e., rounded IDF,
    # prefer the document where the first query term frequency within
    # is more often the sum of both term's frequencies in the other document.
    assert axiom.preference(query, document1, document2) == -1
    assert axiom.preference(query, document2, document1) == 1


def test_m_tdc():
    query = Query("test query words phrases")
    document1 = RankedTextDocument(
        "d1", 2, 1, "this is the test document and contains words and phrases"
    )
    document2 = RankedTextDocument(
        "d2", 1, 2, "another document contains query words but is not very words"
    )
    context = MemoryIndexContext({document1, document2})
    set_index_context(context)

    axiom = M_TDC

    # Prefer the document with more discriminative terms.
    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1


def test_len_m_tdc_false_precondition():
    query = Query("test query words phrases")
    document1 = RankedTextDocument(
        "d1", 2, 1, "this is the test document and contains words and phrases a b c d"
    )
    document2 = RankedTextDocument(
        "d2", 1, 2, "another document contains query words but is not very words"
    )
    context = MemoryIndexContext({document1, document2})
    set_index_context(context)

    axiom = ModifiedTdcAxiom(
        context=context,
    ).with_precondition(
        LenPrecondition(
            context=context,
            margin_fraction=0.1,
        )
    )

    # Precondition is not met.
    assert axiom.preference(query, document1, document2) == 0
    assert axiom.preference(query, document2, document1) == 0


def test_len_m_tdc():
    query = Query("test query words phrases")
    document1 = RankedTextDocument(
        "d1", 2, 1, "this is the test document and contains words and phrases"
    )
    document2 = RankedTextDocument(
        "d2", 1, 2, "another document contains query words but is not very words"
    )
    context = MemoryIndexContext({document1, document2})
    set_index_context(context)

    axiom = ModifiedTdcAxiom(
        context=context,
    ).with_precondition(
        LenPrecondition(
            context=context,
            margin_fraction=0.3,
        )
    )

    # Prefer the document with more discriminative terms.
    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1
