from ir_axioms.axiom import LEN_M_TDC, TFC1, TFC3, M_TDC
from ir_axioms.model import TextQuery, TextDocument
from ir_axioms.precondition import LEN
from tests.util import inject_documents


def test_tfc1() -> None:
    query = TextQuery(id="q1", text="w1 w2")
    document1 = TextDocument(id="d1", text="w1 w1 w2 w3")
    document2 = TextDocument(id="d2", text="w1 w2 w1 w1")
    document3 = TextDocument(id="d3", text="w1 w2 w1 w1")

    axiom = TFC1()

    assert axiom.preference(query, document1, document2) == -1
    assert axiom.preference(query, document2, document1) == 1

    assert axiom.preference(query, document1, document3) == -1
    assert axiom.preference(query, document3, document1) == 1

    assert axiom.preference(query, document2, document3) == 0
    assert axiom.preference(query, document3, document2) == 0


def test_tfc3() -> None:
    query = TextQuery(id="q1", text="w1 w2 w3")
    document1 = TextDocument(id="d1", text="w1 w2 w2")
    document2 = TextDocument(id="d2", text="w2 w3 w1")
    document3 = TextDocument(id="d3", text="w3 w1 w1")

    inject_documents([document1, document2, document3])

    axiom = TFC3()

    # Given query term pairs [(w2, w3)] that have approximately
    # the same term discriminator, i.e., rounded IDF,
    # prefer the document where the first query term frequency within
    # is more often the sum of both term's frequencies in the other document.
    assert axiom.preference(query, document1, document2) == -1
    assert axiom.preference(query, document2, document1) == 1


def test_m_tdc() -> None:
    query = TextQuery(id="q1", text="test query words phrases")
    document1 = TextDocument(
        id="d1",
        text="this is the test document and contains words and phrases",
    )
    document2 = TextDocument(
        id="d2",
        text="another document contains query words but is not very words",
    )

    inject_documents([document1, document2])

    axiom = M_TDC()

    # Prefer the document with more discriminative terms.
    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1


def test_len_m_tdc_false_precondition() -> None:
    query = TextQuery(id="q1", text="test query words phrases")
    document1 = TextDocument(
        id="d1",
        text="this is the test document and contains words and phrases a b c d",
    )
    document2 = TextDocument(
        id="d2",
        text="another document contains query words but is not very words",
    )

    inject_documents([document1, document2])

    precondition = LEN(margin_fraction=0.1)
    axiom = LEN_M_TDC(precondition=precondition)

    # Precondition is not met.
    assert axiom.preference(query, document1, document2) == 0
    assert axiom.preference(query, document2, document1) == 0


def test_len_m_tdc() -> None:
    query = TextQuery("q1", "test query words phrases")
    document1 = TextDocument(
        id="d1",
        text="this is the test document and contains words and phrases",
    )
    document2 = TextDocument(
        id="d2",
        text="another document contains query words but is not very words",
    )

    inject_documents([document1, document2])

    precondition = LEN(margin_fraction=0.3)
    axiom = LEN_M_TDC(precondition=precondition)

    # Prefer the document with more discriminative terms.
    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1
