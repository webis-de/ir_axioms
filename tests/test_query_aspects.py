from ir_axioms.axiom import (
    LEN_AND,
    LEN_DIV,
    LEN_M_AND,
    REG,
    ANTI_REG,
    AND,
    M_AND,
    DIV,
)
from ir_axioms.model import TextQuery, TextDocument
from ir_axioms.precondition import LEN


def test_reg() -> None:
    query = TextQuery(id="q1", text="child human apple")
    document1 = TextDocument(id="d1", text="child human apple human apple")
    document2 = TextDocument(id="d2", text="child human apple child")

    axiom = REG()

    # Prefer document with higher term frequency of the term 'apple'
    # that has the least similarity with other query terms.
    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1


def test_anti_reg() -> None:
    query = TextQuery(id="q1", text="child human apple")
    document1 = TextDocument(id="d1", text="child human apple child")
    document2 = TextDocument(id="d2", text="child human apple human apple")

    axiom = ANTI_REG()

    # Prefer document with higher term frequency of the term 'child'
    # that has the most similarity with other query terms.
    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1


def test_and() -> None:
    query = TextQuery(id="q1", text="q1 q2 apple")
    document1 = TextDocument(id="d1", text="apple b q1 q2 q2 q2 q1 q1 q2")
    document2 = TextDocument(id="d2", text="a b q1 q2 q2 q2 q1 q1 q2")

    axiom = AND()

    # Prefer the document that contains all query terms.
    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1


def test_and_no_winner() -> None:
    query = TextQuery(id="q1", text="q1 q2 q3")
    document1 = TextDocument(id="d1", text="q1 q1 q1 q2")
    document2 = TextDocument(id="d2", text="q1 q1 q1 q1")

    axiom = AND()

    # Neither of the documents contain all query terms.
    assert axiom.preference(query, document1, document2) == 0
    assert axiom.preference(query, document2, document1) == 0


def test_m_and() -> None:
    query = TextQuery(id="q1", text="q1 q2 q3")
    document1 = TextDocument(id="d1", text="q3 b q1 q2 q2 q2 q1 q1 q2")
    document2 = TextDocument(id="d2", text="a b q1 q2 q2 q2 q1 q1 q2")

    axiom = M_AND()

    # Prefer the document that contains more query terms.
    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1


def test_m_and_not_all_query_terms() -> None:
    query = TextQuery(id="q1", text="q1 q2 q3")
    document1 = TextDocument(id="d1", text="q1 q1 q1 q2")
    document2 = TextDocument(id="d2", text="q1 q1 q1 q1")

    axiom = M_AND()

    # Prefer the document that contains more query terms,
    # even if it doesn't contain all.
    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1


def test_len_and_false_precondition() -> None:
    query = TextQuery(id="q1", text="c a")
    document1 = TextDocument(id="d1", text="b c")
    document2 = TextDocument(id="d2", text="a c b")

    precondition = LEN(margin_fraction=0.3)
    axiom = LEN_AND(precondition=precondition)

    assert axiom.preference(query, document1, document2) == 0
    assert axiom.preference(query, document2, document1) == 0


def test_len_and() -> None:
    query = TextQuery(id="q1", text="e b")
    document1 = TextDocument(id="d1", text="b e")
    document2 = TextDocument(id="d2", text="a c b")

    precondition = LEN(margin_fraction=0.4)
    axiom = LEN_AND(precondition=precondition)

    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1


def test_len_m_and_false_precondition() -> None:
    query = TextQuery(id="q1", text="q1 q2 q3")
    document1 = TextDocument(id="d1", text="q3 bar q1 q2 q2 q2 q1 q1 q2")
    document2 = TextDocument(id="d2", text="foo bar foo bar q1 q2 q2 q2 q1 q1 q2")

    precondition = LEN(margin_fraction=0.1)
    axiom = LEN_M_AND(precondition=precondition)

    assert axiom.preference(query, document1, document2) == 0
    assert axiom.preference(query, document2, document1) == 0


def test_len_m_and_false_precondition_no_winner() -> None:
    query = TextQuery(id="q1", text="q1 q2 q3")
    document1 = TextDocument(id="d1", text="q1 q1 q1 q1")
    document2 = TextDocument(id="d2", text="a b c q1 q1 q1 q2")

    precondition = LEN(margin_fraction=0.1)
    axiom = LEN_M_AND(precondition=precondition)

    assert axiom.preference(query, document1, document2) == 0
    assert axiom.preference(query, document2, document1) == 0


def test_len_m_and() -> None:
    query = TextQuery(id="q1", text="q1 q2 q3")
    document1 = TextDocument(id="d1", text="q3 b q1 q2 q2 q2 q1 q1 q2")
    document2 = TextDocument(id="d2", text="a b a b q1 q2 q2 q2 q1 q1 q2")

    precondition = LEN(margin_fraction=0.3)
    axiom = LEN_M_AND(precondition=precondition)

    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1


def test_len_m_and_no_winner() -> None:
    query = TextQuery(id="q1", text="q1 q2 q3")
    document1 = TextDocument(id="d1", text="q1 q1 q1 q1")
    document2 = TextDocument(id="d2", text="a b c q1 q1 q1 q2")

    precondition = LEN(margin_fraction=0.3)
    axiom = LEN_M_AND(precondition=precondition)

    assert axiom.preference(query, document1, document2) == 0
    assert axiom.preference(query, document2, document1) == 0


def test_div() -> None:
    query = TextQuery(id="q1", text="q1 q2 q3")
    document1 = TextDocument(id="d1", text="foo bar baz")
    document2 = TextDocument(id="d2", text="q1 q2 q3")

    axiom = DIV()

    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1


def test_len_div_false_precondition() -> None:
    query = TextQuery(id="q1", text="q1 q2 q3")
    document1 = TextDocument(id="d1", text="q1 q2 q3")
    document2 = TextDocument(id="d2", text="foo bar baz bab bac")

    precondition = LEN(margin_fraction=0.1)
    axiom = LEN_DIV(precondition=precondition)

    assert axiom.preference(query, document1, document2) == 0
    assert axiom.preference(query, document2, document1) == 0


def test_len_div() -> None:
    query = TextQuery(id="q1", text="q1 q2 q3")
    document1 = TextDocument(id="d1", text="foo bar baz bab bac")
    document2 = TextDocument(id="d2", text="q1 q2 q3")

    precondition = LEN(margin_fraction=0.5)
    axiom = LEN_DIV(precondition=precondition)

    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1
