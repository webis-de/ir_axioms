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
from tests.util import inject_documents


def test_reg():
    query = TextQuery("q1", "child human apple")
    document1 = TextDocument("d1", "child human apple human apple")
    document2 = TextDocument("d2", "child human apple child")

    inject_documents([document1, document2])

    axiom = REG()

    # Prefer document with higher term frequency of the term 'apple'
    # that has the least similarity with other query terms.
    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1


def test_anti_reg():
    query = TextQuery("q1", "child human apple")
    document1 = TextDocument("d1", "child human apple child")
    document2 = TextDocument("d2", "child human apple human apple")

    inject_documents([document1, document2])

    axiom = ANTI_REG()

    # Prefer document with higher term frequency of the term 'child'
    # that has the most similarity with other query terms.
    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1


def test_and():
    query = TextQuery("q1", "q1 q2 apple")
    document1 = TextDocument("d1", "apple b q1 q2 q2 q2 q1 q1 q2")
    document2 = TextDocument("d2", "a b q1 q2 q2 q2 q1 q1 q2")

    inject_documents([document1, document2])

    axiom = AND()

    # Prefer the document that contains all query terms.
    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1


def test_and_no_winner():
    query = TextQuery("q1", "q1 q2 q3")
    document1 = TextDocument("d1", "q1 q1 q1 q2")
    document2 = TextDocument("d2", "q1 q1 q1 q1")

    inject_documents([document1, document2])

    axiom = AND()

    # Neither of the documents contain all query terms.
    assert axiom.preference(query, document1, document2) == 0
    assert axiom.preference(query, document2, document1) == 0


def test_m_and():
    query = TextQuery("q1", "q1 q2 q3")
    document1 = TextDocument("d1", "q3 b q1 q2 q2 q2 q1 q1 q2")
    document2 = TextDocument("d2", "a b q1 q2 q2 q2 q1 q1 q2")

    inject_documents([document1, document2])

    axiom = M_AND()

    # Prefer the document that contains more query terms.
    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1


def test_m_and_not_all_query_terms():
    query = TextQuery("q1", "q1 q2 q3")
    document1 = TextDocument("d1", "q1 q1 q1 q2")
    document2 = TextDocument("d2", "q1 q1 q1 q1")

    inject_documents([document1, document2])

    axiom = M_AND()

    # Prefer the document that contains more query terms,
    # even if it doesn't contain all.
    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1


def test_len_and_false_precondition():
    query = TextQuery("q1", "c a")
    document1 = TextDocument("d1", "b c")
    document2 = TextDocument("d2", "a c b")

    inject_documents([document1, document2])

    precondition = LEN(margin_fraction=0.3)
    axiom = LEN_AND(precondition=precondition)

    assert axiom.preference(query, document1, document2) == 0
    assert axiom.preference(query, document2, document1) == 0


def test_len_and():
    query = TextQuery("q1", "e b")
    document1 = TextDocument("d1", "b e")
    document2 = TextDocument("d2", "a c b")

    inject_documents([document1, document2])

    precondition = LEN(margin_fraction=0.4)
    axiom = LEN_AND(precondition=precondition)

    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1


def test_len_m_and_false_precondition():
    query = TextQuery("q1", "q1 q2 q3")
    document1 = TextDocument("d1", "q3 b q1 q2 q2 q2 q1 q1 q2")
    document2 = TextDocument("d2", "a b a b q1 q2 q2 q2 q1 q1 q2")

    inject_documents([document1, document2])

    precondition = LEN(margin_fraction=0.1)
    axiom = LEN_M_AND(precondition=precondition)

    assert axiom.preference(query, document1, document2) == 0
    assert axiom.preference(query, document2, document1) == 0


def test_len_m_and_false_precondition_no_winner():
    query = TextQuery("q1", "q1 q2 q3")
    document1 = TextDocument("d1", "q1 q1 q1 q1")
    document2 = TextDocument("d2", "a b c q1 q1 q1 q2")

    inject_documents([document1, document2])

    precondition = LEN(margin_fraction=0.1)
    axiom = LEN_M_AND(precondition=precondition)

    assert axiom.preference(query, document1, document2) == 0
    assert axiom.preference(query, document2, document1) == 0


def test_len_m_and():
    query = TextQuery("q1", "q1 q2 q3")
    document1 = TextDocument("d1", "q3 b q1 q2 q2 q2 q1 q1 q2")
    document2 = TextDocument("d2", "a b a b q1 q2 q2 q2 q1 q1 q2")

    inject_documents([document1, document2])

    precondition = LEN(margin_fraction=0.3)
    axiom = LEN_M_AND(precondition=precondition)

    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1


def test_len_m_and_no_winner():
    query = TextQuery("q1", "q1 q2 q3")
    document1 = TextDocument("d1", "q1 q1 q1 q1")
    document2 = TextDocument("d2", "a b c q1 q1 q1 q2")

    inject_documents([document1, document2])

    precondition = LEN(margin_fraction=0.3)
    axiom = LEN_M_AND(precondition=precondition)

    assert axiom.preference(query, document1, document2) == 0
    assert axiom.preference(query, document2, document1) == 0


def test_div():
    query = TextQuery("q1", "q1 q2 q3")
    document1 = TextDocument("d1", "foo bar baz")
    document2 = TextDocument("d2", "q1 q2 q3")

    inject_documents([document1, document2])

    axiom = DIV()

    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1


def test_len_div_false_precondition():
    query = TextQuery("q1", "q1 q2 q3")
    document1 = TextDocument("d1", "q1 q2 q3")
    document2 = TextDocument("d2", "foo bar baz bab bac")

    inject_documents([document1, document2])

    precondition = LEN(margin_fraction=0.1)
    axiom = LEN_DIV(precondition=precondition)

    assert axiom.preference(query, document1, document2) == 0
    assert axiom.preference(query, document2, document1) == 0


def test_len_div():
    query = TextQuery("q1", "q1 q2 q3")
    document1 = TextDocument("d1", "foo bar baz bab bac")
    document2 = TextDocument("d2", "q1 q2 q3")

    inject_documents([document1, document2])

    precondition = LEN(margin_fraction=0.5)
    axiom = LEN_DIV(precondition=precondition)

    assert axiom.preference(query, document1, document2) == 1
    assert axiom.preference(query, document2, document1) == -1
