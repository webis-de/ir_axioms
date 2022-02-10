from ir_axioms.axiom import (
    LEN_AND, LEN_M_AND, REG, ANTI_REG, AND, M_AND, DIV, LEN_DIV
)
from ir_axioms.model import Query, RankedTextDocument
from tests.unit.util import MemoryIndexContext


def test_reg():
    query = Query("child human apple")
    document1 = RankedTextDocument("d1", 2, 1, "child human apple human apple")
    document2 = RankedTextDocument("d2", 1, 2, "child human apple child")
    context = MemoryIndexContext({document1, document2})

    axiom = REG()

    # Prefer document with higher term frequency of the term 'apple'
    # that has the least similarity with other query terms.
    assert axiom.preference(context, query, document1, document2) == 1
    assert axiom.preference(context, query, document2, document1) == -1


def test_anti_reg():
    query = Query("child human apple")
    document1 = RankedTextDocument("d1", 2, 1, "child human apple child")
    document2 = RankedTextDocument("d2", 1, 2, "child human apple human apple")
    context = MemoryIndexContext({document2, document1})

    axiom = ANTI_REG()

    # Prefer document with higher term frequency of the term 'child'
    # that has the most similarity with other query terms.
    assert axiom.preference(context, query, document1, document2) == 1
    assert axiom.preference(context, query, document2, document1) == -1


def test_and():
    query = Query("q1 q2 apple")
    document1 = RankedTextDocument("d1", 2, 1, "apple b q1 q2 q2 q2 q1 q1 q2")
    document2 = RankedTextDocument("d2", 1, 2, "a b q1 q2 q2 q2 q1 q1 q2")
    context = MemoryIndexContext({document1, document2})

    axiom = AND()

    # Prefer the document that contains all query terms.
    assert axiom.preference(context, query, document1, document2) == 1
    assert axiom.preference(context, query, document2, document1) == -1


def test_and_no_winner():
    query = Query("q1 q2 q3")
    document1 = RankedTextDocument("d1", 2, 1, "q1 q1 q1 q2")
    document2 = RankedTextDocument("d2", 1, 2, "q1 q1 q1 q1")
    context = MemoryIndexContext({document1, document2})

    axiom = AND()

    # Neither of the documents contain all query terms.
    assert axiom.preference(context, query, document1, document2) == 0
    assert axiom.preference(context, query, document2, document1) == 0


def test_m_and():
    query = Query("q1 q2 q3")
    document1 = RankedTextDocument("d1", 2, 1, "q3 b q1 q2 q2 q2 q1 q1 q2")
    document2 = RankedTextDocument("d2", 1, 2, "a b q1 q2 q2 q2 q1 q1 q2")
    context = MemoryIndexContext({document1, document2})

    axiom = M_AND()

    # Prefer the document that contains more query terms.
    assert axiom.preference(context, query, document1, document2) == 1
    assert axiom.preference(context, query, document2, document1) == -1


def test_m_and_not_all_query_terms():
    query = Query("q1 q2 q3")
    document1 = RankedTextDocument("d1", 2, 1, "q1 q1 q1 q2")
    document2 = RankedTextDocument("d2", 1, 2, "q1 q1 q1 q1")
    context = MemoryIndexContext({document1, document2})

    axiom = M_AND()

    # Prefer the document that contains more query terms,
    # even if it doesn't contain all.
    assert axiom.preference(context, query, document1, document2) == 1
    assert axiom.preference(context, query, document2, document1) == -1


def test_len_and_false_precondition():
    query = Query("c a")
    document1 = RankedTextDocument("d1", 2, 1, "b c")
    document2 = RankedTextDocument("d2", 1, 2, "a c b")
    context = MemoryIndexContext({document1, document2})

    axiom = LEN_AND(0.3)

    assert axiom.preference(context, query, document1, document2) == 0
    assert axiom.preference(context, query, document2, document1) == 0


def test_len_and():
    query = Query("e b")
    document1 = RankedTextDocument("d1", 2, 1, "b e")
    document2 = RankedTextDocument("d2", 1, 2, "a c b")
    context = MemoryIndexContext({document1, document2})

    axiom = LEN_AND(0.4)

    assert axiom.preference(context, query, document1, document2) == 1
    assert axiom.preference(context, query, document2, document1) == -1


def test_len_m_and_false_precondition():
    query = Query("q1 q2 q3")
    document1 = RankedTextDocument("d1", 2, 1, "q3 b q1 q2 q2 q2 q1 q1 q2")
    document2 = RankedTextDocument("d2", 1, 2, "a b a b q1 q2 q2 q2 q1 q1 q2")
    context = MemoryIndexContext({document1, document2})

    axiom = LEN_M_AND(0.1)

    assert axiom.preference(context, query, document1, document2) == 0
    assert axiom.preference(context, query, document2, document1) == 0


def test_len_m_and_false_precondition_no_winner():
    query = Query("q1 q2 q3")
    document1 = RankedTextDocument("d1", 2, 1, "q1 q1 q1 q1")
    document2 = RankedTextDocument("d2", 1, 2, "a b c q1 q1 q1 q2")
    context = MemoryIndexContext({document1, document2})

    axiom = LEN_M_AND(0.1)

    assert axiom.preference(context, query, document1, document2) == 0
    assert axiom.preference(context, query, document2, document1) == 0


def test_len_m_and():
    query = Query("q1 q2 q3")
    document1 = RankedTextDocument("d1", 2, 1, "q3 b q1 q2 q2 q2 q1 q1 q2")
    document2 = RankedTextDocument("d2", 1, 2, "a b a b q1 q2 q2 q2 q1 q1 q2")
    context = MemoryIndexContext({document1, document2})

    axiom = LEN_M_AND(0.3)

    assert axiom.preference(context, query, document1, document2) == 1
    assert axiom.preference(context, query, document2, document1) == -1


def test_len_m_and_no_winner():
    query = Query("q1 q2 q3")
    document1 = RankedTextDocument("d1", 2, 1, "q1 q1 q1 q1")
    document2 = RankedTextDocument("d2", 1, 2, "a b c q1 q1 q1 q2")
    context = MemoryIndexContext({document1, document2})

    axiom = LEN_M_AND(0.3)

    assert axiom.preference(context, query, document1, document2) == 0
    assert axiom.preference(context, query, document2, document1) == 0


def test_div():
    query = Query("q1 q2 q3")
    document1 = RankedTextDocument("d1", 2, 1, "foo bar baz")
    document2 = RankedTextDocument("d2", 1, 2, "q1 q2 q3")
    context = MemoryIndexContext({document1, document2})

    axiom = DIV()

    assert axiom.preference(context, query, document1, document2) == 1
    assert axiom.preference(context, query, document2, document1) == -1


def test_len_div_false_precondition():
    query = Query("q1 q2 q3")
    document1 = RankedTextDocument("d1", 2, 1, "q1 q2 q3")
    document2 = RankedTextDocument("d2", 1, 2, "foo bar baz bab bac")
    context = MemoryIndexContext({document1, document2})

    axiom = LEN_DIV(0.1)

    assert axiom.preference(context, query, document1, document2) == 0
    assert axiom.preference(context, query, document2, document1) == 0


def test_len_div():
    query = Query("q1 q2 q3")
    document1 = RankedTextDocument("d1", 2, 1, "foo bar baz bab bac")
    document2 = RankedTextDocument("d2", 1, 2, "q1 q2 q3")
    context = MemoryIndexContext({document1, document2})

    axiom = LEN_DIV(0.5)

    assert axiom.preference(context, query, document1, document2) == 1
    assert axiom.preference(context, query, document2, document1) == -1
