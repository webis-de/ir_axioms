from pytest import fixture

from ir_axioms.axiom import (
    Axiom, ArgUC, QTArg, QTPArg, aSL, PROX1, PROX2, PROX3, PROX4, PROX5, TFC1,
    TFC3, AND, LEN_AND, M_AND, LEN_M_AND, DIV, LEN_DIV, M_TDC, LEN_M_TDC,
    STMC1, STMC1_f, STMC2, STMC2_f, LNC1, TF_LNC, LB1, REG, ANTI_REG,
    ASPECT_REG, REG_f, ANTI_REG_f, ASPECT_REG_f
)
from ir_axioms.model import Query, RankedTextDocument
from tests.unit.util import MemoryIndexContext


@fixture(params=[
    ArgUC(),
    QTArg(),
    QTPArg(),
    aSL(),
    LNC1(),
    TF_LNC(),
    LB1(),
    PROX1(),
    PROX2(),
    PROX3(),
    PROX4(),
    PROX5(),
    REG(),
    REG_f(),
    ANTI_REG(),
    ANTI_REG_f(),
    ASPECT_REG(),
    ASPECT_REG_f(),
    AND(),
    LEN_AND(),
    M_AND(),
    LEN_M_AND(),
    DIV(),
    LEN_DIV(),
    TFC1(),
    TFC3(),
    M_TDC(),
    LEN_M_TDC(),
    STMC1(),
    STMC1_f(),
    STMC2(),
    STMC2_f(),
])
def axiom(request) -> Axiom:
    return request.param


def test_empty_query(axiom: Axiom):
    query = Query("")
    document1 = RankedTextDocument("d1", 2, 1, "w1 w2 w3")
    document2 = RankedTextDocument("d2", 1, 2, "w1 w2 w3")
    context = MemoryIndexContext({document1, document2})

    assert axiom.preference(context, query, document1, document2) == 0
    assert axiom.preference(context, query, document2, document1) == 0


def test_empty_document1(axiom: Axiom):
    query = Query("w1 w2 w3")
    document1 = RankedTextDocument("d1", 2, 1, "")
    document2 = RankedTextDocument("d2", 1, 2, "w1 w2 w3")
    context = MemoryIndexContext({document1, document2})

    assert (
            axiom.preference(context, query, document1, document2) ==
            -axiom.preference(context, query, document2, document1)
    )


def test_empty_document2(axiom: Axiom):
    query = Query("w1 w2 w3")
    document1 = RankedTextDocument("d1", 2, 1, "w1 w2 w3")
    document2 = RankedTextDocument("d2", 1, 2, "")
    context = MemoryIndexContext({document1, document2})

    assert (
            axiom.preference(context, query, document1, document2) ==
            -axiom.preference(context, query, document2, document1)
    )


def test_empty_documents(axiom: Axiom):
    query = Query("w1 w2 w3")
    document1 = RankedTextDocument("d1", 2, 1, "")
    document2 = RankedTextDocument("d2", 1, 2, "")
    context = MemoryIndexContext({document1, document2})

    assert axiom.preference(context, query, document1, document2) == 0
    assert axiom.preference(context, query, document2, document1) == 0
