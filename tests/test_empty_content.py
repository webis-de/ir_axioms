from typing import Callable
from pytest import fixture, FixtureRequest

from ir_axioms.axiom import (
    Axiom,
    # NOTE: Disabled argument axiom tests for requiring internet connection.
    # ArgUC,
    # QTArg,
    # QTPArg,
    aSL,
    LNC1,
    TF_LNC,
    LB1,
    PROX1,
    PROX2,
    PROX3,
    PROX4,
    PROX5,
    REG,
    ANTI_REG,
    ASPECT_REG,
    AND,
    LEN_AND,
    M_AND,
    LEN_M_AND,
    DIV,
    LEN_DIV,
    TFC1,
    TFC3,
    M_TDC,
    LEN_M_TDC,
    STMC1,
    STMC2,
)
from ir_axioms.model import Query, Document
from tests.util import inject_documents


@fixture(
    params=[
        # NOTE: Disabled argument axiom tests for requiring internet connection.
        # ArgUC,
        # QTArg,
        # QTPArg,
        aSL,
        LNC1,
        TF_LNC,
        LB1,
        PROX1,
        PROX2,
        PROX3,
        PROX4,
        PROX5,
        REG,
        ANTI_REG,
        ASPECT_REG,
        AND,
        LEN_AND,
        M_AND,
        LEN_M_AND,
        DIV,
        LEN_DIV,
        TFC1,
        TFC3,
        M_TDC,
        LEN_M_TDC,
        STMC1,
        STMC2,
    ]
)
def axiom_fn(request: FixtureRequest) -> Callable[[], Axiom]:
    return request.param


def test_empty_query(axiom_fn: Callable[[], Axiom]) -> None:
    query = Query(id="q1", text="")
    document1 = Document(id="d1", text="w1 w2 w3", score=2)
    document2 = Document(id="d2", text="w1 w2 w3", score=1)

    inject_documents([document1, document2])

    axiom = axiom_fn()
    assert axiom.preference(query, document1, document2) == 0
    assert axiom.preference(query, document2, document1) == 0


def test_empty_document1(axiom_fn: Callable[[], Axiom]) -> None:
    query = Query(id="q1", text="w1 w2 w3")
    document1 = Document(id="d1", text="", score=2)
    document2 = Document(id="d2", text="w1 w2 w3", score=1)

    inject_documents([document1, document2])

    axiom = axiom_fn()
    assert axiom.preference(query, document1, document2) == -axiom.preference(
        query, document2, document1
    )


def test_empty_document2(axiom_fn: Callable[[], Axiom]):
    query = Query("q1", "w1 w2 w3")
    document1 = Document("d1", "w1 w2 w3", 2)
    document2 = Document("d2", "", 1)

    inject_documents([document1, document2])

    axiom = axiom_fn()
    assert axiom.preference(query, document1, document2) == -axiom.preference(
        query, document2, document1
    )


def test_empty_documents(axiom_fn: Callable[[], Axiom]):
    query = Query("q1", "w1 w2 w3")
    document1 = Document("d1", "", 2)
    document2 = Document("d2", "", 1)

    inject_documents([document1, document2])

    axiom = axiom_fn()
    assert axiom.preference(query, document1, document2) == 0
    assert axiom.preference(query, document2, document1) == 0
