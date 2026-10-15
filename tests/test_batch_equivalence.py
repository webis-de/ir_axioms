from collections.abc import Callable, Iterator

from numpy import array, float_
from numpy.testing import assert_array_equal
from pytest import fixture, mark, param

from ir_axioms.axiom import (
    Axiom,
    AND,
    ANTI_REG,
    ASPECT_REG,
    DIV,
    LB1,
    LEN_AND,
    LEN_DIV,
    LEN_M_AND,
    LEN_M_TDC,
    LNC1,
    M_AND,
    M_TDC,
    ORACLE,
    ORIG,
    PROX1,
    PROX2,
    PROX3,
    PROX4,
    PROX5,
    REG,
    STMC1,
    STMC2,
    TF_LNC,
    TFC1,
    TFC3,
    aSLDoc,
)
from ir_axioms.dependency_injection import injector as _default_injector
from ir_axioms.model import Query, Document
from ir_axioms.tools import TermSimilarity
from ir_axioms.utils.injection import reset_binding_scopes
from tests.util import inject_documents


class _CharacterOverlapTermSimilarity(TermSimilarity):
    """Deterministic and fast term similarity for testing."""

    def similarity(self, term1: str, term2: str) -> float:
        characters1 = set(term1)
        characters2 = set(term2)
        return len(characters1 & characters2) / len(characters1 | characters2)


_QUERY = Query(
    id="q1",
    text="blue car fast",
)

_DOCUMENTS = [
    Document(
        id="d1",
        score=1.00,
        relevance=2,
        text="a blue car goes fast through the city. it is a nice car.",
    ),
    Document(
        id="d2",
        score=1.02,
        relevance=1,
        text="the car is blue and the car is fast",
    ),
    Document(
        id="d3",
        score=1.05,
        relevance=1,
        text="fast fast car without any color. really fast.",
    ),
    Document(
        id="d4",
        score=0.99,
        relevance=0,
        text="blue sky over the city with a car",
    ),
    Document(
        id="d5",
        score=2.00,
        relevance=0,
        text="nothing relevant in here at all",
    ),
    Document(
        id="d6",
        score=1.01,
        relevance=2,
        text="blue car fast blue car fast",
    ),
    Document(
        id="d7",
        score=1.03,
        relevance=1,
        text="a fast and blue but rather old car. old but fast.",
    ),
    Document(
        id="d8",
        score=1.00,
        relevance=0,
        text="cars are blue and quick, not fast",
    ),
    # Same query term frequencies but different lengths (for LNC1).
    Document(
        id="d9",
        score=1.00,
        relevance=1,
        text="blue car fast city",
    ),
    Document(
        id="d10",
        score=1.00,
        relevance=1,
        text="blue car fast city blue car fast city",
    ),
    # Same sum but different query term frequencies (for M_TDC).
    Document(
        id="d11",
        score=1.00,
        relevance=2,
        text="blue blue car road",
    ),
    Document(
        id="d12",
        score=1.00,
        relevance=0,
        text="blue car car road",
    ),
]


@fixture
def injected_tools() -> Iterator[None]:
    """
    Bind the test tools to the default injector (used by the axiom shorthands)
    and restore the previous bindings afterwards.
    """
    binder = _default_injector.binder
    previous_bindings = dict(binder._bindings)
    inject_documents(_DOCUMENTS)
    binder.bind(
        interface=TermSimilarity,
        to=_CharacterOverlapTermSimilarity(),
    )
    reset_binding_scopes()
    yield
    binder._bindings.clear()
    binder._bindings.update(previous_bindings)
    reset_binding_scopes()


@mark.parametrize(
    "axiom_shorthand",
    [
        param(ORIG, id="ORIG"),
        param(ORACLE, id="ORACLE"),
        param(TFC1, id="TFC1"),
        param(TFC3, id="TFC3"),
        param(M_TDC, id="M_TDC"),
        param(LEN_M_TDC, id="LEN_M_TDC"),
        param(LNC1, id="LNC1"),
        param(TF_LNC, id="TF_LNC"),
        param(LB1, id="LB1"),
        param(PROX1, id="PROX1"),
        param(PROX2, id="PROX2"),
        param(PROX3, id="PROX3"),
        param(PROX4, id="PROX4"),
        param(PROX5, id="PROX5"),
        param(STMC1, id="STMC1"),
        param(STMC2, id="STMC2"),
        param(REG, id="REG"),
        param(ANTI_REG, id="ANTI_REG"),
        param(ASPECT_REG, id="ASPECT_REG"),
        param(AND, id="AND"),
        param(LEN_AND, id="LEN_AND"),
        param(M_AND, id="M_AND"),
        param(LEN_M_AND, id="LEN_M_AND"),
        param(DIV, id="DIV"),
        param(LEN_DIV, id="LEN_DIV"),
        param(aSLDoc, id="aSLDoc"),
        # ArgUC, QTArg, and QTPArg are excluded, as they query the remote TARGER API.
    ],
)
def test_batched_preferences_equal_pairwise_preferences(
    injected_tools: None,
    axiom_shorthand: Callable[..., Axiom[Query, Document]],
) -> None:
    axiom = axiom_shorthand()

    query = _QUERY
    documents = _DOCUMENTS

    batched = axiom.preferences(query, documents)
    pairwise = array(
        [
            axiom.preference(query, document1, document2)
            for document1 in documents
            for document2 in documents
        ],
        dtype=float_,
    ).reshape((len(documents), len(documents)))

    assert_array_equal(batched, pairwise)
