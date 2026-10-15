from collections.abc import Callable, Iterator

from numpy import array, float_
from numpy.testing import assert_array_equal
from pytest import fixture, mark, param

from ir_axioms.axiom import (
    Axiom,
    CLAR2,
    COH1,
    COH2,
    CONS1,
    CONS2,
    CONS3,
    CORR1,
    COV1,
    COV2,
    COV3,
    GEN_AND,
    GEN_ANTI_REG,
    GEN_DIV,
    GEN_LEN_AND,
    GEN_LEN_DIV,
    GEN_LNC1,
    GEN_M_AND,
    GEN_PROX1,
    GEN_PROX2,
    GEN_PROX3,
    GEN_PROX4,
    GEN_PROX5,
    GEN_REG,
    GEN_STMC1,
    GEN_STMC2,
    GEN_TF_LNC,
    GEN_TFC1,
    GEN_aSL,
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
from ir_axioms.model import Query, Document, GenerationRequest, GenerationResponse
from ir_axioms.tools import SentenceSimilarity, TermSimilarity
from ir_axioms.tools.similarity import AverageTermSimilaritySentenceSimilarity
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


_REQUEST = GenerationRequest(
    id="r1",
    text="blue car fast",
    context=[
        "The Tesla Model S is a fast electric car.",
        "Porsche built a blue car with a top speed of 300 km/h in 2019.",
    ],
)

_RESPONSES = [
    GenerationResponse(
        id="g1",
        text="The Tesla Model S is a fast car [1]. It is often blue [2].",
    ),
    GenerationResponse(
        id="g2",
        text="A blue car is fast. The Porsche reaches 300 km/h [1].",
    ),
    GenerationResponse(
        id="g3",
        text=(
            "Porsche built a blue car in 2019. "
            "Porsche built a red car in 2021. "
            "Both cars are fast."
        ),
    ),
    GenerationResponse(
        id="g4",
        text="Fast cars are fast. Blue cars are blue. Cars are cars.",
    ),
    GenerationResponse(
        id="g5",
        text="I do not know which car is fast.",
    ),
    GenerationResponse(
        id="g6",
        text=(
            "The Tesla Model S, an electric car produced in California since 2012, "
            "is notably fast, while many blue cars, including those built by "
            "Porsche in Germany, reach a top speed of 300 km/h [3]."
        ),
    ),
    GenerationResponse(
        id="g7",
        text="Blue car fast. Blue car fast blue car fast.",
    ),
    GenerationResponse(
        id="g8",
        text="See https://example.com for fast blue cars. The Model S is fast [2].",
    ),
    # Contradictory mentions of the same entity (for CONS3).
    GenerationResponse(
        id="g9",
        text="Tesla sells blue cars in Germany. Tesla does not sell cars in Germany.",
    ),
    # Same query term frequencies but different lengths (for GEN_LNC1).
    GenerationResponse(
        id="g10",
        text="blue car fast city",
    ),
    GenerationResponse(
        id="g11",
        text="blue car fast city blue car fast city",
    ),
    # Same length but different average sentence lengths (for GEN_aSL).
    GenerationResponse(
        id="g12",
        text="The blue car is fast and the red car is slow but both are old.",
    ),
    GenerationResponse(
        id="g13",
        text="The blue car. Is fast. And the red car. Is slow. But both. Are old.",
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
    # Avoid downloading sentence embedding models.
    binder.bind(
        interface=SentenceSimilarity,
        to=AverageTermSimilaritySentenceSimilarity,
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


@mark.parametrize(
    "axiom_shorthand",
    [
        param(CLAR2, id="CLAR2"),
        param(COH1, id="COH1"),
        param(COH2, id="COH2"),
        param(CONS1, id="CONS1"),
        param(CONS2, id="CONS2"),
        param(CONS3, id="CONS3"),
        param(CORR1, id="CORR1"),
        param(COV1, id="COV1"),
        param(COV2, id="COV2"),
        param(COV3, id="COV3"),
        param(GEN_aSL, id="GEN_aSL"),
        param(GEN_LNC1, id="GEN_LNC1"),
        param(GEN_TF_LNC, id="GEN_TF_LNC"),
        param(GEN_PROX1, id="GEN_PROX1"),
        param(GEN_PROX2, id="GEN_PROX2"),
        param(GEN_PROX3, id="GEN_PROX3"),
        param(GEN_PROX4, id="GEN_PROX4"),
        param(GEN_PROX5, id="GEN_PROX5"),
        param(GEN_REG, id="GEN_REG"),
        param(GEN_ANTI_REG, id="GEN_ANTI_REG"),
        param(GEN_AND, id="GEN_AND"),
        param(GEN_LEN_AND, id="GEN_LEN_AND"),
        param(GEN_M_AND, id="GEN_M_AND"),
        param(GEN_DIV, id="GEN_DIV"),
        param(GEN_LEN_DIV, id="GEN_LEN_DIV"),
        param(GEN_TFC1, id="GEN_TFC1"),
        param(GEN_STMC1, id="GEN_STMC1"),
        param(GEN_STMC2, id="GEN_STMC2"),
        # CLAR1 is excluded, as it downloads and runs a LanguageTool server.
        # GEN_ArgUC, GEN_QTArg, and GEN_QTPArg are excluded,
        # as they query the remote TARGER API.
    ],
)
def test_generation_batched_preferences_equal_pairwise_preferences(
    injected_tools: None,
    axiom_shorthand: Callable[..., Axiom[GenerationRequest, GenerationResponse]],
) -> None:
    axiom = axiom_shorthand()

    request = _REQUEST
    responses = _RESPONSES

    batched = axiom.preferences(request, responses)
    pairwise = array(
        [
            axiom.preference(request, response1, response2)
            for response1 in responses
            for response2 in responses
        ],
        dtype=float_,
    ).reshape((len(responses), len(responses)))

    assert_array_equal(batched, pairwise)
