# Re-export from sub-modules.

from axioms.axiom.generation.clarity import (  # noqa: F401
    GrammarErrorsClarityAxiom,
    CLAR1,
    GrammarErrorTypesClarityAxiom,
    CLAR2,
    GrammarErrorProportionClarityAxiom,
    CLAR3,
    SentenceRedundancyClarityAxiom,
    CLAR4,
    MisspellingsClarityAxiom,
    CLAR5,
    WordCommonnessClarityAxiom,
    CLAR6,
    NormalizedWordCommonnessClarityAxiom,
    CLAR7,
)

# from axioms.axiom.generation.coherence import (  # noqa: F401
# )

from axioms.axiom.generation.consistency import (  # noqa: F401
    AspectOverlapConsistenyAxiom,
    CONS1,
    PenalizedAspectOverlapConsistencyAxiom,
    CONS2,
    AspectSimilarityConsistencyAxiom,
    CONS3,
    SentenceContradictionConsistencyAxiom,
    CONS4,
)

# from axioms.axiom.generation.correctness import (  # noqa: F401
# )

from axioms.axiom.generation.coverage import (  # noqa: F401
    AspectOverlapCoverageAxiom,
    COV1,
    PenalizedAspectOverlapCoverageAxiom,
    COV2,
    AspectSimilarityCoverageAxiom,
    COV3,
    AspectCountCoverageAxiom,
    COV4,
    AspectRedundancyCoverageAxiom,
    COV5,
)

from axioms.axiom.generation.retrieval import (  # noqa: F401
    GenerativeArgumentativeUnitsCountAxiom,
    GEN_ArgUC,
    GenerativeQueryTermOccurrenceInArgumentativeUnitsAxiom,
    GEN_QTArg,
    GenerativeQueryTermPositionInArgumentativeUnitsAxiom,
    GEN_QTPArg,
    GenerativeAverageSentenceLengthAxiom,
    GEN_aSL,
    GenerativeLnc1Axiom,
    GEN_LNC1,
    GenerativeTfLncAxiom,
    GEN_TF_LNC,
    GenerativeProx1Axiom,
    GEN_PROX1,
    GenerativeProx2Axiom,
    GEN_PROX2,
    GenerativeProx3Axiom,
    GEN_PROX3,
    GenerativeProx4Axiom,
    GEN_PROX4,
    GenerativeProx5Axiom,
    GEN_PROX5,
    GenerativeRegAxiom,
    GEN_REG,
    GenerativeAntiRegAxiom,
    GEN_ANTI_REG,
    GenerativeAndAxiom,
    GEN_AND,
    GenerativeLenAndAxiom,
    GEN_LEN_AND,
    GenerativeModifiedAndAxiom,
    GEN_M_AND,
    GenerativeModifiedTdcAxiom,
    GEN_M_TDC,
    GenerativeLenModifiedTdcAxiom,
    GEN_LEN_M_TDC,
    GenerativeDivAxiom,
    GEN_DIV,
    GenerativeLenDivAxiom,
    GEN_LEN_DIV,
    GenerativeTfc1Axiom,
    GEN_TFC1,
    GenerativeStmc1Axiom,
    GEN_STMC1,
    GenerativeStmc2Axiom,
    GEN_STMC2,
)


# TODO: generated text length axiom (for normalization), in which utility dimension does it fit best?


from axioms.axiom.generation.crowd import (  # noqa: F401
    TrecRagCrowdAxiom,
)

from axioms.axiom.generation.trec import (  # noqa: F401
    TrecRagNuggetAxiom,
)
