# Re-export from sub-modules.

from ir_axioms.axiom.retrieval.argumentative import (
    ArgumentativeUnitsCountAxiom,
    ArgUC,
    QueryTermOccurrenceInArgumentativeUnitsAxiom,
    QTArg,
    QueryTermPositionInArgumentativeUnitsAxiom,
    QTPArg,
    AverageSentenceLengthAxiom,
    aSLDoc,
    aSL,
)

from ir_axioms.axiom.retrieval.length_norm import (
    Lnc1Axiom,
    LNC1,
    TfLncAxiom,
    TF_LNC,
)

from ir_axioms.axiom.retrieval.lower_bound import (
    Lb1Axiom,
    LB1,
)

from ir_axioms.axiom.retrieval.proximity import (
    Prox1Axiom,
    PROX1,
    Prox2Axiom,
    PROX2,
    Prox3Axiom,
    PROX3,
    Prox4Axiom,
    PROX4,
    Prox5Axiom,
    PROX5,
)

from ir_axioms.axiom.retrieval.query_aspects import (
    RegAxiom,
    REG,
    AntiRegAxiom,
    ANTI_REG,
    AspectRegAxiom,
    ASPECT_REG,
    AndAxiom,
    AND,
    LenAndAxiom,
    LEN_AND,
    ModifiedAndAxiom,
    M_AND,
    LenModifiedAndAxiom,
    LEN_M_AND,
    DivAxiom,
    DIV,
    LenDivAxiom,
    LEN_DIV,
)

from ir_axioms.axiom.retrieval.simple import (
    OriginalAxiom,
    ORIG,
    OracleAxiom,
    ORACLE,
)

from ir_axioms.axiom.retrieval.term_frequency import (
    Tfc1Axiom,
    TFC1,
    Tfc3Axiom,
    TFC3,
    ModifiedTdcAxiom,
    M_TDC,
    LenModifiedTdcAxiom,
    LEN_M_TDC,
)

from ir_axioms.axiom.retrieval.term_similarity import (
    Stmc1Axiom,
    STMC1,
    Stmc2Axiom,
    STMC2,
)

from ir_axioms.axiom.retrieval.trec import (
    TrecOracleAxiom,
)

__all__ = [
    "ArgumentativeUnitsCountAxiom",
    "ArgUC",
    "QueryTermOccurrenceInArgumentativeUnitsAxiom",
    "QTArg",
    "QueryTermPositionInArgumentativeUnitsAxiom",
    "QTPArg",
    "AverageSentenceLengthAxiom",
    "aSLDoc",
    "aSL",
    "Lnc1Axiom",
    "LNC1",
    "TfLncAxiom",
    "TF_LNC",
    "Lb1Axiom",
    "LB1",
    "Prox1Axiom",
    "PROX1",
    "Prox2Axiom",
    "PROX2",
    "Prox3Axiom",
    "PROX3",
    "Prox4Axiom",
    "PROX4",
    "Prox5Axiom",
    "PROX5",
    "RegAxiom",
    "REG",
    "AntiRegAxiom",
    "ANTI_REG",
    "AspectRegAxiom",
    "ASPECT_REG",
    "AndAxiom",
    "AND",
    "LenAndAxiom",
    "LEN_AND",
    "ModifiedAndAxiom",
    "M_AND",
    "LenModifiedAndAxiom",
    "LEN_M_AND",
    "DivAxiom",
    "DIV",
    "LenDivAxiom",
    "LEN_DIV",
    "OriginalAxiom",
    "ORIG",
    "OracleAxiom",
    "ORACLE",
    "Tfc1Axiom",
    "TFC1",
    "Tfc3Axiom",
    "TFC3",
    "ModifiedTdcAxiom",
    "M_TDC",
    "LenModifiedTdcAxiom",
    "LEN_M_TDC",
    "Stmc1Axiom",
    "STMC1",
    "Stmc2Axiom",
    "STMC2",
    "TrecOracleAxiom",
]