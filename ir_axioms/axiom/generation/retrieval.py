from dataclasses import dataclass
from typing import Final, Sequence

from injector import inject

from ir_axioms.axiom.base import Axiom
from ir_axioms.axiom.retrieval import (
    ArgumentativeUnitsCountAxiom,
    QueryTermOccurrenceInArgumentativeUnitsAxiom,
    QueryTermPositionInArgumentativeUnitsAxiom,
    AverageSentenceLengthAxiom,
    Lnc1Axiom,
    TfLncAxiom,
    Prox1Axiom,
    Prox2Axiom,
    Prox3Axiom,
    Prox4Axiom,
    Prox5Axiom,
    RegAxiom,
    AntiRegAxiom,
    AndAxiom,
    LenAndAxiom,
    ModifiedAndAxiom,
    DivAxiom,
    LenDivAxiom,
    Tfc1Axiom,
    Stmc1Axiom,
    Stmc2Axiom,
)
from ir_axioms.model import (
    GenerationInput,
    GenerationOutput,
    Preference,
    PreferenceMatrix,
    Query,
    Document,
    TextDocument,
)
from ir_axioms.utils.lazy import lazy_inject


class _RetrievalAxiomWrapper(Axiom[GenerationInput, GenerationOutput]):
    axiom: Axiom[Query, Document]

    def preference(
        self,
        input: GenerationInput,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        return self.axiom.preference(
            input=Query(
                id=input.id if input.id is not None else "",
                text=input.text if input.text is not None else "",
            ),
            output1=TextDocument(
                id=output1.id if output1.id is not None else "",
                text=output1.text if output1.text is not None else "",
            ),
            output2=TextDocument(
                id=output2.id if output2.id is not None else "",
                text=output2.text if output2.text is not None else "",
            ),
        )

    def preferences(
        self,
        input: GenerationInput,
        outputs: Sequence[GenerationOutput],
    ) -> PreferenceMatrix:
        retrieval_input = Query(
            id=input.id if input.id is not None else "",
            text=input.text if input.text is not None else "",
        )
        retrieval_outputs = [
            TextDocument(
                id=output.id if output.id is not None else "",
                text=output.text if output.text is not None else "",
            )
            for output in outputs
        ]
        return self.axiom.preferences(retrieval_input, retrieval_outputs)


@inject
@dataclass(frozen=True, kw_only=True)
class GenerativeArgumentativeUnitsCountAxiom(_RetrievalAxiomWrapper):
    axiom: ArgumentativeUnitsCountAxiom  # pyright: ignore[reportIncompatibleVariableOverride]


GEN_ArgUC: Final = lazy_inject(GenerativeArgumentativeUnitsCountAxiom)


@inject
@dataclass(frozen=True, kw_only=True)
class GenerativeQueryTermOccurrenceInArgumentativeUnitsAxiom(_RetrievalAxiomWrapper):
    axiom: QueryTermOccurrenceInArgumentativeUnitsAxiom  # pyright: ignore[reportIncompatibleVariableOverride]


GEN_QTArg: Final = lazy_inject(GenerativeQueryTermOccurrenceInArgumentativeUnitsAxiom)


@inject
@dataclass(frozen=True, kw_only=True)
class GenerativeQueryTermPositionInArgumentativeUnitsAxiom(_RetrievalAxiomWrapper):
    axiom: QueryTermPositionInArgumentativeUnitsAxiom  # pyright: ignore[reportIncompatibleVariableOverride]


GEN_QTPArg: Final = lazy_inject(GenerativeQueryTermPositionInArgumentativeUnitsAxiom)


@inject
@dataclass(frozen=True, kw_only=True)
class GenerativeAverageSentenceLengthAxiom(_RetrievalAxiomWrapper):
    axiom: AverageSentenceLengthAxiom  # pyright: ignore[reportIncompatibleVariableOverride]


GEN_aSL: Final = lazy_inject(GenerativeAverageSentenceLengthAxiom)


@inject
@dataclass(frozen=True, kw_only=True)
class GenerativeLnc1Axiom(_RetrievalAxiomWrapper):
    axiom: Lnc1Axiom  # pyright: ignore[reportIncompatibleVariableOverride]


GEN_LNC1: Final = lazy_inject(GenerativeLnc1Axiom)


@inject
@dataclass(frozen=True, kw_only=True)
class GenerativeTfLncAxiom(_RetrievalAxiomWrapper):
    axiom: TfLncAxiom  # pyright: ignore[reportIncompatibleVariableOverride]


GEN_TF_LNC: Final = lazy_inject(GenerativeTfLncAxiom)


@inject
@dataclass(frozen=True, kw_only=True)
class GenerativeProx1Axiom(_RetrievalAxiomWrapper):
    axiom: Prox1Axiom  # pyright: ignore[reportIncompatibleVariableOverride]


GEN_PROX1: Final = lazy_inject(GenerativeProx1Axiom)


@inject
@dataclass(frozen=True, kw_only=True)
class GenerativeProx2Axiom(_RetrievalAxiomWrapper):
    axiom: Prox2Axiom  # pyright: ignore[reportIncompatibleVariableOverride]


GEN_PROX2: Final = lazy_inject(GenerativeProx2Axiom)


@inject
@dataclass(frozen=True, kw_only=True)
class GenerativeProx3Axiom(_RetrievalAxiomWrapper):
    axiom: Prox3Axiom  # pyright: ignore[reportIncompatibleVariableOverride]


GEN_PROX3: Final = lazy_inject(GenerativeProx3Axiom)


@inject
@dataclass(frozen=True, kw_only=True)
class GenerativeProx4Axiom(_RetrievalAxiomWrapper):
    axiom: Prox4Axiom  # pyright: ignore[reportIncompatibleVariableOverride]


GEN_PROX4: Final = lazy_inject(GenerativeProx4Axiom)


@inject
@dataclass(frozen=True, kw_only=True)
class GenerativeProx5Axiom(_RetrievalAxiomWrapper):
    axiom: Prox5Axiom  # pyright: ignore[reportIncompatibleVariableOverride]


GEN_PROX5: Final = lazy_inject(GenerativeProx5Axiom)


@inject
@dataclass(frozen=True, kw_only=True)
class GenerativeRegAxiom(_RetrievalAxiomWrapper):
    axiom: RegAxiom  # pyright: ignore[reportIncompatibleVariableOverride]


GEN_REG: Final = lazy_inject(GenerativeRegAxiom)


@inject
@dataclass(frozen=True, kw_only=True)
class GenerativeAntiRegAxiom(_RetrievalAxiomWrapper):
    axiom: AntiRegAxiom  # pyright: ignore[reportIncompatibleVariableOverride]


GEN_ANTI_REG: Final = lazy_inject(GenerativeAntiRegAxiom)


@inject
@dataclass(frozen=True, kw_only=True)
class GenerativeAndAxiom(_RetrievalAxiomWrapper):
    axiom: AndAxiom  # pyright: ignore[reportIncompatibleVariableOverride]


GEN_AND: Final = lazy_inject(GenerativeAndAxiom)


@inject
@dataclass(frozen=True, kw_only=True)
class GenerativeLenAndAxiom(_RetrievalAxiomWrapper):
    axiom: LenAndAxiom  # pyright: ignore[reportIncompatibleVariableOverride]


GEN_LEN_AND: Final = lazy_inject(GenerativeLenAndAxiom)


@inject
@dataclass(frozen=True, kw_only=True)
class GenerativeModifiedAndAxiom(_RetrievalAxiomWrapper):
    axiom: ModifiedAndAxiom  # pyright: ignore[reportIncompatibleVariableOverride]


GEN_M_AND: Final = lazy_inject(GenerativeModifiedAndAxiom)


@inject
@dataclass(frozen=True, kw_only=True)
class GenerativeDivAxiom(_RetrievalAxiomWrapper):
    axiom: DivAxiom  # pyright: ignore[reportIncompatibleVariableOverride]


GEN_DIV: Final = lazy_inject(GenerativeDivAxiom)


@inject
@dataclass(frozen=True, kw_only=True)
class GenerativeLenDivAxiom(_RetrievalAxiomWrapper):
    axiom: LenDivAxiom  # pyright: ignore[reportIncompatibleVariableOverride]


GEN_LEN_DIV: Final = lazy_inject(GenerativeLenDivAxiom)


@inject
@dataclass(frozen=True, kw_only=True)
class GenerativeTfc1Axiom(_RetrievalAxiomWrapper):
    axiom: Tfc1Axiom  # pyright: ignore[reportIncompatibleVariableOverride]


GEN_TFC1: Final = lazy_inject(GenerativeTfc1Axiom)


@inject
@dataclass(frozen=True, kw_only=True)
class GenerativeStmc1Axiom(_RetrievalAxiomWrapper):
    axiom: Stmc1Axiom  # pyright: ignore[reportIncompatibleVariableOverride]


GEN_STMC1: Final = lazy_inject(GenerativeStmc1Axiom)


@inject
@dataclass(frozen=True, kw_only=True)
class GenerativeStmc2Axiom(_RetrievalAxiomWrapper):
    axiom: Stmc2Axiom  # pyright: ignore[reportIncompatibleVariableOverride]


GEN_STMC2: Final = lazy_inject(GenerativeStmc2Axiom)
