from dataclasses import dataclass
from typing import Iterable

from ir_axioms.axiom.base import Axiom
from ir_axioms.model import Query, RankedDocument
from ir_axioms.model.context import RerankingContext


@dataclass(frozen=True)
class UniformAxiom(Axiom):
    preference: float

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        return self.preference


@dataclass(frozen=True)
class SumAxiom(Axiom):
    axioms: Iterable[Axiom]

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        return sum(
            axiom.preference(context, query, document1, document2)
            for axiom in self.axioms
        )


@dataclass(frozen=True)
class WeightedAxiom(Axiom):
    axiom: Axiom
    weight: float

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        return self.weight * self.axiom.preference(
            context,
            query,
            document1,
            document2
        )
