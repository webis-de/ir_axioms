from dataclasses import dataclass
from functools import reduce
from operator import mul
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
class ProductAxiom(Axiom):
    axioms: Iterable[Axiom]

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        return reduce(
            mul,
            (
                axiom.preference(context, query, document1, document2)
                for axiom in self.axioms
            ),
        )






@dataclass(frozen=True)
class InvertedAxiom(Axiom):
    axiom: Axiom

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        return 1 / self.axiom.preference(
            context,
            query,
            document1,
            document2
        )


@dataclass(frozen=True)
class AndAxiom(Axiom):
    axioms: Iterable[Axiom]

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        preferences = [
            axiom.preference(context, query, document1, document2)
            for axiom in self.axioms
        ]
        if all(preference > 0 for preference in preferences):
            return 1
        elif all(preference > 0 for preference in preferences):
            return -1
        else:
            return 0


@dataclass(frozen=True)
class MajorityVoteAxiom(Axiom):
    axioms: Iterable[Axiom]
    minimum_votes: float = 0.5
    """
    Minimum portion of votes in favor or against either document,
    to be considered a majority, 
    for example, 0.5 for absolute majority, 0.6 for qualified majority,
    or 0 for relative majority.
    """

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        preferences = [
            axiom.preference(context, query, document1, document2)
            for axiom in self.axioms
        ]
        count = len(preferences)
        positive_count = sum(1 for preference in preferences if preference > 0)
        negative_count = sum(1 for preference in preferences if preference < 0)
        positive_proportion = positive_count / count
        negative_proportion = negative_count / count
        if (
                positive_proportion > negative_proportion and
                positive_proportion >= self.minimum_votes
        ):
            return 1
        elif (
                negative_proportion > positive_proportion and
                negative_proportion >= self.minimum_votes
        ):
            return -1
        else:
            # Draw.
            return 0


@dataclass(frozen=True)
class NormalizedAxiom(Axiom):
    axiom: Axiom

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        preference = self.axiom.preference(
            context,
            query,
            document1,
            document2
        )
        if preference > 0:
            return 1
        elif preference < 0:
            return -1
        else:
            return 0
