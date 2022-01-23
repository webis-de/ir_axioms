from abc import ABC, abstractmethod
from dataclasses import dataclass
from inspect import isabstract
from typing import final, List

from ir_axioms import registry
from ir_axioms.model import Query, RankedDocument
from ir_axioms.model.context import RerankingContext


class Axiom(ABC):
    name: str = None

    def __init_subclass__(cls, **kwargs):
        if not isabstract(cls) and cls.name is not None:
            registry[cls.name] = cls

    @abstractmethod
    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        pass

    @final
    def weighted(self, weight: float) -> "Axiom":
        from ir_axioms.axiom.arithmetic import WeightedAxiom
        return WeightedAxiom(self, weight)

    @final
    def __mul__(self, weight: float):
        return self.weighted(weight)

    @final
    def aggregate(self, *others: "Axiom") -> "Axiom":
        from ir_axioms.axiom.arithmetic import SumAxiom
        return SumAxiom([self, *others])

    @final
    def __add__(self, other: "Axiom"):
        return self.aggregate(other)

    @final
    def normalized(self) -> "Axiom":
        return NormalizedAxiom(self)

    @final
    def cached(self, capacity: int = 4096) -> "Axiom":
        from ir_axioms.axiom.cache import CachedAxiom
        return CachedAxiom(self, capacity)

    @final
    def rerank(
            self,
            context: RerankingContext,
            query: Query,
            ranking: List[RankedDocument],
    ) -> List[RankedDocument]:
        from ir_axioms.axiom.utils_actions import _kwiksort, _reset_score

        ranking = _kwiksort(self, query, context, ranking)
        ranking = _reset_score(ranking)
        return ranking

    @final
    def preferences(
            self,
            context: RerankingContext,
            query: Query,
            ranking: List[RankedDocument],
    ) -> List[List[float]]:
        return [
            [
                self.preference(context, query, document1, document2)
                for document2 in ranking
            ]
            for document1 in ranking
        ]

    @final
    def is_permutated(
            self,
            context: RerankingContext,
            query: Query,
            document_1: RankedDocument,
            document_2: RankedDocument
    ):
        if document_1 is document_2:
            return False
        preference = self.preference(context, query, document_1, document_2)
        if preference == 0 and document_1.rank == document_2.rank:
            return False
        elif preference > 0 and document_1.rank < document_2.rank:
            return False
        elif preference < 0 and document_1.rank > document_2.rank:
            return False
        else:
            return True

    @final
    def permutations(
            self,
            context: RerankingContext,
            query: Query,
            ranking: List[RankedDocument],
    ) -> List[List[bool]]:
        return [
            [
                self.is_permutated(context, query, document1, document2)
                if index1 != index2 else False
                for index2, document2 in enumerate(ranking)
            ]
            for index1, document1 in enumerate(ranking)
        ]

    @final
    def permutation_count(
            self,
            context: RerankingContext,
            query: Query,
            ranking: List[RankedDocument],
    ) -> List[int]:
        return [
            sum(1 for is_pair_permutated in pairs if is_pair_permutated)
            for pairs in self.permutations(context, query, ranking)
        ]

    @final
    def permutation_frequency(
            self,
            context: RerankingContext,
            query: Query,
            ranking: List[RankedDocument],
    ) -> List[float]:
        return [
            (count - 1) / len(ranking) if len(ranking) > 0 else 0
            for count in self.permutation_count(context, query, ranking)
        ]


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
