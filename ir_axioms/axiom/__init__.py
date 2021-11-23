from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Dict, Tuple

from ir_axioms.model import Query, RankedDocument
from ir_axioms.model.context import RerankingContext


class Axiom(ABC):

    @abstractmethod
    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        pass

    def weighted(self, weight: float) -> "Axiom":
        return WeightedAxiom(self, weight)

    def __mul__(self, weight: float):
        return self.weighted(weight)

    def aggregate(self, *others: "Axiom") -> "Axiom":
        return AggregatedAxiom([self, *others])

    def __add__(self, other: "Axiom"):
        return self.aggregate(other)

    def normalized(self) -> "Axiom":
        return NormalizedAxiom(self)

    def cached(self) -> "Axiom":
        return CachedAxiom(self)


@dataclass
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


@dataclass
class AggregatedAxiom(Axiom):
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


@dataclass
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


@dataclass
class CachedAxiom(Axiom):
    axiom: Axiom
    _cache: Dict[Tuple[int, str, str], float] = field(
        default_factory=lambda: {},
        init=False,
        repr=False
    )

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        if (query.id, document1.id, document2.id) in self._cache:
            return self._cache[query.id, document1.id, document2.id]
        elif (query.id, document2.id, document1.id) in self._cache:
            return -self._cache[query.id, document2.id, document1.id]
        else:
            preference = self.axiom.preference(
                context,
                query,
                document1,
                document2
            )
            self._cache[query.id, document1.id, document2.id] = preference
            return preference
