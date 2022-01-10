from abc import ABC, abstractmethod
from dataclasses import dataclass
from inspect import isabstract
from typing import Iterable

from ir_axioms.axiom.cache import _AxiomLRUCache
from ir_axioms.model import Query, RankedDocument
from ir_axioms.model.context import RerankingContext
from ir_axioms import registry


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

    def cached(self, capacity: int = 4096) -> "Axiom":
        return CachedAxiom(self, capacity)


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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class CachedAxiom(Axiom):
    axiom: Axiom
    capacity: int = 4096

    _cache = _AxiomLRUCache(capacity)

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        if (context, query, document1, document2) in self._cache:
            return self._cache[context, query, document1, document2]
        else:
            preference = self.axiom.preference(
                context,
                query,
                document1,
                document2
            )
            self._cache[context, query, document1, document2] = preference
            return preference
