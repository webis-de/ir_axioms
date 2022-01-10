from abc import ABC, abstractmethod
from collections import OrderedDict as ordereddict
from dataclasses import dataclass, field
from typing import Iterable, Tuple, OrderedDict

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
class _AxiomLRUCache:
    capacity: int = 4096

    # In the cache, the document pairs are stored
    # in ascending hash value order in order to benefit
    # from axioms' symmetric.
    _cache: OrderedDict[
        Tuple[RerankingContext, Query, RankedDocument, RankedDocument],
        float
    ] = field(
        default_factory=lambda: ordereddict(),
        init=False,
        repr=False
    )

    @staticmethod
    def _key(
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> Tuple[RerankingContext, Query, RankedDocument, RankedDocument]:
        if hash(document1) <= hash(document2):
            return context, query, document1, document2
        else:
            return context, query, document2, document1

    @staticmethod
    def _sign(
            document1: RankedDocument,
            document2: RankedDocument
    ) -> int:
        if hash(document1) <= hash(document2):
            return 1
        else:
            return -1

    def __contains__(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        key = self._key(context, query, document1, document2)
        return key in self._cache

    def __getitem__(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        key = self._key(context, query, document1, document2)
        sign = self._sign(document1, document2)

        value = self._cache[key] * sign
        self._cache.move_to_end(key)
        return value

    def __setitem__(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument,
            preference: float
    ):
        key = self._key(context, query, document1, document2)
        sign = self._sign(document1, document2)

        value = preference * sign
        self._cache[key] = value
        self._cache.move_to_end(key)

        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)


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
