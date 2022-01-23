from collections import OrderedDict as OrderedDictImpl
from dataclasses import field, dataclass
from typing import OrderedDict, Tuple

from ir_axioms.axiom.base import Axiom
from ir_axioms.model import RankedDocument, Query
from ir_axioms.model.context import RerankingContext


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
        default_factory=lambda: OrderedDictImpl(),
        init=False,
        repr=False
    )

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
            key: Tuple[
                RerankingContext,
                Query,
                RankedDocument,
                RankedDocument
            ],
    ):
        return key in self._cache

    def __getitem__(
            self,
            key: Tuple[
                RerankingContext,
                Query,
                RankedDocument,
                RankedDocument
            ],
    ):
        (_, _, document1, document2) = key
        sign = self._sign(document1, document2)

        value = self._cache[key] * sign
        self._cache.move_to_end(key)
        return value

    def __setitem__(
            self,
            key: Tuple[
                RerankingContext,
                Query,
                RankedDocument,
                RankedDocument
            ],
            preference: float,
    ):
        (_, _, document1, document2) = key
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
            return self._cache[(context, query, document1, document2)]
        else:
            preference = self.axiom.preference(
                context,
                query,
                document1,
                document2
            )
            self._cache[(context, query, document1, document2)] = preference
            return preference

    def cached(self, capacity: int = 4096) -> Axiom:
        if self.capacity >= capacity:
            # This axiom is already cached with a larger.
            return self
        else:
            # Cache wrapped axiom with higher capacity.
            return CachedAxiom(self.axiom, capacity)
