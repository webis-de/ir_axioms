from collections import OrderedDict as OrderedDictImpl
from dataclasses import field, dataclass
from typing import OrderedDict, Tuple

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
