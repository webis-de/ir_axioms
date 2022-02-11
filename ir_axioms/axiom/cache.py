from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from diskcache import Cache

from ir_axioms.axiom.base import Axiom
from ir_axioms.model import RankedDocument, Query, IndexContext


@lru_cache(None)
def _cache(context: IndexContext) -> Optional[Cache]:
    if context.cache_dir is None:
        return None
    return Cache(str(context.cache_dir.absolute()))


@dataclass(frozen=True)
class CachedAxiom(Axiom):
    axiom: Axiom

    def _key(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> str:
        return (
            f"{self.axiom!r},{context!r},"
            f"{query.title},{document1.id},{document2.id}"
        )

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        cache = _cache(context)

        if cache is None:
            return self.axiom.preference(context, query, document1, document2)

        key = self._key(context, query, document1, document2)
        preference: float

        if key in cache:
            # Cache hit.
            preference = cache[key]
        else:
            symmetric_key = self._key(context, query, document2, document1)
            if symmetric_key in cache:
                # Cache hit for symmetric key.
                inverse_preference = cache[symmetric_key]
                preference = -inverse_preference
            else:
                # Cache miss.
                preference = self.axiom.preference(
                    context,
                    query,
                    document1,
                    document2
                )
            cache[key] = preference

        return preference

    def cached(self) -> Axiom:
        return self
