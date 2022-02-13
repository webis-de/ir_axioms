from dataclasses import dataclass

from diskcache import Cache

from ir_axioms.axiom.base import Axiom
from ir_axioms.model import RankedDocument, Query, IndexContext


@dataclass(frozen=True)
class CachedAxiom(Axiom):
    axiom: Axiom
    disk: bool = False

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
        cache: Cache = context.cache

        if cache is None:
            return self.axiom.preference(context, query, document1, document2)

        key = self._key(context, query, document1, document2)

        if key in cache:
            # Cache hit.
            return cache[key]

        symmetric_key = self._key(context, query, document2, document1)
        if symmetric_key in cache:
            # Cache hit for symmetric key.
            preference = -cache[symmetric_key]
            cache[key] = preference
            return preference

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
