from dataclasses import dataclass
from functools import cached_property
from random import Random
from typing import Any, Optional

from ir_axioms.axiom.base import Axiom
from ir_axioms.axiom.utils import strictly_less
from ir_axioms.model import Query, RankedDocument, IndexContext


@dataclass(frozen=True)
class NopAxiom(Axiom):
    name = "NOP"

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        return 0


@dataclass(frozen=True)
class OriginalAxiom(Axiom):
    name = "ORIG"

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        return strictly_less(document1.rank, document2.rank)


@dataclass(frozen=True)
class RandomAxiom(Axiom):
    name = "RANDOM"

    seed: Optional[Any] = None

    @cached_property
    def _random(self) -> Random:
        return Random(self.seed)

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        return self._random.randint(-1, 1)


# Aliases for shorter names:
NOP = NopAxiom
ORIG = OriginalAxiom
RANDOM = RandomAxiom
