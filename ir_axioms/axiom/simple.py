from dataclasses import dataclass
from random import Random

from ir_axioms.axiom.base import Axiom
from ir_axioms.axiom.utils import strictly_less
from ir_axioms.model import Query, RankedDocument
from ir_axioms.model.context import RerankingContext


@dataclass(frozen=True)
class NopAxiom(Axiom):
    name = "nop"

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        return 0


@dataclass(frozen=True)
class OriginalAxiom(Axiom):
    name = "original"

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        return strictly_less(document1.rank, document2.rank)


@dataclass(frozen=True)
class RandomAxiom(Axiom):
    name = "random"

    random: Random = Random()

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        return self.random.randint(-1, 1)
