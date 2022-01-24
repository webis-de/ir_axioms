from random import Random

from ir_axioms.axiom.base import Axiom
from ir_axioms.axiom.utils import strictly_less
from ir_axioms.model import Query, RankedDocument
from ir_axioms.model.context import RerankingContext


class NopAxiom(Axiom):
    name = "NopAxiom"

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        return 0


class OriginalAxiom(Axiom):
    name = "OriginalAxiom"

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        return strictly_less(document1.rank, document2.rank)


class RandomAxiom(Axiom):
    name = "RandomAxiom"
    _random: Random

    def __init__(self, random: Random = Random()):
        self._random = random

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        return self._random.randint(-1, 1)
