from dataclasses import dataclass
from math import isclose

from axioms.axiom.base import Axiom
from axioms.axiom.utils import approximately_equal
from axioms.model import Query, RankedDocument, IndexContext


@dataclass(frozen=True, kw_only=True)
class LB1(Axiom):
    name = "LB1"

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        if not approximately_equal(document1.score, document2.score):
            return 0

        # TODO: Do we really want to use the term set here, not the list?
        for term in context.term_set(query):
            tf1 = context.term_frequency(document1, term)
            tf2 = context.term_frequency(document2, term)
            if isclose(tf1, 0) and tf2 > 0:
                return -1
            if isclose(tf2, 0) and tf1 > 0:
                return 1
        return 0
