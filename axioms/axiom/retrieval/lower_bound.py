from dataclasses import dataclass
from math import isclose
from typing import Final

from axioms.axiom.base import Axiom
from axioms.axiom.utils import approximately_equal
from axioms.model import Query, RankedDocument, IndexContext
from axioms.model.retrieval import get_index_context


@dataclass(frozen=True, kw_only=True)
class Lb1Axiom(Axiom):
    context: IndexContext

    def preference(
            self,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        if not approximately_equal(document1.score, document2.score):
            return 0

        # TODO: Do we really want to use the term set here, not the list?
        for term in self.context.term_set(query):
            tf1 = self.context.term_frequency(document1, term)
            tf2 = self.context.term_frequency(document2, term)
            if isclose(tf1, 0) and tf2 > 0:
                return -1
            if isclose(tf2, 0) and tf1 > 0:
                return 1
        return 0

LB1: Final = Lb1Axiom(
    context=get_index_context(),
)
