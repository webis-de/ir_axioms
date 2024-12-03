from dataclasses import dataclass
from typing import Final

from axioms.axiom.base import Axiom
from axioms.axiom.utils import approximately_equal, strictly_less, strictly_greater
from axioms.model import Query, Document, IndexContext
from axioms.model.retrieval import get_index_context


@dataclass(frozen=True, kw_only=True)
class Lnc1Axiom(Axiom):
    context: IndexContext

    def preference(self, query: Query, document1: Document, document2: Document):
        if not all(
            approximately_equal(
                self.context.term_frequency(document1, term),
                self.context.term_frequency(document2, term),
            )
            for term in self.context.term_set(query)
        ):
            return 0

        # Prefer the shorter document.
        return strictly_less(
            len(self.context.terms(document1)),
            len(self.context.terms(document2)),
        )


LNC1: Final = Lnc1Axiom(
    context=get_index_context(),
)


@dataclass(frozen=True, kw_only=True)
class TfLncAxiom(Axiom):
    context: IndexContext

    def preference(self, query: Query, document1: Document, document2: Document):
        sum_document1 = 0
        sum_document2 = 0

        for query_term in self.context.term_set(query):
            tf_d1 = self.context.term_frequency(document1, query_term)
            tf_d2 = self.context.term_frequency(document2, query_term)

            len_d1 = len(
                [term for term in self.context.terms(document1) if term != query_term]
            )
            len_d2 = len(
                [term for term in self.context.terms(document2) if term != query_term]
            )

            if len_d1 == len_d2:
                if tf_d1 > tf_d2:
                    sum_document1 += 1
                elif tf_d2 > tf_d1:
                    sum_document2 += 1

        return strictly_greater(sum_document1, sum_document2)


TF_LNC: Final = TfLncAxiom(
    context=get_index_context(),
)
