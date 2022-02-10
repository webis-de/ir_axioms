from dataclasses import dataclass

from ir_axioms.axiom.base import Axiom
from ir_axioms.axiom.utils import (
    approximately_equal, strictly_less, strictly_greater
)
from ir_axioms.model import Query, RankedDocument, IndexContext


@dataclass(frozen=True)
class LNC1(Axiom):
    name = "LNC1"

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        if not all(
                approximately_equal(
                    context.term_frequency(document1, term),
                    context.term_frequency(document2, term)
                )
                for term in context.term_set(query)
        ):
            return 0

        # Prefer the shorter document.
        return strictly_less(
            len(context.terms(document1)),
            len(context.terms(document2)),
        )


@dataclass(frozen=True)
class TF_LNC(Axiom):
    name = "TF-LNC"

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        sum_document1 = 0
        sum_document2 = 0

        for query_term in context.term_set(query):
            tf_d1 = context.term_frequency(document1, query_term)
            tf_d2 = context.term_frequency(document2, query_term)

            len_d1 = len([
                term
                for term in context.terms(document1)
                if term != query_term
            ])
            len_d2 = len([
                term
                for term in context.terms(document2)
                if term != query_term
            ])

            if len_d1 == len_d2:
                if tf_d1 > tf_d2:
                    sum_document1 += 1
                elif tf_d2 > tf_d1:
                    sum_document2 += 1

        return strictly_greater(sum_document1, sum_document2)
