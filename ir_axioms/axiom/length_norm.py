from ir_axioms.axiom import Axiom
from ir_axioms.axiom.utils import (
    approximately_equal, strictly_less, strictly_greater
)
from ir_axioms.model import Query, RankedDocument
from ir_axioms.model.context import RerankingContext


class LNC1(Axiom):
    name = "LNC1"

    def preference(
            self,
            context: RerankingContext,
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


class TF_LNC(Axiom):
    name = "TF_LNC"

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):

        sd1 = 0
        sd2 = 0

        for term in context.term_set(query):
            tf_d1 = context.term_frequency(document1, term)
            tf_d2 = context.term_frequency(document2, term)
            len_d1 = len(context.terms(document1))
            len_d2 = len(context.terms(document2))
            tf_len_d1 = len_d1 + tf_d2 - tf_d1
            tf_len_d2 = len_d2 + tf_d1 - tf_d2
            if tf_d1 > tf_d2 and len_d1 == tf_len_d2:
                sd1 += 1
            elif tf_d2 > tf_d1 and len_d2 == tf_len_d1:
                sd2 += 1

        return strictly_greater(sd1, sd2)
