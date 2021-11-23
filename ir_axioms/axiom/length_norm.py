from ir_axioms.axiom import Axiom
from ir_axioms.axiom.utils import (
    approximately_equal, strictly_less, strictly_greater
)
from ir_axioms.model import Query, RankedDocument
from ir_axioms.model.context import RerankingContext


class LNC1(Axiom):
    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        if not all(
                approximately_equal(
                    context.term_frequency(document1.content, term),
                    context.term_frequency(document2.content, term)
                )
                for term in context.term_set(query.title)
        ):
            return 0

        # Prefer the shorter document.
        return strictly_less(
            len(context.terms(document1.content)),
            len(context.terms(document2.content)),
        )


class LNC2(Axiom):
    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        # LNC2 makes no sense as implemented and was useless in previous trials
        # TODO: May we delete it?
        return 0


class TF_LNC(Axiom):
    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):

        sd1 = 0
        sd2 = 0

        for t in context.term_set(query.title):
            tf_d1 = context.term_frequency(document1.content, t)
            tf_d2 = context.term_frequency(document2.content, t)
            len_d1 = len(context.terms(document1.content))
            len_d2 = len(context.terms(document2.content))
            tf_len_d1 = len_d1 + tf_d2 - tf_d1
            tf_len_d2 = len_d2 + tf_d1 - tf_d2
            if tf_d1 > tf_d2 and len_d1 == tf_len_d2:
                sd1 += 1
            elif tf_d2 > tf_d1 and len_d2 == tf_len_d1:
                sd2 += 1

        return strictly_greater(sd1, sd2)
