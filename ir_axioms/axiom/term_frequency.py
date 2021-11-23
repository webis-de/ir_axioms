from dataclasses import dataclass
from itertools import combinations
from math import floor

from ir_axioms.axiom import Axiom
from ir_axioms.axiom.utils import (
    approximately_equal, strictly_greater, approximately_same_length
)
from ir_axioms.model import Query, RankedDocument
from ir_axioms.model.context import RerankingContext


class TFC1(Axiom):
    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        if not approximately_same_length(context, document1, document2):
            return 0

        tf1 = 0
        tf2 = 0
        for qt in context.terms(query.title):
            tf1 += context.term_frequency(document1.content, qt)
            tf2 += context.term_frequency(document2.content, qt)

        if approximately_equal(tf1, tf2):
            # Less than 10% difference.
            return 0

        return strictly_greater(tf1, tf2)


class TFC3(Axiom):

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        if not approximately_same_length(context, document1, document2):
            return 0

        sd1 = 0
        sd2 = 0

        query_terms = set(context.terms(query.title))
        for qt1, qt2 in combinations(query_terms, 2):
            td1 = floor(100 * context.inverse_document_frequency(qt1))
            td2 = floor(100 * context.inverse_document_frequency(qt2))

            if approximately_equal(td1, td2):
                d1q1 = context.term_frequency(document1.content, qt1)
                d2q1 = context.term_frequency(document2.content, qt1)
                d1q2 = context.term_frequency(document1.content, qt2)
                d2q2 = context.term_frequency(document2.content, qt2)

                sd1 += (
                        (d2q1 == d1q1 + d1q2) and
                        (d2q2 == 0) and
                        (d1q1 != 0) and
                        (d1q2 != 0)
                )
                sd2 += (
                        (d1q1 == d2q1 + d2q2) and
                        (d1q2 == 0) and
                        (d2q1 != 0) and
                        (d2q2 != 0)
                )

        return strictly_greater(sd1, sd2)


class M_TDC(Axiom):
    """
    Modified TDC as in:

    Shi, S., Wen, J.R., Yu, Q., Song, R., Ma, W.Y.: Gravitation-based model
    for information retrieval. In: SIGIR ’05.
    """

    @staticmethod
    def precondition(
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        query_terms = context.term_set(query.title)
        sum_term_frequency1 = 0
        sum_term_frequency2 = 0
        one_count_different = False
        for t in query_terms:
            count1 = context.term_frequency(document1.content, t)
            count2 = context.term_frequency(document2.content, t)
            if count1 != count2:
                one_count_different = True
            sum_term_frequency1 += count1
            sum_term_frequency2 += count2

        return (
                sum_term_frequency1 == sum_term_frequency2 and
                one_count_different
        )

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        if not self.precondition(context, query, document1, document2):
            return 0

        query_terms = context.term_set(query.title)

        score = 0

        for qt1, qt2 in combinations(query_terms, 2):

            if (
                    context.inverse_document_frequency(qt1) <
                    context.inverse_document_frequency(qt2)
            ):
                # Query term 1 is rarer. Swap query terms.
                qt1, qt2 = qt2, qt1

            tf_d1_qt1 = context.term_frequency(document1.content, qt1)
            tf_d1_qt2 = context.term_frequency(document1.content, qt2)
            tf_d2_qt1 = context.term_frequency(document2.content, qt1)
            tf_d2_qt2 = context.term_frequency(document2.content, qt2)
            tf_q_qt1 = context.term_frequency(query.title, qt1)
            tf_q_qt2 = context.term_frequency(query.title, qt2)
            if not (
                    (
                            tf_d1_qt1 == tf_d2_qt2 and
                            tf_d1_qt2 == tf_d2_qt1
                    ) or
                    tf_q_qt1 > tf_q_qt2
            ):
                # Term pair is valid.
                continue

            # Document with more occurrences of query term 1 gets a point.
            difference = tf_d1_qt1 - tf_d2_qt1
            if difference > 0:
                score += 1
            elif difference < 0:
                score -= 1
            else:
                # Don't change score.
                pass

        return strictly_greater(score, 0)


@dataclass
class LEN_M_TDC(M_TDC):
    """
    Modified M_TDC

    The precondition for the documents' lengths can be varied.
    Default margin fraction: 0.1
    """
    margin_fraction: float = 0.1

    def precondition(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        if not approximately_equal(
                len(context.terms(document1.content)),
                len(context.terms(document2.content)),
                margin_fraction=self.margin_fraction
        ):
            return False

        return super(LEN_M_TDC, self).precondition(
            context,
            query,
            document1,
            document2
        )
