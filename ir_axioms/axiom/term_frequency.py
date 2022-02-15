from dataclasses import dataclass
from itertools import combinations
from math import isclose

from ir_axioms.axiom.base import Axiom
from ir_axioms.axiom.preconditions import LEN_Mixin
from ir_axioms.axiom.utils import approximately_equal, strictly_greater
from ir_axioms.model import Query, RankedDocument, IndexContext


@dataclass(frozen=True)
class _TFC1(Axiom):

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        term_frequency1: float = 0
        term_frequency2: float = 0
        for qt in context.terms(query):
            term_frequency1 += context.term_frequency(document1, qt)
            term_frequency2 += context.term_frequency(document2, qt)

        if approximately_equal(term_frequency1, term_frequency2):
            # Less than 10% difference.
            return 0

        return strictly_greater(term_frequency1, term_frequency2)


@dataclass(frozen=True)
class TFC1(LEN_Mixin, _TFC1):
    name = "TFC1"


@dataclass(frozen=True)
class _TFC3(Axiom):

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        sum_document1 = 0
        sum_document2 = 0

        query_terms = context.term_set(query)
        for query_term1, query_term2 in combinations(query_terms, 2):
            term_discrimination1 = context.inverse_document_frequency(
                query_term1
            )
            term_discrimination2 = context.inverse_document_frequency(
                query_term2
            )

            if approximately_equal(term_discrimination1, term_discrimination2):
                d1q1 = context.term_frequency(document1, query_term1)
                d2q1 = context.term_frequency(document2, query_term1)
                d1q2 = context.term_frequency(document1, query_term2)
                d2q2 = context.term_frequency(document2, query_term2)

                if d1q1 != 0 and d1q2 != 0:
                    if isclose(d2q1, d1q1 + d1q2) and isclose(d2q2, 0):
                        sum_document1 += 1
                    if isclose(d2q2, d1q2 + d1q1) and isclose(d2q1, 0):
                        sum_document1 += 1
                if d2q1 != 0 and d2q2 != 0:
                    if isclose(d1q1, d2q1 + d2q2) and isclose(d1q2, 0):
                        sum_document2 += 1
                    if isclose(d1q2, d2q2 + d2q1) and isclose(d1q1, 0):
                        sum_document2 += 1

        return strictly_greater(sum_document1, sum_document2)


@dataclass(frozen=True)
class TFC3(LEN_Mixin, _TFC3):
    name = "TFC3"


def _single_different_term_frequency(
        context: IndexContext,
        query: Query,
        document1: RankedDocument,
        document2: RankedDocument
):
    query_terms = context.term_set(query)
    sum_term_frequency1 = 0
    sum_term_frequency2 = 0
    term_frequency_different = False
    for term in query_terms:
        count1 = context.term_frequency(document1, term)
        count2 = context.term_frequency(document2, term)
        if count1 != count2:
            term_frequency_different = True
        sum_term_frequency1 += count1
        sum_term_frequency2 += count2

    return (
            isclose(sum_term_frequency1, sum_term_frequency2) and
            term_frequency_different
    )


@dataclass(frozen=True)
class M_TDC(Axiom):
    """
    Modified TDC as in:

    Shi, S., Wen, J.R., Yu, Q., Song, R., Ma, W.Y.: Gravitation-based model
    for information retrieval. In: SIGIR ’05.
    """
    name = "M-TDC"

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        if not _single_different_term_frequency(
                context,
                query,
                document1,
                document2
        ):
            return 0

        query_terms = context.term_set(query)

        score1 = 0
        score2 = 0

        for query_term1, query_term2 in combinations(query_terms, 2):
            idf_qt1 = context.inverse_document_frequency(query_term1)
            idf_qt2 = context.inverse_document_frequency(query_term2)

            if isclose(idf_qt1, idf_qt2):
                # Skip query term pair, as they are equally rare.
                # We would introduce randomness into this axiom otherwise.
                continue

            if idf_qt1 < idf_qt2:
                # Query term 1 is rarer. Swap query terms.
                query_term1, query_term2 = query_term2, query_term1

            tf_d1_qt1 = context.term_frequency(document1, query_term1)
            tf_d1_qt2 = context.term_frequency(document1, query_term2)
            tf_d2_qt1 = context.term_frequency(document2, query_term1)
            tf_d2_qt2 = context.term_frequency(document2, query_term2)
            tf_q_qt1 = context.term_frequency(query, query_term1)
            tf_q_qt2 = context.term_frequency(query, query_term2)

            if not (
                    (
                            isclose(tf_d1_qt1, tf_d2_qt2) and
                            isclose(tf_d1_qt2, tf_d2_qt1)
                    ) or
                    tf_q_qt1 >= tf_q_qt2
            ):
                # Term pair is valid.
                continue

            if (
                    tf_q_qt1 < tf_q_qt2 and
                    (tf_d1_qt1 != tf_d2_qt2 or tf_d1_qt2 != tf_d2_qt1)
            ):
                # Term pair is valid.
                continue

            # Document with more occurrences of query term 1 gets a point.
            if tf_d1_qt1 > tf_d2_qt1:
                score1 += 1
            elif tf_d1_qt1 < tf_d2_qt1:
                score2 += 1
            else:
                # Don't change score.
                pass

        return strictly_greater(score1, score2)


@dataclass(frozen=True)
class LEN_M_TDC(LEN_Mixin, M_TDC):
    name = "LEN-M-TDC"
