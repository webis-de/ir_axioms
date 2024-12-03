from dataclasses import dataclass
from itertools import combinations
from math import isclose
from typing import Final

from axioms.axiom.base import Axiom
from axioms.model.retrieval import get_index_context
from axioms.precondition.length import LEN
from axioms.axiom.utils import approximately_equal, strictly_greater
from axioms.model import Query, RankedDocument, IndexContext


@dataclass(frozen=True, kw_only=True)
class Tfc1Axiom(Axiom):
    context: IndexContext

    def preference(
        self, query: Query, document1: RankedDocument, document2: RankedDocument
    ) -> float:
        term_frequency1: float = 0
        term_frequency2: float = 0
        for qt in self.context.terms(query):
            term_frequency1 += self.context.term_frequency(document1, qt)
            term_frequency2 += self.context.term_frequency(document2, qt)

        if approximately_equal(term_frequency1, term_frequency2):
            # Less than 10% difference.
            return 0

        return strictly_greater(term_frequency1, term_frequency2)


TFC1: Final = Tfc1Axiom(
    context=get_index_context(),
).with_precondition(LEN)


@dataclass(frozen=True, kw_only=True)
class Tfc3Axiom(Axiom):
    context: IndexContext

    def preference(
        self, query: Query, document1: RankedDocument, document2: RankedDocument
    ) -> float:
        sum_document1 = 0
        sum_document2 = 0

        query_terms = self.context.term_set(query)
        for query_term1, query_term2 in combinations(query_terms, 2):
            term_discrimination1 = self.context.inverse_document_frequency(query_term1)
            term_discrimination2 = self.context.inverse_document_frequency(query_term2)

            if approximately_equal(term_discrimination1, term_discrimination2):
                d1q1 = self.context.term_frequency(document1, query_term1)
                d2q1 = self.context.term_frequency(document2, query_term1)
                d1q2 = self.context.term_frequency(document1, query_term2)
                d2q2 = self.context.term_frequency(document2, query_term2)

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


TFC3: Final = Tfc3Axiom(
    context=get_index_context(),
).with_precondition(LEN)


def _single_different_term_frequency(
    context: IndexContext,
    query: Query,
    document1: RankedDocument,
    document2: RankedDocument,
):
    query_terms = context.term_set(query)
    sum_term_frequency1 = 0.0
    sum_term_frequency2 = 0.0
    term_frequency_different = False
    for term in query_terms:
        count1 = context.term_frequency(document1, term)
        count2 = context.term_frequency(document2, term)
        if count1 != count2:
            term_frequency_different = True
        sum_term_frequency1 += count1
        sum_term_frequency2 += count2

    return (
        isclose(sum_term_frequency1, sum_term_frequency2) and term_frequency_different
    )


@dataclass(frozen=True, kw_only=True)
class ModifiedTdcAxiom(Axiom):
    """
    Modified TDC as in:

    Shi, S., Wen, J.R., Yu, Q., Song, R., Ma, W.Y.: Gravitation-based model
    for information retrieval. In: SIGIR ’05.
    """

    context: IndexContext

    def preference(
        self, query: Query, document1: RankedDocument, document2: RankedDocument
    ):
        if not _single_different_term_frequency(
            self.context, query, document1, document2
        ):
            return 0

        query_terms = self.context.term_set(query)

        score1 = 0
        score2 = 0

        for query_term1, query_term2 in combinations(query_terms, 2):
            idf_qt1 = self.context.inverse_document_frequency(query_term1)
            idf_qt2 = self.context.inverse_document_frequency(query_term2)

            if isclose(idf_qt1, idf_qt2):
                # Skip query term pair, as they are equally rare.
                # We would introduce randomness into this axiom otherwise.
                continue

            if idf_qt1 < idf_qt2:
                # Query term 1 is rarer. Swap query terms.
                query_term1, query_term2 = query_term2, query_term1

            tf_d1_qt1 = self.context.term_frequency(document1, query_term1)
            tf_d1_qt2 = self.context.term_frequency(document1, query_term2)
            tf_d2_qt1 = self.context.term_frequency(document2, query_term1)
            tf_d2_qt2 = self.context.term_frequency(document2, query_term2)
            tf_q_qt1 = self.context.term_frequency(query, query_term1)
            tf_q_qt2 = self.context.term_frequency(query, query_term2)

            if not (
                (isclose(tf_d1_qt1, tf_d2_qt2) and isclose(tf_d1_qt2, tf_d2_qt1))
                or tf_q_qt1 >= tf_q_qt2
            ):
                # Term pair is valid.
                continue

            if tf_q_qt1 < tf_q_qt2 and (
                tf_d1_qt1 != tf_d2_qt2 or tf_d1_qt2 != tf_d2_qt1
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


M_TDC: Final = ModifiedTdcAxiom(
    context=get_index_context(),
)
LEN_M_TDC: Final = M_TDC.with_precondition(LEN)
