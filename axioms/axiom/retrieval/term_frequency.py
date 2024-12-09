from dataclasses import dataclass, field
from itertools import combinations
from math import isclose
from typing import AbstractSet, Final, Union

from injector import inject, NoInject

from axioms.axiom.base import Axiom
from axioms.axiom.precondition import PreconditionMixin
from axioms.dependency_injection import injector
from axioms.precondition.base import Precondition
from axioms.precondition.length import LEN
from axioms.axiom.utils import approximately_equal, strictly_greater
from axioms.model import Query, Document, Preference
from axioms.tools import IndexStatistics, TermTokenizer, TextContents, TextStatistics
from axioms.utils.lazy import lazy_inject


@inject
@dataclass(frozen=True, kw_only=True)
class Tfc1Axiom(PreconditionMixin[Query, Document], Axiom[Query, Document]):
    text_contents: TextContents[Union[Query, Document]]
    term_tokenizer: TermTokenizer
    text_statistics: TextStatistics[Document]
    precondition: NoInject[Precondition[Query, Document]] = field(default_factory=LEN)

    def preference(
        self,
        input: Query,
        output1: Document,
        output2: Document,
    ) -> Preference:
        query_terms = self.term_tokenizer.terms(
            self.text_contents.contents(input),
        )

        term_frequency1: float = 0
        term_frequency2: float = 0
        for qt in query_terms:
            term_frequency1 += self.text_statistics.term_frequency(output1, qt)
            term_frequency2 += self.text_statistics.term_frequency(output2, qt)

        if approximately_equal(term_frequency1, term_frequency2):
            # Less than 10% difference.
            return 0

        return strictly_greater(term_frequency1, term_frequency2)


TFC1: Final = lazy_inject(Tfc1Axiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class Tfc3Axiom(PreconditionMixin[Query, Document], Axiom[Query, Document]):
    text_contents: TextContents[Union[Query, Document]]
    term_tokenizer: TermTokenizer
    index_statistics: IndexStatistics
    text_statistics: TextStatistics[Document]
    precondition: NoInject[Precondition[Query, Document]] = field(default_factory=LEN)

    def preference(
        self,
        input: Query,
        output1: Document,
        output2: Document,
    ) -> Preference:
        query_terms = self.term_tokenizer.terms(
            self.text_contents.contents(input),
        )
        query_unique_terms = set(query_terms)

        sum_document1 = 0
        sum_document2 = 0
        for query_term1, query_term2 in combinations(query_unique_terms, 2):
            term_discrimination1 = self.index_statistics.inverse_document_frequency(
                query_term1
            )
            term_discrimination2 = self.index_statistics.inverse_document_frequency(
                query_term2
            )

            if approximately_equal(term_discrimination1, term_discrimination2):
                d1q1 = self.text_statistics.term_frequency(output1, query_term1)
                d2q1 = self.text_statistics.term_frequency(output2, query_term1)
                d1q2 = self.text_statistics.term_frequency(output1, query_term2)
                d2q2 = self.text_statistics.term_frequency(output2, query_term2)

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


TFC3: Final = lazy_inject(Tfc3Axiom, injector)


def _single_different_term_frequency(
    query_unique_terms: AbstractSet,
    text_statistics: TextStatistics,
    output1: Document,
    output2: Document,
) -> bool:
    sum_term_frequency1 = 0.0
    sum_term_frequency2 = 0.0
    term_frequency_different = False
    for term in query_unique_terms:
        count1 = text_statistics.term_frequency(output1, term)
        count2 = text_statistics.term_frequency(output2, term)
        if count1 != count2:
            term_frequency_different = True
        sum_term_frequency1 += count1
        sum_term_frequency2 += count2

    return (
        isclose(sum_term_frequency1, sum_term_frequency2) and term_frequency_different
    )


@inject
@dataclass(frozen=True, kw_only=True)
class ModifiedTdcAxiom(Axiom[Query, Document]):
    """
    Modified TDC as in:

    Shi, S., Wen, J.R., Yu, Q., Song, R., Ma, W.Y.: Gravitation-based model
    for information retrieval. In: SIGIR ’05.
    """

    text_contents: TextContents[Union[Query, Document]]
    term_tokenizer: TermTokenizer
    index_statistics: IndexStatistics
    text_statistics: TextStatistics[Union[Query, Document]]

    def preference(
        self,
        input: Query,
        output1: Document,
        output2: Document,
    ) -> Preference:
        query_terms = self.term_tokenizer.terms(
            self.text_contents.contents(input),
        )
        query_unique_terms = set(query_terms)

        if not _single_different_term_frequency(
            query_unique_terms=query_unique_terms,
            text_statistics=self.text_statistics,
            output1=output1,
            output2=output2,
        ):
            return 0

        score1 = 0
        score2 = 0

        for query_term1, query_term2 in combinations(query_unique_terms, 2):
            idf_qt1 = self.index_statistics.inverse_document_frequency(query_term1)
            idf_qt2 = self.index_statistics.inverse_document_frequency(query_term2)

            if isclose(idf_qt1, idf_qt2):
                # Skip query term pair, as they are equally rare.
                # We would introduce randomness into this axiom otherwise.
                continue

            if idf_qt1 < idf_qt2:
                # Query term 1 is rarer. Swap query terms.
                query_term1, query_term2 = query_term2, query_term1

            tf_d1_qt1 = self.text_statistics.term_frequency(output1, query_term1)
            tf_d1_qt2 = self.text_statistics.term_frequency(output1, query_term2)
            tf_d2_qt1 = self.text_statistics.term_frequency(output2, query_term1)
            tf_d2_qt2 = self.text_statistics.term_frequency(output2, query_term2)
            tf_q_qt1 = self.text_statistics.term_frequency(input, query_term1)
            tf_q_qt2 = self.text_statistics.term_frequency(input, query_term2)

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


M_TDC: Final = lazy_inject(ModifiedTdcAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class LenModifiedTdcAxiom(PreconditionMixin[Query, Document], ModifiedTdcAxiom):
    precondition: NoInject[Precondition[Query, Document]] = field(default_factory=LEN)


LEN_M_TDC: Final = lazy_inject(LenModifiedTdcAxiom, injector)
