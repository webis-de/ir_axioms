from collections.abc import Mapping, Set, Sequence
from dataclasses import dataclass, field
from itertools import chain, combinations
from math import isclose
from typing import Final

from injector import inject, NoInject
from numpy import array, float_
from tqdm.auto import tqdm

from ir_axioms.axiom.base import Axiom
from ir_axioms.axiom.precondition import PreconditionMixin
from ir_axioms.precondition.base import Precondition
from ir_axioms.precondition.length import LEN
from ir_axioms.axiom.utils import strictly_greater
from ir_axioms.model import PreferenceMatrix, Query, Document, Preference
from ir_axioms.tools import IndexStatistics, TermTokenizer, TextContents, TextStatistics
from ir_axioms.utils.lazy import lazy_inject


@inject
@dataclass(frozen=True, kw_only=True)
class Tfc1Axiom(PreconditionMixin[Query, Document], Axiom[Query, Document]):
    text_contents: TextContents[Query | Document]
    term_tokenizer: TermTokenizer
    text_statistics: TextStatistics[Document]
    precondition: NoInject[Precondition[Query, Document]] = field(default_factory=LEN)
    margin_fraction: NoInject[float] = 0.1

    def _preference(
        self,
        term_frequency_sum1: float,
        term_frequency_sum2: float,
    ) -> Preference:
        if isclose(
            term_frequency_sum1,
            term_frequency_sum2,
            rel_tol=self.margin_fraction,
        ):
            return 0

        return strictly_greater(term_frequency_sum1, term_frequency_sum2)

    def preference(
        self,
        input: Query,
        output1: Document,
        output2: Document,
    ) -> Preference:
        query_terms = self.term_tokenizer.terms_unordered(
            self.text_contents.contents(input),
        )

        term_frequency_sum1 = sum(
            self.text_statistics.term_frequency(output1, term) for term in query_terms
        )
        term_frequency_sum2 = sum(
            self.text_statistics.term_frequency(output2, term) for term in query_terms
        )

        return self._preference(term_frequency_sum1, term_frequency_sum2)

    def preferences(
        self,
        input: Query,
        outputs: Sequence[Document],
    ) -> PreferenceMatrix:
        query_terms = self.term_tokenizer.terms_unordered(
            self.text_contents.contents(input),
        )

        term_frequency_sums = [
            sum(
                self.text_statistics.term_frequency(output, term)
                for term in query_terms
            )
            for output in tqdm(
                outputs,
                desc="Sum term frequencies",
                unit="document",
                disable=not self.verbose,
            )
        ]

        return array(
            [
                self._preference(term_frequency_sum1, term_frequency_sum2)
                for term_frequency_sum1 in term_frequency_sums
                for term_frequency_sum2 in term_frequency_sums
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


TFC1: Final = lazy_inject(Tfc1Axiom)


@inject
@dataclass(frozen=True, kw_only=True)
class Tfc3Axiom(PreconditionMixin[Query, Document], Axiom[Query, Document]):
    text_contents: TextContents[Query | Document]
    term_tokenizer: TermTokenizer
    index_statistics: IndexStatistics
    text_statistics: TextStatistics[Document]
    precondition: NoInject[Precondition[Query, Document]] = field(default_factory=LEN)
    margin_fraction: NoInject[float] = 0.1

    def preference(
        self,
        input: Query,
        output1: Document,
        output2: Document,
    ) -> Preference:
        query_unique_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(input),
        )

        sum_document1 = 0
        sum_document2 = 0
        for query_term1, query_term2 in combinations(query_unique_terms, 2):
            term_discrimination1 = self.index_statistics.inverse_document_frequency(
                query_term1
            )
            term_discrimination2 = self.index_statistics.inverse_document_frequency(
                query_term2
            )

            if isclose(
                term_discrimination1,
                term_discrimination2,
                rel_tol=self.margin_fraction,
            ):
                d1q1 = self.text_statistics.term_frequency(output1, query_term1)
                d2q1 = self.text_statistics.term_frequency(output2, query_term1)
                d1q2 = self.text_statistics.term_frequency(output1, query_term2)
                d2q2 = self.text_statistics.term_frequency(output2, query_term2)

                if d1q1 != 0 and d1q2 != 0:
                    if isclose(d2q1, d1q1 + d1q2) and d2q2 == 0:
                        sum_document1 += 1
                    if isclose(d2q2, d1q2 + d1q1) and d2q1 == 0:
                        sum_document1 += 1
                if d2q1 != 0 and d2q2 != 0:
                    if isclose(d1q1, d2q1 + d2q2) and d1q2 == 0:
                        sum_document2 += 1
                    if isclose(d1q2, d2q2 + d2q1) and d1q1 == 0:
                        sum_document2 += 1

        return strictly_greater(sum_document1, sum_document2)

    def preferences(
        self,
        input: Query,
        outputs: Sequence[Document],
    ) -> PreferenceMatrix:
        query_unique_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(input),
        )
        query_term_pairs = [
            (query_term1, query_term2)
            for query_term1, query_term2 in tqdm(
                combinations(query_unique_terms, 2),
                desc="Query term pairs",
                unit="pair",
                disable=not self.verbose,
            )
            if isclose(
                self.index_statistics.inverse_document_frequency(query_term1),
                self.index_statistics.inverse_document_frequency(query_term2),
                rel_tol=self.margin_fraction,
            )
        ]
        considered_query_terms = set(chain.from_iterable(query_term_pairs))
        term_frequencies = [
            {
                query_term: self.text_statistics.term_frequency(output, query_term)
                for query_term in considered_query_terms
            }
            for output in tqdm(
                outputs,
                desc="Term frequencies",
                unit="document",
                disable=not self.verbose,
            )
        ]

        return array(
            list(
                tqdm(
                    (
                        strictly_greater(
                            sum(
                                1
                                for query_term1, query_term2 in query_term_pairs
                                if term_frequencies[i1][query_term1] != 0
                                and term_frequencies[i1][query_term2] != 0
                                and (
                                    (
                                        isclose(
                                            term_frequencies[i2][query_term1],
                                            term_frequencies[i1][query_term1]
                                            + term_frequencies[i1][query_term2],
                                        )
                                        and term_frequencies[i2][query_term2] == 0
                                    )
                                    or (
                                        isclose(
                                            term_frequencies[i2][query_term2],
                                            term_frequencies[i1][query_term2]
                                            + term_frequencies[i1][query_term1],
                                        )
                                        and term_frequencies[i2][query_term1] == 0
                                    )
                                )
                            ),
                            sum(
                                1
                                for query_term1, query_term2 in query_term_pairs
                                if term_frequencies[i2][query_term1] != 0
                                and term_frequencies[i2][query_term2] != 0
                                and (
                                    (
                                        isclose(
                                            term_frequencies[i1][query_term1],
                                            term_frequencies[i2][query_term1]
                                            + term_frequencies[i2][query_term2],
                                        )
                                        and term_frequencies[i1][query_term2] == 0
                                    )
                                    or (
                                        isclose(
                                            term_frequencies[i1][query_term2],
                                            term_frequencies[i2][query_term2]
                                            + term_frequencies[i2][query_term1],
                                        )
                                        and term_frequencies[i1][query_term1] == 0
                                    )
                                )
                            ),
                        )
                        for i1 in range(len(outputs))
                        for i2 in range(len(outputs))
                    ),
                    desc="Compare",
                    unit="pair",
                    disable=not self.verbose,
                )
            ),
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


TFC3: Final = lazy_inject(Tfc3Axiom)


def _single_different_term_frequency(
    query_unique_terms: Set,
    term_frequencies1: Mapping[str, float],
    term_frequencies2: Mapping[str, float],
) -> bool:
    sum_term_frequency1 = 0.0
    sum_term_frequency2 = 0.0
    term_frequency_different = False
    for term in query_unique_terms:
        count1 = term_frequencies1.get(term, 0)
        count2 = term_frequencies2.get(term, 0)
        if count1 != count2:
            term_frequency_different = True
        sum_term_frequency1 += count1
        sum_term_frequency2 += count2

    return (
        isclose(sum_term_frequency1, sum_term_frequency2) and term_frequency_different
    )


def _modified_tdc_preference(
    query_term_pairs: Sequence[tuple[str, str]],
    query_term_frequencies: Mapping[str, float],
    term_frequencies1: Mapping[str, float],
    term_frequencies2: Mapping[str, float],
) -> Preference:
    score1 = 0
    score2 = 0

    for query_term1, query_term2 in query_term_pairs:
        tf_d1_qt1 = term_frequencies1.get(query_term1, 0)
        tf_d1_qt2 = term_frequencies1.get(query_term2, 0)
        tf_d2_qt1 = term_frequencies2.get(query_term1, 0)
        tf_d2_qt2 = term_frequencies2.get(query_term2, 0)
        tf_q_qt1 = query_term_frequencies.get(query_term1, 0)
        tf_q_qt2 = query_term_frequencies.get(query_term2, 0)

        if not (
            (isclose(tf_d1_qt1, tf_d2_qt2) and isclose(tf_d1_qt2, tf_d2_qt1))
            or tf_q_qt1 >= tf_q_qt2
        ):
            # Term pair is valid.
            continue

        if tf_q_qt1 < tf_q_qt2 and (tf_d1_qt1 != tf_d2_qt2 or tf_d1_qt2 != tf_d2_qt1):
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


@inject
@dataclass(frozen=True, kw_only=True)
class ModifiedTdcAxiom(Axiom[Query, Document]):
    """
    Modified TDC as in:

    Shi, S., Wen, J.R., Yu, Q., Song, R., Ma, W.Y.: Gravitation-based model
    for information retrieval. In: SIGIR ’05.
    """

    text_contents: TextContents[Query | Document]
    term_tokenizer: TermTokenizer
    index_statistics: IndexStatistics
    text_statistics: TextStatistics[Query | Document]

    def preference(
        self,
        input: Query,
        output1: Document,
        output2: Document,
    ) -> Preference:
        query_unique_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(input),
        )
        term_frequencies1 = self.text_statistics.term_frequencies(output1)
        term_frequencies2 = self.text_statistics.term_frequencies(output2)

        if not _single_different_term_frequency(
            query_unique_terms=query_unique_terms,
            term_frequencies1=term_frequencies1,
            term_frequencies2=term_frequencies2,
        ):
            return 0

        return _modified_tdc_preference(
            query_term_pairs=self._query_term_pairs(query_unique_terms),
            query_term_frequencies=self.text_statistics.term_frequencies(input),
            term_frequencies1=term_frequencies1,
            term_frequencies2=term_frequencies2,
        )

    def _query_term_pairs(
        self,
        query_unique_terms: Set[str],
    ) -> Sequence[tuple[str, str]]:
        """
        Query term pairs with different term discrimination values,
        ordered such that the first term has the higher IDF.
        """
        inverse_document_frequencies = (
            self.index_statistics.inverse_document_frequencies(query_unique_terms)
        )
        query_term_pairs = []
        for query_term1, query_term2 in combinations(query_unique_terms, 2):
            idf_qt1 = inverse_document_frequencies[query_term1]
            idf_qt2 = inverse_document_frequencies[query_term2]

            if isclose(idf_qt1, idf_qt2):
                # Skip query term pair, as they are equally rare.
                # We would introduce randomness into this axiom otherwise.
                continue

            if idf_qt1 < idf_qt2:
                # Query term 1 is rarer. Swap query terms.
                query_term1, query_term2 = query_term2, query_term1

            query_term_pairs.append((query_term1, query_term2))
        return query_term_pairs

    def preferences(
        self,
        input: Query,
        outputs: Sequence[Document],
    ) -> PreferenceMatrix:
        query_unique_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(input),
        )
        query_term_pairs = self._query_term_pairs(query_unique_terms)
        query_term_frequencies = self.text_statistics.term_frequencies(input)
        term_frequencies = [
            self.text_statistics.term_frequencies(output)
            for output in tqdm(
                outputs,
                desc="Term frequencies",
                unit="document",
                disable=not self.verbose,
            )
        ]
        return array(
            [
                (
                    _modified_tdc_preference(
                        query_term_pairs=query_term_pairs,
                        query_term_frequencies=query_term_frequencies,
                        term_frequencies1=term_frequencies1,
                        term_frequencies2=term_frequencies2,
                    )
                    if _single_different_term_frequency(
                        query_unique_terms=query_unique_terms,
                        term_frequencies1=term_frequencies1,
                        term_frequencies2=term_frequencies2,
                    )
                    else 0
                )
                for term_frequencies1 in term_frequencies
                for term_frequencies2 in term_frequencies
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


M_TDC: Final = lazy_inject(ModifiedTdcAxiom)


@inject
@dataclass(frozen=True, kw_only=True)
class LenModifiedTdcAxiom(PreconditionMixin[Query, Document], ModifiedTdcAxiom):
    precondition: NoInject[Precondition[Query, Document]] = field(default_factory=LEN)


LEN_M_TDC: Final = lazy_inject(LenModifiedTdcAxiom)
