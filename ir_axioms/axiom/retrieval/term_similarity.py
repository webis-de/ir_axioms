from collections.abc import Collection, Set, Mapping, Sequence
from dataclasses import dataclass
from math import isclose, nan
from typing import Final

from injector import inject, NoInject
from numpy import array, float_
from tqdm.auto import tqdm

from ir_axioms.axiom.base import Axiom
from ir_axioms.axiom.utils import strictly_greater
from ir_axioms.model import Query, Document, Preference, PreferenceMatrix
from ir_axioms.tools import TextContents, TermTokenizer, TermSimilarity, TextStatistics
from ir_axioms.utils.lazy import lazy_inject


@inject
@dataclass(frozen=True, kw_only=True)
class Stmc1Axiom(Axiom[Query, Document]):
    text_contents: TextContents[Query | Document]
    term_tokenizer: TermTokenizer
    term_similarity: TermSimilarity

    def preference(
        self,
        input: Query,
        output1: Document,
        output2: Document,
    ) -> Preference:
        query_unique_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(input),
        )
        document1_unique_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(output1),
        )
        document2_unique_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(output2),
        )
        average_similarity1 = self.term_similarity.average_similarity(
            document1_unique_terms, query_unique_terms
        )
        average_similarity2 = self.term_similarity.average_similarity(
            document2_unique_terms, query_unique_terms
        )
        return strictly_greater(average_similarity1, average_similarity2)

    def preferences(
        self,
        input: Query,
        outputs: Sequence[Document],
    ) -> PreferenceMatrix:
        query_unique_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(input),
        )
        document_unique_terms = (
            self.term_tokenizer.unique_terms(self.text_contents.contents(output))
            for output in outputs
        )
        average_similarities = [
            self.term_similarity.average_similarity(document_terms, query_unique_terms)
            for document_terms in tqdm(
                document_unique_terms,
                total=len(outputs),
                desc="Compute avg. similarities",
                unit="document",
                disable=not self.verbose,
            )
        ]

        return array(
            [
                strictly_greater(average_similarities1, average_similarities2)
                for average_similarities1 in average_similarities
                for average_similarities2 in average_similarities
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


STMC1: Final = lazy_inject(Stmc1Axiom)


def _safe_ratio(a: float, b: float) -> float:
    if b == 0:
        return nan
    else:
        return a / b


def _stmc2_preference(
    max_similarity_pairs: Set[tuple[str, str]],
    document1_length: int,
    document2_length: int,
    term_frequencies1: Mapping[str, float],
    term_frequencies2: Mapping[str, float],
    margin_fraction: float,
) -> Preference:
    if len(max_similarity_pairs) == 0:
        return 0

    if all(
        isclose(
            _safe_ratio(document2_length, document1_length),
            _safe_ratio(
                term_frequencies2.get(non_query_term, 0),
                term_frequencies1.get(query_term, 0),
            ),
            rel_tol=margin_fraction,
        )
        for query_term, non_query_term in max_similarity_pairs
    ):
        return 1
    elif all(
        isclose(
            _safe_ratio(document1_length, document2_length),
            _safe_ratio(
                term_frequencies1.get(non_query_term, 0),
                term_frequencies2.get(query_term, 0),
            ),
            rel_tol=margin_fraction,
        )
        for query_term, non_query_term in max_similarity_pairs
    ):
        return -1

    return 0


@inject
@dataclass(frozen=True, kw_only=True)
class Stmc2Axiom(Axiom[Query, Document]):
    text_contents: TextContents[Query | Document]
    term_tokenizer: TermTokenizer
    term_similarity: TermSimilarity
    text_statistics: TextStatistics[Document]
    margin_fraction: NoInject[float] = 0.2

    def preference(
        self,
        input: Query,
        output1: Document,
        output2: Document,
    ) -> Preference:
        """
        Given the most similar pair of a query term and non-query term,
        prefer the first document if the second document's non-query term frequency
        compared to the first document's query term frequency
        is similar to the second document's length
        compared to the first document's length.
        """

        query_unique_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(input),
        )
        document1_terms = self.term_tokenizer.terms_unordered(
            self.text_contents.contents(output1),
        )
        document2_terms = self.term_tokenizer.terms_unordered(
            self.text_contents.contents(output2),
        )

        document1_length = len(document1_terms)
        document2_length = len(document2_terms)

        document_unique_terms = set(document1_terms) | set(document2_terms)
        non_query_terms = document_unique_terms - query_unique_terms

        _, max_similarity_pairs = self.term_similarity.max_similarity_pairs(
            query_unique_terms,
            non_query_terms,
        )
        if len(max_similarity_pairs) == 0:
            return 0

        term_frequencies1 = self.text_statistics.term_frequencies(output1)
        term_frequencies2 = self.text_statistics.term_frequencies(output2)

        return _stmc2_preference(
            max_similarity_pairs=max_similarity_pairs,
            document1_length=document1_length,
            document2_length=document2_length,
            term_frequencies1=term_frequencies1,
            term_frequencies2=term_frequencies2,
            margin_fraction=self.margin_fraction,
        )

    def _document_max_similarity_pairs(
        self,
        query_unique_terms: Set[str],
        document_terms: Collection[str],
    ) -> tuple[float, Set[tuple[str, str]]]:
        """
        Most similar pairs of a query term and a non-query term of one document,
        along with their similarity.
        """
        return self.term_similarity.max_similarity_pairs(
            query_unique_terms,
            set(document_terms) - query_unique_terms,
        )

    def preferences(
        self,
        input: Query,
        outputs: Sequence[Document],
    ) -> PreferenceMatrix:
        query_unique_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(input),
        )
        document_terms = list(
            self.term_tokenizer.iter_terms_unordered(
                self.text_contents.iter_contents(outputs),
            )
        )
        term_frequencies = [
            self.text_statistics.term_frequencies(output) for output in outputs
        ]
        # The most similar pairs over the union of two documents' non-query terms are the most similar pairs of whichever document has the higher maximum.
        max_similarity_pairs = [
            self._document_max_similarity_pairs(query_unique_terms, terms)
            for terms in tqdm(
                document_terms,
                desc="Max. similarity pairs",
                unit="document",
                disable=not self.verbose,
            )
        ]

        def pair_preference(i1: int, i2: int) -> Preference:
            max_similarity1, pairs1 = max_similarity_pairs[i1]
            max_similarity2, pairs2 = max_similarity_pairs[i2]
            max_similarity = max(max_similarity1, max_similarity2)
            pairs = {
                # Include maximum similiarity pairs from each document whose max similarity matches the global maximum similarity.
                *(pairs1 if max_similarity1 >= max_similarity else set()),
                *(pairs2 if max_similarity2 >= max_similarity else set()),
            }
            return _stmc2_preference(
                max_similarity_pairs=pairs,
                document1_length=len(document_terms[i1]),
                document2_length=len(document_terms[i2]),
                term_frequencies1=term_frequencies[i1],
                term_frequencies2=term_frequencies[i2],
                margin_fraction=self.margin_fraction,
            )

        return array(
            [
                pair_preference(i1, i2)
                for i1 in range(len(outputs))
                for i2 in range(len(outputs))
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


STMC2: Final = lazy_inject(Stmc2Axiom)
