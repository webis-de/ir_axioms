from dataclasses import dataclass
from math import nan
from typing import Final, Union

from injector import inject

from axioms.axiom.base import Axiom
from axioms.axiom.utils import strictly_greater, approximately_equal
from axioms.dependency_injection import injector
from axioms.model import Query, Document, Preference
from axioms.tools import TextContents, TermTokenizer, TermSimilarity, TextStatistics
from axioms.utils.lazy import lazy_inject


@inject
@dataclass(frozen=True, kw_only=True)
class Stmc1Axiom(Axiom[Query, Document]):
    text_contents: TextContents[Union[Query, Document]]
    term_tokenizer: TermTokenizer
    term_similarity: TermSimilarity

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
        document1_terms = self.term_tokenizer.terms(
            self.text_contents.contents(output1),
        )
        document1_unique_terms = set(document1_terms)
        document2_terms = self.term_tokenizer.terms(
            self.text_contents.contents(output2),
        )
        document2_unique_terms = set(document2_terms)

        return strictly_greater(
            self.term_similarity.average_similarity(
                document1_unique_terms, query_unique_terms
            ),
            self.term_similarity.average_similarity(
                document2_unique_terms, query_unique_terms
            ),
        )


STMC1: Final = lazy_inject(Stmc1Axiom, injector)


def _safe_ratio(a: float, b: float) -> float:
    if b == 0:
        return nan
    else:
        return a / b


@inject
@dataclass(frozen=True, kw_only=True)
class Stmc2Axiom(Axiom[Query, Document]):
    text_contents: TextContents[Union[Query, Document]]
    term_tokenizer: TermTokenizer
    term_similarity: TermSimilarity
    text_statistics: TextStatistics[Document]
    margin_fraction: float = 0.2

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
        document1_terms = self.term_tokenizer.terms(
            self.text_contents.contents(output1),
        )
        document1_unique_terms = set(document1_terms)
        document2_terms = self.term_tokenizer.terms(
            self.text_contents.contents(output2),
        )
        document2_unique_terms = set(document2_terms)

        document_terms = document1_unique_terms | document2_unique_terms

        non_query_terms = document_terms - query_unique_terms

        max_similarity_pairs = self.term_similarity.max_similarity_pairs(
            query_unique_terms,
            non_query_terms,
        )
        if len(max_similarity_pairs) == 0:
            return 0

        document1_length = len(document1_terms)
        document2_length = len(document2_terms)

        if all(
            approximately_equal(
                _safe_ratio(document2_length, document1_length),
                _safe_ratio(
                    self.text_statistics.term_frequency(output2, non_query_term),
                    self.text_statistics.term_frequency(output1, query_term),
                ),
                margin_fraction=self.margin_fraction,
            )
            for query_term, non_query_term in max_similarity_pairs
        ):
            return 1
        elif all(
            approximately_equal(
                _safe_ratio(document1_length, document2_length),
                _safe_ratio(
                    self.text_statistics.term_frequency(output1, non_query_term),
                    self.text_statistics.term_frequency(output2, query_term),
                ),
                margin_fraction=self.margin_fraction,
            )
            for query_term, non_query_term in max_similarity_pairs
        ):
            return -1

        return 0


STMC2: Final = lazy_inject(Stmc2Axiom, injector)
