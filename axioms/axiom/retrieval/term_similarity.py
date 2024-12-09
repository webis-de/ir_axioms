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


@inject
@dataclass(frozen=True, kw_only=True)
class Stmc2Axiom(Axiom[Query, Document]):
    text_contents: TextContents[Union[Query, Document]]
    term_tokenizer: TermTokenizer
    term_similarity: TermSimilarity
    text_statistics: TextStatistics[Document]

    def preference(
        self,
        input: Query,
        output1: Document,
        output2: Document,
    ) -> Preference:
        """
        Given the most similar query term and non-query term,
        prefer the first document if
        the second document's non-query term frequency
        compared to the first document's query term frequency
        is similar to the second document's length
        compared to the first document's length.

        Note that the selection of the most similar query non-query term pair
        is non-deterministic if there are multiple equally most similar pairs.
        """

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

        document_terms = document1_unique_terms | document2_unique_terms

        non_query_terms = document_terms - query_unique_terms

        most_similar_terms = self.term_similarity.most_similar_pair(
            query_unique_terms,
            non_query_terms,
        )
        if most_similar_terms is None:
            return 0

        most_similar_query_term, most_similar_non_query_term = most_similar_terms

        def term_frequency_ratio(
            document_a: Document,
            document_b: Document,
        ):
            tf_most_similar_query_term = self.text_statistics.term_frequency(
                document_b, most_similar_query_term
            )
            tf_most_similar_non_query_term = self.text_statistics.term_frequency(
                document_a, most_similar_non_query_term
            )
            if tf_most_similar_query_term <= 0:
                return nan
            return tf_most_similar_non_query_term / tf_most_similar_query_term

        if len(document1_unique_terms) > 0 and approximately_equal(
            len(document2_unique_terms) / len(document1_unique_terms),
            term_frequency_ratio(output2, output1),
            margin_fraction=0.2,
        ):
            return 1
        elif len(document2_unique_terms) > 0 and approximately_equal(
            len(document1_unique_terms) / len(document2_unique_terms),
            term_frequency_ratio(output1, output2),
            margin_fraction=0.2,
        ):
            return -1

        return 0


STMC2: Final = lazy_inject(Stmc2Axiom, injector)
