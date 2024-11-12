from abc import ABC
from dataclasses import dataclass
from math import nan

from axioms.axiom.base import Axiom
from axioms.axiom.utils import strictly_greater, approximately_equal
from axioms.model import Query, RankedDocument, IndexContext
from axioms.modules.similarity import (
    TermSimilarityMixin, WordNetSynonymSetTermSimilarityMixin
)


@dataclass(frozen=True)
class _STMC1(Axiom, TermSimilarityMixin, ABC):

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        query_terms = context.term_set(query)
        document1_terms = context.term_set(document1)
        document2_terms = context.term_set(document2)

        return strictly_greater(
            self.average_similarity(document1_terms, query_terms),
            self.average_similarity(document2_terms, query_terms)
        )


@dataclass(frozen=True)
class STMC1(_STMC1, WordNetSynonymSetTermSimilarityMixin):
    name = "STMC1"


@dataclass(frozen=True)
class _STMC2(Axiom, TermSimilarityMixin, ABC):

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
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

        query_terms = context.term_set(query)
        document1_terms = context.term_set(document1)
        document2_terms = context.term_set(document2)
        document_terms = document1_terms | document2_terms

        non_query_terms = document_terms - query_terms

        most_similar_terms = self.most_similar_pair(
            query_terms,
            non_query_terms,
        )
        if most_similar_terms is None:
            return 0

        most_similar_query_term, most_similar_non_query_term = (
            most_similar_terms
        )

        def term_frequency_ratio(
                document_a: RankedDocument,
                document_b: RankedDocument
        ):
            tf_most_similar_query_term = context.term_frequency(
                document_b,
                most_similar_query_term
            )
            tf_most_similar_non_query_term = context.term_frequency(
                document_a,
                most_similar_non_query_term
            )
            if tf_most_similar_query_term <= 0:
                return nan
            return tf_most_similar_non_query_term / tf_most_similar_query_term

        if (
                len(document1_terms) > 0 and
                approximately_equal(
                    len(document2_terms) / len(document1_terms),
                    term_frequency_ratio(document2, document1),
                    margin_fraction=0.2
                )
        ):
            return 1
        elif (
                len(document2_terms) > 0 and
                approximately_equal(
                    len(document1_terms) / len(document2_terms),
                    term_frequency_ratio(document1, document2),
                    margin_fraction=0.2
                )
        ):
            return -1

        return 0


@dataclass(frozen=True)
class STMC2(_STMC2, WordNetSynonymSetTermSimilarityMixin):
    name = "STMC2"
