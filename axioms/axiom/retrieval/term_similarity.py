from dataclasses import dataclass
from math import nan
from typing import Final

from axioms.axiom.base import Axiom
from axioms.model.retrieval import get_index_context
from axioms.axiom.utils import strictly_greater, approximately_equal
from axioms.model import Query, RankedDocument, IndexContext
from axioms.tools import TermSimilarity, WordNetSynonymSetTermSimilarity


@dataclass(frozen=True, kw_only=True)
class Stmc1Axiom(Axiom):
    context: IndexContext
    term_similarity: TermSimilarity

    def preference(
        self, query: Query, document1: RankedDocument, document2: RankedDocument
    ):
        query_terms = self.context.term_set(query)
        document1_terms = self.context.term_set(document1)
        document2_terms = self.context.term_set(document2)

        return strictly_greater(
            self.term_similarity.average_similarity(document1_terms, query_terms),
            self.term_similarity.average_similarity(document2_terms, query_terms),
        )


STMC1: Final = Stmc1Axiom(
    context=get_index_context(),
    term_similarity=WordNetSynonymSetTermSimilarity(),
)


@dataclass(frozen=True, kw_only=True)
class Stmc2Axiom(Axiom):
    context: IndexContext
    term_similarity: TermSimilarity

    def preference(
        self, query: Query, document1: RankedDocument, document2: RankedDocument
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

        query_terms = self.context.term_set(query)
        document1_terms = self.context.term_set(document1)
        document2_terms = self.context.term_set(document2)
        document_terms = document1_terms | document2_terms

        non_query_terms = document_terms - query_terms

        most_similar_terms = self.term_similarity.most_similar_pair(
            query_terms,
            non_query_terms,
        )
        if most_similar_terms is None:
            return 0

        most_similar_query_term, most_similar_non_query_term = most_similar_terms

        def term_frequency_ratio(
            document_a: RankedDocument, document_b: RankedDocument
        ):
            tf_most_similar_query_term = self.context.term_frequency(
                document_b, most_similar_query_term
            )
            tf_most_similar_non_query_term = self.context.term_frequency(
                document_a, most_similar_non_query_term
            )
            if tf_most_similar_query_term <= 0:
                return nan
            return tf_most_similar_non_query_term / tf_most_similar_query_term

        if len(document1_terms) > 0 and approximately_equal(
            len(document2_terms) / len(document1_terms),
            term_frequency_ratio(document2, document1),
            margin_fraction=0.2,
        ):
            return 1
        elif len(document2_terms) > 0 and approximately_equal(
            len(document1_terms) / len(document2_terms),
            term_frequency_ratio(document1, document2),
            margin_fraction=0.2,
        ):
            return -1

        return 0


STMC2: Final = Stmc2Axiom(
    context=get_index_context(),
    term_similarity=WordNetSynonymSetTermSimilarity(),
)
