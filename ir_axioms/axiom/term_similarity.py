from abc import ABC, abstractmethod
from functools import lru_cache, cached_property
from math import nan
from statistics import mean
from typing import Set, Tuple

from pymagnitude import Magnitude

from ir_axioms.axiom import Axiom
from ir_axioms.axiom.utils import synonym_set_similarity, strictly_greater, \
    approximately_equal
from ir_axioms.model import Query, RankedDocument
from ir_axioms.model.context import RerankingContext


class _TermSimilarity(ABC):
    @abstractmethod
    def similarity(self, term1: str, term2: str) -> float:
        pass


class _WordNetTermSimilarity(_TermSimilarity):
    @lru_cache
    def similarity(self, term1: str, term2: str) -> float:
        return synonym_set_similarity(term1, term2)


class _WordEmbeddingTermSimilarity(_TermSimilarity):
    @property
    @abstractmethod
    def embeddings_path(self) -> str:
        pass

    @cached_property
    def _embeddings(self):
        return Magnitude(self.embeddings_path, stream=True)

    @lru_cache
    def similarity(self, term1: str, term2: str):
        return self._embeddings.similarity(term1, term2)


class _FastTextWikiNewsTermSimilarity(_WordEmbeddingTermSimilarity):
    @property
    def embeddings_path(self) -> str:
        return (
            "https://magnitude.plasticity.ai/"
            "fasttext/medium/wiki-news-300d-1M-subword.magnitude"
        )


class _STMC1Base(Axiom, _TermSimilarity, ABC):

    def _average_similarity(self, terms1: Set[str], terms2: Set[str]) -> float:
        return mean(
            self.similarity(term1, term2)
            for term1 in terms1
            for term2 in terms2
        )

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        document1_terms = context.term_set(document1.content)
        document2_terms = context.term_set(document2.content)
        query_terms = context.term_set(query.title)

        return strictly_greater(
            self._average_similarity(document1_terms, query_terms),
            self._average_similarity(document2_terms, query_terms)
        )


class STMC1(_STMC1Base, _WordNetTermSimilarity):
    pass


class STMC1_f(_STMC1Base, _FastTextWikiNewsTermSimilarity):
    pass


class _STMC2Base(Axiom, _TermSimilarity, ABC):

    def _tuple_similarity(self, terms: Tuple[str, str]) -> float:
        term1, term2 = terms
        return self.similarity(term1, term2)

    def _most_similar_terms(
            self,
            terms1: Set[str],
            terms2: Set[str]
    ) -> Tuple[str, str]:
        return max(
            (
                (term1, term2)
                for term1 in terms1
                for term2 in terms2
            ),
            key=self._tuple_similarity
        )

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        document1_terms = context.term_set(document1.content)
        document2_terms = context.term_set(document2.content)
        document_terms = document1_terms & document2_terms
        query_terms = context.term_set(query.title)
        non_query_terms = document_terms - query_terms

        most_similar_query_term, most_similar_non_query_term = \
            self._most_similar_terms(query_terms, non_query_terms)

        def term_frequency_ratio(
                document_a: RankedDocument,
                document_b: RankedDocument
        ):
            tf_most_similar_query_term = context.term_frequency(
                document_b.content,
                most_similar_query_term
            )
            tf_most_similar_non_query_term = context.term_frequency(
                document_a.content,
                most_similar_non_query_term
            )
            if tf_most_similar_query_term <= 0:
                return nan
            return tf_most_similar_non_query_term / tf_most_similar_query_term

        if approximately_equal(
                len(document2_terms) / len(document1_terms),
                term_frequency_ratio(document2, document1),
                margin_fraction=0.2
        ):
            return 1
        elif approximately_equal(
                len(document1_terms) / len(document2_terms),
                term_frequency_ratio(document1, document2),
                margin_fraction=0.2
        ):
            return -1

        return 0


class STMC2(_STMC2Base, _WordNetTermSimilarity):
    pass


class STMC2_f(_STMC2Base, _FastTextWikiNewsTermSimilarity):
    pass