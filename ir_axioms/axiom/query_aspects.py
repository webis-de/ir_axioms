from dataclasses import dataclass
from typing import Set

from ir_axioms.axiom.base import Axiom
from ir_axioms.axiom.utils import (
    strictly_greater, approximately_same_length,
    vocabulary_overlap, synonym_set_similarity_sums
)
from ir_axioms.model import Query, RankedDocument
from ir_axioms.model.context import RerankingContext


class REG(Axiom):
    """
    Reference:
    Zheng, W., Fang, H.: Query aspect based term weighting regularization
    in information retrieval. In: Gurrin, C., et al. (eds.) ECIR 2010.
    """
    name = "REG"

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        query_terms = context.term_set(query)
        similarity_sum = synonym_set_similarity_sums(query_terms)

        print(similarity_sum)
        minimum_similarity = min(
            similarity
            for _, similarity in similarity_sum.items()
        )
        minimum_similarity_terms: Set[str] = {
            term
            for term, similarity in similarity_sum.items()
            if similarity == minimum_similarity
        }
        assert len(minimum_similarity_terms) == 1
        minimum_similarity_term = next(iter(minimum_similarity_terms))

        return strictly_greater(
            context.term_frequency(document1, minimum_similarity_term),
            context.term_frequency(document2, minimum_similarity_term),
        )


class ANTI_REG(Axiom):
    """
    Reference:
    Zheng, W., Fang, H.: Query aspect based term weighting regularization
    in information retrieval. In: Gurrin, C., et al. (eds.) ECIR 2010.

    Modified to use maximum similarity instead of minimum similarity.
    """
    name = "ANTI_REG"

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        query_terms = context.term_set(query)
        similarity_sum = synonym_set_similarity_sums(query_terms)

        maximum_similarity = max(
            similarity
            for _, similarity in similarity_sum.items()
        )
        maximum_similarity_terms: Set[str] = {
            term
            for term, similarity in similarity_sum.items()
            if similarity == maximum_similarity
        }
        assert len(maximum_similarity_terms) == 1

        maximum_similarity_term = next(iter(maximum_similarity_terms))

        return strictly_greater(
            context.term_frequency(document1, maximum_similarity_term),
            context.term_frequency(document2, maximum_similarity_term),
        )


class AND(Axiom):
    name = "AND"

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        query_terms = context.term_set(query)
        document1_terms = context.term_set(document1)
        document2_terms = context.term_set(document2)
        s1 = query_terms.issubset(document1_terms)
        s2 = query_terms.issubset(document2_terms)
        return strictly_greater(s1, s2)


@dataclass(frozen=True)
class LEN_AND(AND):
    """
    Modified AND:
    The precondition for the documents' lengths can be varied.

    Default margin fraction: 0.1
    """
    name = "LEN_AND"

    margin_fraction: float = 0.1

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        if not approximately_same_length(
                context,
                document1,
                document2,
                self.margin_fraction
        ):
            return 0

        return super().preference(
            context,
            query,
            document1,
            document2
        )


class M_AND(Axiom):
    """
    Modified AND:
    One document contains a larger subset of query terms.
    """
    name = "M_AND"

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        query_terms = context.term_set(query)
        document1_terms = context.term_set(document1)
        document2_terms = context.term_set(document2)
        query_term_count1 = query_terms & document1_terms
        query_term_count2 = query_terms & document2_terms
        return strictly_greater(len(query_term_count1), len(query_term_count2))


@dataclass(frozen=True)
class LEN_M_AND(M_AND):
    """
    Modified M_AND:
    The precondition for the documents' lengths can be varied.

    Default margin fraction: 0.1
    """
    name = "LEN_M_AND"

    margin_fraction: float = 0.1

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        if not approximately_same_length(
                context,
                document1,
                document2,
                self.margin_fraction
        ):
            return 0

        return super().preference(
            context,
            query,
            document1,
            document2
        )


class DIV(Axiom):
    name = "DIV"

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        query_terms = context.term_set(query)
        overlap1 = vocabulary_overlap(
            query_terms,
            context.term_set(document1)
        )
        overlap2 = vocabulary_overlap(
            query_terms,
            context.term_set(document2)
        )

        return strictly_greater(overlap2, overlap1)


@dataclass(frozen=True)
class LEN_DIV(DIV):
    """
    Modified DIV:
    The precondition for the documents' lengths can be varied.

    Default margin fraction: 0.1
    """
    name = "LEN_DIV"

    margin_fraction: float = 0.1

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        if not approximately_same_length(
                context,
                document1,
                document2,
                self.margin_fraction
        ):
            return 0

        return super().preference(
            context,
            query,
            document1,
            document2
        )
