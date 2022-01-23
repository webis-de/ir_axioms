from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import List, Dict

from ir_axioms.axiom.base import Axiom
from ir_axioms.axiom.utils import (
    strictly_greater, synonym_set_similarity, approximately_same_length,
    vocabulary_overlap
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
        query_terms: List[str] = list(context.term_set(query))
        if len(query_terms) < 1:
            return 0

        similarity_sum: Dict[str, float] = defaultdict(lambda: 0)
        min_sim_term: str = query_terms[0]

        for query_term1, query_term2 in combinations(query_terms, 2):
            similarity = synonym_set_similarity(query_term1, query_term2)

            similarity_sum[query_term1] += similarity
            if similarity_sum[query_term1] < similarity_sum[min_sim_term]:
                min_sim_term = query_term1

            similarity_sum[query_term2] += similarity
            if similarity_sum[query_term2] < similarity_sum[min_sim_term]:
                min_sim_term = query_term2

        return strictly_greater(
            context.term_frequency(document1, min_sim_term),
            context.term_frequency(document2, min_sim_term),
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
        query_terms: List[str] = list(context.term_set(query))
        if len(query_terms) < 1:
            return 0

        similarity_sum: Dict[str, float] = defaultdict(lambda: 0)
        max_sim_term: str = query_terms[0]

        for query_term1, query_term2 in combinations(query_terms, 2):
            similarity = synonym_set_similarity(query_term1, query_term2)

            similarity_sum[query_term1] += similarity
            if similarity_sum[query_term1] > similarity_sum[max_sim_term]:
                max_sim_term = query_term1

            similarity_sum[query_term2] += similarity
            if similarity_sum[query_term2] > similarity_sum[max_sim_term]:
                max_sim_term = query_term2

        return strictly_greater(
            context.term_frequency(document1, max_sim_term),
            context.term_frequency(document2, max_sim_term),
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

        return super(LEN_AND, self).preference(
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

        return super(LEN_M_AND, self).preference(
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

        return super(LEN_DIV, self).preference(
            context,
            query,
            document1,
            document2
        )
