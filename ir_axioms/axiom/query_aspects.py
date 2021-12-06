from dataclasses import dataclass
from itertools import combinations, repeat

from ir_axioms.axiom import Axiom
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

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        query_terms: list[str] = list(context.term_set(query.title))
        similarity_sum: list[float] = list(repeat(0, len(query_terms)))
        min_similarity_index: int = 0

        for i1, i2 in combinations(range(len(query_terms)), 2):
            sim = synonym_set_similarity(query_terms[i1], query_terms[i2])

            similarity_sum[i1] += sim
            if similarity_sum[i1] < similarity_sum[min_similarity_index]:
                min_similarity_index = i1

            similarity_sum[i2] += sim
            if similarity_sum[i2] < similarity_sum[min_similarity_index]:
                min_similarity_index = i2

        term_min = query_terms[min_similarity_index]
        return strictly_greater(
            context.term_frequency(document1.content, term_min),
            context.term_frequency(document2.content, term_min),
        )


class ANTI_REG(Axiom):
    """
    Reference:
    Zheng, W., Fang, H.: Query aspect based term weighting regularization
    in information retrieval. In: Gurrin, C., et al. (eds.) ECIR 2010.

    Modified to use maximum similarity instead of minimum similarity.
    """

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        query_terms: list[str] = list(context.term_set(query.title))
        similarity_sum: list[float] = list(repeat(0, len(query_terms)))
        max_similarity_index: int = 0

        for i1, i2 in combinations(range(len(query_terms)), 2):
            sim = synonym_set_similarity(query_terms[i1], query_terms[i2])

            similarity_sum[i1] += sim
            if similarity_sum[i1] > similarity_sum[max_similarity_index]:
                max_similarity_index = i1

            similarity_sum[i2] += sim
            if similarity_sum[i2] > similarity_sum[max_similarity_index]:
                max_similarity_index = i2

        term_max = query_terms[max_similarity_index]
        return strictly_greater(
            context.term_frequency(document1.content, term_max),
            context.term_frequency(document2.content, term_max),
        )


class AND(Axiom):

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        query_terms = context.term_set(query.title)
        document1_terms = context.term_set(document1.content)
        document2_terms = context.term_set(document2.content)
        s1 = query_terms & document1_terms == query_terms
        s2 = query_terms & document2_terms == query_terms
        return strictly_greater(s1, s2)


@dataclass(frozen=True)
class LEN_AND(AND):
    """
    Modified AND:
    The precondition for the documents' lengths can be varied.

    Default margin fraction: 0.1
    """
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

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        query_terms = context.term_set(query.title)
        document1_terms = context.term_set(document1.content)
        document2_terms = context.term_set(document2.content)
        s1 = query_terms & document1_terms
        s2 = query_terms & document2_terms
        return strictly_greater(len(s1), len(s2))


class LEN_M_AND(M_AND):
    """
    Modified M_AND:
    The precondition for the documents' lengths can be varied.

    Default margin fraction: 0.1
    """
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
    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        query_terms = context.term_set(query.title)
        overlap1 = vocabulary_overlap(
            query_terms,
            context.term_set(document1.content)
        )
        overlap2 = vocabulary_overlap(
            query_terms,
            context.term_set(document2.content)
        )

        return strictly_greater(overlap2, overlap1)


class LEN_DIV(DIV):
    """
    Modified DIV:
    The precondition for the documents' lengths can be varied.

    Default margin fraction: 0.1
    """
    margin_fraction = 0.1

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
