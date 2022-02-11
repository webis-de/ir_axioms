from abc import ABC
from dataclasses import dataclass
from typing import Set

from ir_axioms import logger
from ir_axioms.axiom.base import Axiom
from ir_axioms.axiom.utils import (
    strictly_greater, approximately_same_length, vocabulary_overlap
)
from ir_axioms.model import Query, RankedDocument, IndexContext
from ir_axioms.utils.similarity import (
    TermSimilarityMixin, WordNetSynonymSetTermSimilarityMixin,
    FastTextWikiNewsTermSimilarityMixin
)


@dataclass(frozen=True)
class _REG(Axiom, TermSimilarityMixin, ABC):
    """
    Reference:
    Zheng, W., Fang, H.: Query aspect based term weighting regularization
    in information retrieval. In: Gurrin, C., et al. (eds.) ECIR 2010.
    """

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        query_terms = context.term_set(query)
        if len(query_terms) == 0:
            return 0

        similarity_sum = self.similarity_sums(query_terms)

        minimum_similarity = min(
            similarity
            for _, similarity in similarity_sum.items()
        )
        minimum_similarity_terms: Set[str] = {
            term
            for term, similarity in similarity_sum.items()
            if similarity == minimum_similarity
        }

        if len(minimum_similarity_terms) != 1:
            logger.debug(
                f"The following terms were equally similar "
                f"during evaluating the {self.name} axiom: "
                f"{', '.join(minimum_similarity_terms)}"
            )
            return 0

        minimum_similarity_term = next(iter(minimum_similarity_terms))

        return strictly_greater(
            context.term_frequency(document1, minimum_similarity_term),
            context.term_frequency(document2, minimum_similarity_term),
        )


@dataclass(frozen=True)
class REG(_REG, WordNetSynonymSetTermSimilarityMixin):
    name = "REG"


@dataclass(frozen=True)
class REG_fastText(_REG, FastTextWikiNewsTermSimilarityMixin):
    name = "REG-fastText"


@dataclass(frozen=True)
class _ANTI_REG(Axiom, TermSimilarityMixin, ABC):
    """
    Reference:
    Zheng, W., Fang, H.: Query aspect based term weighting regularization
    in information retrieval. In: Gurrin, C., et al. (eds.) ECIR 2010.

    Modified to use maximum similarity instead of minimum similarity.
    """

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        query_terms = context.term_set(query)
        if len(query_terms) == 0:
            return 0

        similarity_sum = self.similarity_sums(query_terms)

        maximum_similarity = max(
            similarity
            for _, similarity in similarity_sum.items()
        )
        maximum_similarity_terms: Set[str] = {
            term
            for term, similarity in similarity_sum.items()
            if similarity == maximum_similarity
        }

        if len(maximum_similarity_terms) != 1:
            logger.debug(
                f"The following terms were equally similar "
                f"during evaluating the {self.name} axiom: "
                f"{', '.join(maximum_similarity_terms)}"
            )
            return 0

        maximum_similarity_term = next(iter(maximum_similarity_terms))

        return strictly_greater(
            context.term_frequency(document1, maximum_similarity_term),
            context.term_frequency(document2, maximum_similarity_term),
        )


@dataclass(frozen=True)
class ANTI_REG(_ANTI_REG, WordNetSynonymSetTermSimilarityMixin):
    name = "ANTI-REG"


@dataclass(frozen=True)
class ANTI_REG_fastText(_ANTI_REG, FastTextWikiNewsTermSimilarityMixin):
    name = "ANTI-REG-fastText"


@dataclass(frozen=True)
class AND(Axiom):
    name = "AND"

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
    name = "LEN-AND"

    margin_fraction: float = 0.1

    def preference(
            self,
            context: IndexContext,
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


@dataclass(frozen=True)
class M_AND(Axiom):
    """
    Modified AND:
    One document contains a larger subset of query terms.
    """
    name = "M-AND"

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
    name = "LEN-M-AND"

    margin_fraction: float = 0.1

    def preference(
            self,
            context: IndexContext,
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


@dataclass(frozen=True)
class DIV(Axiom):
    name = "DIV"

    def preference(
            self,
            context: IndexContext,
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
    name = "LEN-DIV"

    margin_fraction: float = 0.1

    def preference(
            self,
            context: IndexContext,
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


# Shorthand names:
REG_f = REG_fastText
ANTI_REG_f = ANTI_REG_fastText
