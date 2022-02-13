from abc import ABC
from dataclasses import dataclass
from itertools import combinations
from statistics import mean
from typing import Set, FrozenSet

from ir_axioms import logger
from ir_axioms.axiom.base import Axiom
from ir_axioms.axiom.preconditions import LEN_Mixin
from ir_axioms.axiom.utils import strictly_greater, approximately_equal
from ir_axioms.model import Query, RankedDocument, IndexContext
from ir_axioms.utils.similarity import (
    TermSimilarityMixin, WordNetSynonymSetTermSimilarityMixin,
    FastTextWikiNewsTermSimilarityMixin
)


def _vocabulary_overlap(vocabulary1: FrozenSet[str],
                        vocabulary2: FrozenSet[str]):
    """
    Vocabulary overlap as calculated by the Jaccard coefficient.
    """
    intersection_length = len(vocabulary1 & vocabulary2)
    if intersection_length == 0:
        return 0
    return (
            intersection_length /
            (len(vocabulary1) + len(vocabulary2) - intersection_length)
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
class _ASPECT_REG(Axiom, TermSimilarityMixin, ABC):
    """
    Similar to REG but follows the query aspect clustering
    from the paper and then counts the number of aspects covered
    in each document.

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
        document1_terms = context.term_set(document1)
        document2_terms = context.term_set(document2)
        if len(query_terms) == 0:
            return 0

        term_discriminators = {
            context.inverse_document_frequency(term)
            for term in query_terms
        }
        if not approximately_equal(*term_discriminators):
            # Require same term discriminator for all query terms.
            return 0

        average_similarity = mean(
            self.similarity(term1, term2)
            for term1, term2 in combinations(query_terms, 2)
        )

        query_aspects: Set[Set[str]] = {{term} for term in query_terms}
        while len(query_aspects) > 1:
            for a1, a2 in combinations(query_aspects, 2):
                a1: Set[str]
                a2: Set[str]
                if any(
                        self.similarity(term1, term2) > average_similarity
                        for term1 in a1
                        for term2 in a2
                ):
                    # Remove mergable aspects.
                    query_aspects = {
                        aspect
                        for i, aspect in enumerate(query_aspects)
                        if aspect != a1 and aspect != a2
                    }
                    # Merge aspects.
                    merged_aspect = a1 | a2
                    query_aspects.add(merged_aspect)
                    break  # Go to begin of while-loop.

            # We iterated over all aspect pairs
            # and did not find any mergable aspects.
            break

        count_document1_aspects = {
            1 for aspect in query_aspects
            if not document1_terms.isdisjoint(aspect)
        }
        count_document2_aspects = {
            1 for aspect in query_aspects
            if not document2_terms.isdisjoint(aspect)
        }
        return strictly_greater(
            count_document1_aspects,
            count_document2_aspects
        )


@dataclass(frozen=True)
class ASPECT_REG(_ASPECT_REG, WordNetSynonymSetTermSimilarityMixin):
    name = "ASPECT-REG"


@dataclass(frozen=True)
class ASPECT_REG_fastText(_ASPECT_REG, FastTextWikiNewsTermSimilarityMixin):
    name = "ASPECT-REG-fastText"


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
class LEN_AND(LEN_Mixin, AND):
    name = "LEN-AND"


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
class LEN_M_AND(LEN_Mixin, M_AND):
    name = "LEN-M-AND"


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
        overlap1 = _vocabulary_overlap(
            query_terms,
            context.term_set(document1)
        )
        overlap2 = _vocabulary_overlap(
            query_terms,
            context.term_set(document2)
        )

        return strictly_greater(overlap2, overlap1)


@dataclass(frozen=True)
class LEN_DIV(LEN_Mixin, DIV):
    name = "LEN-DIV"


# Shorthand names:
REG_f = REG_fastText
ANTI_REG_f = ANTI_REG_fastText
ASPECT_REG_f = ASPECT_REG_fastText
