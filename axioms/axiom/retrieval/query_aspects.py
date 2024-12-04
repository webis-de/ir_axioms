from dataclasses import dataclass
from typing import Final, Set, FrozenSet, List

from axioms.axiom.base import Axiom
from axioms.model.retrieval import get_index_context
from axioms.precondition.length import LEN
from axioms.axiom.utils import strictly_greater, approximately_equal
from axioms.model import Query, Document, IndexContext
from axioms.tools import (
    TermSimilarity,
    WordNetSynonymSetTermSimilarity,
)


def _vocabulary_overlap(
    vocabulary1: FrozenSet[str],
    vocabulary2: FrozenSet[str],
):
    """
    Vocabulary overlap as calculated by the Jaccard coefficient.
    """
    intersection_length = len(vocabulary1 & vocabulary2)
    if intersection_length == 0:
        return 0
    return intersection_length / (
        len(vocabulary1) + len(vocabulary2) - intersection_length
    )


@dataclass(frozen=True, kw_only=True)
class RegAxiom(Axiom[Query, Document]):
    """
    Reference:
    Zheng, W., Fang, H.: Query aspect based term weighting regularization
    in information retrieval. In: Gurrin, C., et al. (eds.) ECIR 2010.
    """

    context: IndexContext
    term_similarity: TermSimilarity

    def preference(
        self,
        input: Query,
        output1: Document,
        output2: Document,
    ):
        query_terms = self.context.term_set(input)

        minimum_similarity_term = self.term_similarity.least_similar_term(query_terms)
        if minimum_similarity_term is None:
            return 0

        return strictly_greater(
            self.context.term_frequency(output1, minimum_similarity_term),
            self.context.term_frequency(output2, minimum_similarity_term),
        )


REG: Final = RegAxiom(
    context=get_index_context(),
    term_similarity=WordNetSynonymSetTermSimilarity(),
).with_precondition(LEN)


@dataclass(frozen=True, kw_only=True)
class AntiRegAxiom(Axiom[Query, Document]):
    """
    Reference:
    Zheng, W., Fang, H.: Query aspect based term weighting regularization
    in information retrieval. In: Gurrin, C., et al. (eds.) ECIR 2010.

    Modified to use maximum similarity instead of minimum similarity.
    """

    context: IndexContext
    term_similarity: TermSimilarity

    def preference(
        self,
        input: Query,
        output1: Document,
        output2: Document,
    ):
        query_terms = self.context.term_set(input)

        maximum_similarity_term = self.term_similarity.most_similar_term(query_terms)
        if maximum_similarity_term is None:
            return 0

        return strictly_greater(
            self.context.term_frequency(output1, maximum_similarity_term),
            self.context.term_frequency(output2, maximum_similarity_term),
        )


ANTI_REG: Final = AntiRegAxiom(
    context=get_index_context(),
    term_similarity=WordNetSynonymSetTermSimilarity(),
)


@dataclass(frozen=True, kw_only=True)
class AspectRegAxiom(Axiom[Query, Document]):
    """
    Similar to REG but follows the query aspect clustering
    from the paper and then counts the number of aspects covered
    in each document.

    Reference:
    Zheng, W., Fang, H.: Query aspect based term weighting regularization
    in information retrieval. In: Gurrin, C., et al. (eds.) ECIR 2010.
    """

    context: IndexContext
    term_similarity: TermSimilarity
    term_discriminator_margin_fraction: float = 0.1

    def preference(
        self,
        input: Query,
        output1: Document,
        output2: Document,
    ):
        query_terms = self.context.term_set(input)
        document1_terms = self.context.term_set(output1)
        document2_terms = self.context.term_set(output2)
        if len(query_terms) == 0:
            return 0

        term_discriminators = {
            self.context.inverse_document_frequency(term) for term in query_terms
        }
        if not approximately_equal(
            *term_discriminators, self.term_discriminator_margin_fraction
        ):
            # Require same term discriminator for all query terms.
            return 0

        average_similarity = self.term_similarity.average_similarity(
            query_terms, query_terms
        )

        query_aspects: List[Set[str]] = [{term} for term in query_terms]

        # Iterate aspect 1 from start.
        for i1 in range(0, len(query_aspects) - 1, +1):
            a1 = query_aspects[i1]
            # Iterate aspect 2 from end.
            for i2 in range(len(query_aspects) - 1, i1 + 1, -1):
                a2 = query_aspects[i2]

                # Is any term pair similar enough to merge the aspects?
                if any(
                    self.term_similarity.similarity(term1, term2) > average_similarity
                    for term1 in a1
                    for term2 in a2
                ):
                    # Merge aspect 2 into aspect 1.
                    a1.update(a2)
                    # Remove merged aspect 2.
                    query_aspects.pop(i2)

        count_document1_aspects = {
            1 for aspect in query_aspects if not document1_terms.isdisjoint(aspect)
        }
        count_document2_aspects = {
            1 for aspect in query_aspects if not document2_terms.isdisjoint(aspect)
        }
        return strictly_greater(
            len(count_document1_aspects) > 0,
            len(count_document2_aspects) > 0,
        )


ASPECT_REG: Final = AspectRegAxiom(
    context=get_index_context(),
    term_similarity=WordNetSynonymSetTermSimilarity(),
)


@dataclass(frozen=True, kw_only=True)
class AndAxiom(Axiom[Query, Document]):
    context: IndexContext

    def preference(
        self,
        input: Query,
        output1: Document,
        output2: Document,
    ):
        query_terms = self.context.term_set(input)
        document1_terms = self.context.term_set(output1)
        document2_terms = self.context.term_set(output2)
        query_term_count1 = query_terms & document1_terms
        query_term_count2 = query_terms & document2_terms
        all_query_terms1 = len(query_term_count1) == len(query_terms)
        all_query_terms2 = len(query_term_count2) == len(query_terms)
        return strictly_greater(all_query_terms1, all_query_terms2)


AND: Final = AndAxiom(
    context=get_index_context(),
)
LEN_AND: Final = AND.with_precondition(LEN)


@dataclass(frozen=True, kw_only=True)
class ModifiedAndAxiom(Axiom[Query, Document]):
    """
    Modified AND:
    One document contains a larger subset of query terms.
    """

    context: IndexContext

    def preference(
        self,
        input: Query,
        output1: Document,
        output2: Document,
    ):
        query_terms = self.context.term_set(input)
        document1_terms = self.context.term_set(output1)
        document2_terms = self.context.term_set(output2)
        query_term_count1 = query_terms & document1_terms
        query_term_count2 = query_terms & document2_terms

        return strictly_greater(len(query_term_count1), len(query_term_count2))


M_AND: Final = ModifiedAndAxiom(
    context=get_index_context(),
)
LEN_M_AND: Final = M_AND.with_precondition(LEN)


@dataclass(frozen=True, kw_only=True)
class DivAxiom(Axiom[Query, Document]):
    context: IndexContext

    def preference(
        self,
        input: Query,
        output1: Document,
        output2: Document,
    ):
        query_terms = self.context.term_set(input)
        overlap1 = _vocabulary_overlap(query_terms, self.context.term_set(output1))
        overlap2 = _vocabulary_overlap(query_terms, self.context.term_set(output2))

        return strictly_greater(overlap2, overlap1)


DIV: Final = DivAxiom(
    context=get_index_context(),
)
LEN_DIV: Final = DIV.with_precondition(LEN)
