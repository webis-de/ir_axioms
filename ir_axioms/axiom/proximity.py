from bisect import bisect_left
# noinspection PyPep8Naming
from collections import Counter as counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from math import inf
from statistics import mean
from typing import Counter, FrozenSet, Sequence

from ir_axioms.axiom.base import Axiom
from ir_axioms.axiom.utils import (
    strictly_less, strictly_greater,

)
from ir_axioms.model import Query, RankedDocument, IndexContext


def _same_query_term_subset(
        context: IndexContext,
        query: Query,
        document1: RankedDocument,
        document2: RankedDocument
) -> bool:
    """
    Both documents contain the same set of query terms.
    """

    query_terms = context.term_set(query)
    document1_terms = context.term_set(document1)
    document2_terms = context.term_set(document2)

    if len(query_terms) <= 1:
        return False

    in_document1 = query_terms & document1_terms
    in_document2 = query_terms & document2_terms

    # Both contain the same subset of at least two terms.
    return (in_document1 == in_document2) and len(in_document1) > 1


def _average_between_query_terms(
        query_terms: FrozenSet[str],
        document_terms: Sequence[str]
) -> float:
    query_term_pairs = set(combinations(query_terms, 2))
    if len(query_term_pairs) == 0:
        # Single-term query.
        return 0

    number_words = 0
    for item in query_term_pairs:
        element1_position = document_terms.index(item[0])
        element2_position = document_terms.index(item[1])
        number_words += abs(element1_position - element2_position - 1)
    return number_words / len(query_term_pairs)


def _all_query_terms_in_documents(
        context: IndexContext,
        query: Query,
        document1: RankedDocument,
        document2: RankedDocument
):
    query_terms = context.term_set(query)
    document1_terms = context.term_set(document1)
    document2_terms = context.term_set(document2)

    if len(query_terms) <= 1:
        return False

    return (
            len(query_terms & document1_terms) == len(query_terms) and
            len(query_terms & document2_terms) == len(query_terms)
    )


def _take_closest(sorted_items: Sequence[int], target: int):
    """
    Return closest value to n.
    If two numbers are equally close, return the smallest number.

    It is assumed that l is sorted.
    See: https://stackoverflow.com/questions/12141150
    """
    position = bisect_left(sorted_items, target)
    if position == 0:
        return sorted_items[0]
    if position == len(sorted_items):
        return sorted_items[-1]
    before = sorted_items[position - 1]
    after = sorted_items[position]
    if after - target < target - before:
        return after
    else:
        return before


def _query_term_index_groups(
        query_terms: FrozenSet[str],
        document_terms: Sequence[str]
) -> Sequence[Sequence[int]]:
    index_groups = []
    indexes = defaultdict(lambda: [])
    for index, term in enumerate(document_terms):
        if term in query_terms:
            indexes[term].append(index)
    for term in query_terms:
        other_query_terms = query_terms - {term}
        for index in indexes[term]:
            group = (index,) + tuple(
                _take_closest(tuple(indexes[other_term]), index)
                for other_term in other_query_terms
                if len(indexes[other_term]) > 0
            )
            index_groups.append(group)
    return index_groups


def _average_smallest_span(
        query_terms: FrozenSet[str],
        document_terms: Sequence[str]
) -> float:
    index_groups = _query_term_index_groups(query_terms, document_terms)
    if len(index_groups) == 0:
        return inf
    return mean(
        max(group) - min(group)
        for group in index_groups
    )


def _closest_grouping_size_and_count(
        query_terms: FrozenSet[str],
        document_terms: Sequence[str]
):
    index_groups = _query_term_index_groups(query_terms, document_terms)

    # Number of non-query terms within groups.
    non_query_term_occurrences = [
        len([
            term
            for term in document_terms[min(index_group) + 1:max(index_group)]
            if term not in query_terms
        ])
        for index_group in index_groups
    ]

    occurrences_counter: Counter = counter(non_query_term_occurrences)
    if len(occurrences_counter.keys()) == 0:
        return 0, 0
    min_occurrences = min(occurrences_counter.keys())
    min_occurrences_count = occurrences_counter[min_occurrences]
    return min_occurrences, min_occurrences_count


@dataclass(frozen=True)
class PROX1(Axiom):
    name = "PROX1"

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        if not _same_query_term_subset(context, query, document1, document2):
            return 0

        query_terms = context.term_set(query)
        document1_terms = context.terms(document1)
        document2_terms = context.terms(document2)

        overlapping_terms = (
                query_terms &
                set(document1_terms) &
                set(document2_terms)
        )

        average1 = _average_between_query_terms(
            overlapping_terms,
            document1_terms
        )
        average2 = _average_between_query_terms(
            overlapping_terms,
            document2_terms
        )

        return strictly_greater(average2, average1)


@dataclass(frozen=True)
class PROX2(Axiom):
    name = "PROX2"

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        if not _same_query_term_subset(context, query, document1, document2):
            return 0

        query_terms = context.term_set(query)
        document1_terms = context.terms(document1)
        document2_terms = context.terms(document2)
        terms = set(document1_terms) & set(document2_terms)

        first_position_sum1 = 0
        first_position_sum2 = 0

        for term in query_terms:
            if term in terms:
                first_position_sum1 += document1_terms.index(term)
                first_position_sum2 += document2_terms.index(term)

        return strictly_greater(first_position_sum2, first_position_sum1)


def _find_index(query_terms: Sequence[str], document_terms: Sequence[str]):
    query_terms_length = len(query_terms)
    terms_length = len(document_terms)
    for index, term in enumerate(document_terms):
        if (
                term == query_terms[0] and
                index + query_terms_length <= terms_length and
                document_terms[index:(index + query_terms_length)] ==
                query_terms
        ):
            return index
    return inf


@dataclass(frozen=True)
class PROX3(Axiom):
    name = "PROX3"

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        if not _same_query_term_subset(context, query, document1, document2):
            return 0
        query_terms = context.terms(query)
        document1_terms = context.terms(document1)
        document2_terms = context.terms(document2)
        return strictly_less(
            _find_index(query_terms, document1_terms),
            _find_index(query_terms, document2_terms)
        )


@dataclass(frozen=True)
class PROX4(Axiom):
    name = "PROX4"

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        if not _all_query_terms_in_documents(
                context,
                query,
                document1,
                document2
        ):
            return 0

        query_terms = context.term_set(query)
        document1_terms = context.terms(document1)
        document2_terms = context.terms(document2)

        occurrences1, count1 = _closest_grouping_size_and_count(
            query_terms,
            document1_terms
        )
        occurrences2, count2 = _closest_grouping_size_and_count(
            query_terms,
            document2_terms
        )

        if occurrences1 != occurrences2:
            return strictly_less(occurrences1, occurrences2)
        else:
            return strictly_greater(count1, count2)


@dataclass(frozen=True)
class PROX5(Axiom):
    name = "PROX5"

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        if not _all_query_terms_in_documents(
                context,
                query,
                document1,
                document2
        ):
            return 0

        query_terms = context.term_set(query)
        document1_terms = context.terms(document1)
        document2_terms = context.terms(document2)

        smallest_span1 = _average_smallest_span(query_terms, document1_terms)
        smallest_span2 = _average_smallest_span(query_terms, document2_terms)

        return strictly_less(smallest_span1, smallest_span2)

# TODO: QPHRA axiom:
#  For queries with highlighted phrases (e.g., via double quotes),
#  prefer documents containing all the query phrases over
#  documents not containing all phrases. [hagen:2016d]
