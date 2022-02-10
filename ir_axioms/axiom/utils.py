from bisect import bisect_left
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import product, combinations
from statistics import mean
from typing import List, Set, Iterator, Dict, Iterable

from nltk.corpus import wordnet

from ir_axioms.model import Query, RankedDocument, IndexContext
from ir_axioms.utils.nltk import download_nltk_dependencies


def strictly_greater(x, y):
    if x > y:
        return 1
    elif y > x:
        return -1
    return 0


def strictly_less(x, y):
    if y > x:
        return 1
    elif x > y:
        return -1
    return 0


def approximately_equal(*args, margin_fraction: float = 0.1):
    """
    True if all numeric args are
    within (100 * margin_fraction)% of the largest.
    """

    abs_max = max(args, key=lambda item: abs(item))
    if abs_max == 0:
        # All values are 0.
        return True

    boundaries = (
        abs_max * (1 + margin_fraction),
        abs_max * (1 - margin_fraction),
    )
    boundary_min = min(boundaries)
    boundary_max = max(boundaries)

    return all(boundary_min < item < boundary_max for item in args)


def all_query_terms_in_documents(
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


def same_query_term_subset(
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


def approximately_same_length(
        context: IndexContext,
        document1: RankedDocument,
        document2: RankedDocument,
        margin_fraction: float = 0.1
) -> bool:
    return approximately_equal(
        len(context.terms(document1)),
        len(context.terms(document2)),
        margin_fraction=margin_fraction
    )


@lru_cache(maxsize=4096)
def synonym_set(
        term: str,
        smoothing: int = 0
) -> List[str]:
    download_nltk_dependencies("wordnet", "omw-1.4")
    cutoff = smoothing + 1
    return wordnet.synsets(term)[:cutoff]


@lru_cache(maxsize=4096)
def synonym_set_similarity(
        term1: str,
        term2: str,
        smoothing: int = 0
) -> float:
    synonyms_term1 = synonym_set(term1, smoothing)
    synonyms_term2 = synonym_set(term2, smoothing)

    n = 0
    similarity_sum = 0

    for synonym1, synonym2 in product(synonyms_term1, synonyms_term2):
        similarity = wordnet.wup_similarity(synonym1, synonym2)
        if similarity is not None:
            similarity_sum += similarity
            n += 1

    if n == 0:
        return 0

    return similarity_sum / n


def synonym_set_similarity_sums(terms: Iterable[str]) -> Dict[str, float]:
    similarity_sums: Dict[str, float] = defaultdict(lambda: 0)
    for term1, term2 in combinations(terms, 2):
        similarity = synonym_set_similarity(term1, term2)
        similarity_sums[term1] += similarity
        similarity_sums[term2] += similarity
    return similarity_sums


def vocabulary_overlap(vocabulary1: Set[str], vocabulary2: Set[str]):
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


def average_between_query_terms(
        query_terms: Set[str],
        document_terms: List[str]
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


def take_closest(sorted_list: List[int], target: int):
    """
    Return closest value to n.
    If two numbers are equally close, return the smallest number.

    It is assumed that l is sorted.
    See: https://stackoverflow.com/questions/12141150
    """
    position = bisect_left(sorted_list, target)
    if position == 0:
        return sorted_list[0]
    if position == len(sorted_list):
        return sorted_list[-1]
    before = sorted_list[position - 1]
    after = sorted_list[position]
    if after - target < target - before:
        return after
    else:
        return before


def query_term_index_groups(
        query_terms: Set[str],
        document_terms: List[str]
) -> Iterator[List[int]]:
    indexes = defaultdict(list)
    for index, term in enumerate(document_terms):
        if term in query_terms:
            indexes[term].append(index)
    for term in query_terms:
        other_query_terms = query_terms - {term}
        for index in indexes[term]:
            group = [index] + [
                take_closest(indexes[other_term], index)
                for other_term in other_query_terms
                if len(indexes[other_term]) > 0
            ]
            yield group


def closest_grouping_size_and_count(
        query_terms: Set[str],
        document_terms: List[str]
):
    index_groups = query_term_index_groups(query_terms, document_terms)

    # Number of non-query terms within groups.
    non_query_term_occurrences = [
        len([
            term
            for term in document_terms[min(index_group) + 1:max(index_group)]
            if term not in query_terms
        ])
        for index_group in index_groups
    ]

    occurrences_counter = Counter(non_query_term_occurrences)
    min_occurrences = min(occurrences_counter.keys())
    min_occurrences_count = occurrences_counter[min_occurrences]
    return min_occurrences, min_occurrences_count


def average_smallest_span(
        query_terms: Set[str],
        document_terms: List[str]
):
    return mean(
        max(group) - min(group)
        for group in query_term_index_groups(query_terms, document_terms)
    )
