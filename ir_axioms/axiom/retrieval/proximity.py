from bisect import bisect_left
from collections import Counter as counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from math import inf
from statistics import mean
from typing import Counter, Final, AbstractSet, Sequence, Union

from injector import inject
from numpy import array, float_
from tqdm.auto import tqdm

from ir_axioms.axiom.base import Axiom
from ir_axioms.axiom.utils import strictly_less, strictly_greater
from ir_axioms.dependency_injection import injector
from ir_axioms.model import Query, Document, Preference, PreferenceMatrix
from ir_axioms.tools import TextContents, TermTokenizer
from ir_axioms.utils.lazy import lazy_inject


def _same_query_term_subset(
    query_terms: AbstractSet[str],
    document1_terms: AbstractSet[str],
    document2_terms: AbstractSet[str],
) -> bool:
    """
    Both documents contain the same set of query terms.
    """

    if len(query_terms) <= 1:
        return False

    in_document1 = query_terms & document1_terms
    in_document2 = query_terms & document2_terms

    # Both contain the same subset of at least two terms.
    return (in_document1 == in_document2) and len(in_document1) > 1


def _average_between_query_terms(
    query_terms: AbstractSet[str], document_terms: Sequence[str]
) -> float:
    query_term_pairs = set(combinations(query_terms, 2))
    if len(query_term_pairs) == 0:
        # Single-term query.
        return 0

    number_words = 0
    for term1, term2 in query_term_pairs:
        element1_position = document_terms.index(term1)
        element2_position = document_terms.index(term2)
        number_words += abs(element1_position - element2_position - 1)
    return number_words / len(query_term_pairs)


def _all_query_terms_in_documents(
    query_terms: AbstractSet[str],
    document1_terms: AbstractSet[str],
    document2_terms: AbstractSet[str],
) -> bool:
    if len(query_terms) <= 1:
        return False

    return len(query_terms & document1_terms) == len(query_terms) and len(
        query_terms & document2_terms
    ) == len(query_terms)


def _take_closest(
    sorted_items: Sequence[int],
    target: int,
) -> int:
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
    query_terms: AbstractSet[str],
    document_terms: Sequence[str],
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
    query_terms: AbstractSet[str],
    document_terms: Sequence[str],
) -> float:
    index_groups = _query_term_index_groups(query_terms, document_terms)
    if len(index_groups) == 0:
        return inf
    return mean(max(group) - min(group) for group in index_groups)


def _closest_grouping_size_and_count(
    query_terms: AbstractSet[str],
    document_terms: Sequence[str],
) -> tuple[int, int]:
    index_groups = _query_term_index_groups(query_terms, document_terms)

    # Number of non-query terms within groups.
    non_query_term_occurrences = [
        len(
            [
                term
                for term in document_terms[min(index_group) + 1 : max(index_group)]
                if term not in query_terms
            ]
        )
        for index_group in index_groups
    ]

    occurrences_counter: Counter = counter(non_query_term_occurrences)
    if len(occurrences_counter.keys()) == 0:
        return 0, 0
    min_occurrences = min(occurrences_counter.keys())
    min_occurrences_count = occurrences_counter[min_occurrences]
    return min_occurrences, min_occurrences_count


@inject
@dataclass(frozen=True, kw_only=True)
class Prox1Axiom(Axiom[Query, Document]):
    text_contents: TextContents[Union[Query, Document]]
    term_tokenizer: TermTokenizer

    def preference(
        self,
        input: Query,
        output1: Document,
        output2: Document,
    ) -> Preference:
        query_unique_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(input),
        )
        document1_terms = self.term_tokenizer.terms(
            self.text_contents.contents(output1),
        )
        document1_unique_terms = set(document1_terms)
        document2_terms = self.term_tokenizer.terms(
            self.text_contents.contents(output2),
        )
        document2_unique_terms = set(document2_terms)

        if not _same_query_term_subset(
            query_terms=query_unique_terms,
            document1_terms=document1_unique_terms,
            document2_terms=document2_unique_terms,
        ):
            return 0

        overlapping_terms = (
            query_unique_terms & document1_unique_terms & document2_unique_terms
        )

        average1 = _average_between_query_terms(overlapping_terms, document1_terms)
        average2 = _average_between_query_terms(overlapping_terms, document2_terms)

        return strictly_greater(average2, average1)

    # ADDITION: Come up with a better way to batch-compute preference-matrices.
    # The largest hurdle seems to be the overlapping terms computation.


PROX1: Final = lazy_inject(Prox1Axiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class Prox2Axiom(Axiom[Query, Document]):
    text_contents: TextContents[Union[Query, Document]]
    term_tokenizer: TermTokenizer

    def preference(
        self,
        input: Query,
        output1: Document,
        output2: Document,
    ) -> Preference:
        query_unique_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(input),
        )
        document1_terms = self.term_tokenizer.terms(
            self.text_contents.contents(output1),
        )
        document1_unique_terms = set(document1_terms)
        document2_terms = self.term_tokenizer.terms(
            self.text_contents.contents(output2),
        )
        document2_unique_terms = set(document2_terms)

        if not _same_query_term_subset(
            query_terms=query_unique_terms,
            document1_terms=document1_unique_terms,
            document2_terms=document2_unique_terms,
        ):
            return 0

        common_terms = (
            query_unique_terms & document1_unique_terms & document2_unique_terms
        )

        first_position_sum1 = sum(document1_terms.index(term) for term in common_terms)
        first_position_sum2 = sum(document2_terms.index(term) for term in common_terms)
        return strictly_greater(first_position_sum2, first_position_sum1)

    def preferences(
        self,
        input: Query,
        outputs: Sequence[Document],
    ) -> PreferenceMatrix:
        query_unique_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(input),
        )
        document_terms = [
            self.term_tokenizer.terms(self.text_contents.contents(output))
            for output in tqdm(
                outputs,
                desc="Tokenize documents",
                unit="document",
            )
        ]
        return array(
            [
                (
                    strictly_greater(
                        sum(
                            document_terms1.index(term)
                            for term in query_unique_terms
                            & set(document_terms1)
                            & set(document_terms2)
                        ),
                        sum(
                            document_terms2.index(term)
                            for term in query_unique_terms
                            & set(document_terms1)
                            & set(document_terms2)
                        ),
                    )
                    if _same_query_term_subset(
                        query_terms=query_unique_terms,
                        document1_terms=set(document_terms1),
                        document2_terms=set(document_terms2),
                    )
                    else 0
                )
                for document_terms1 in document_terms
                for document_terms2 in document_terms
            ],
            dtype=float_,
        ).reshape(len(outputs), len(outputs))


PROX2: Final = lazy_inject(Prox2Axiom, injector)


def _find_index(
    query_terms: Sequence[str],
    document_terms: Sequence[str],
):
    query_terms_length = len(query_terms)
    terms_length = len(document_terms)
    for index, term in enumerate(document_terms):
        if (
            term == query_terms[0]
            and index + query_terms_length <= terms_length
            and document_terms[index : (index + query_terms_length)] == query_terms
        ):
            return index
    return inf


@inject
@dataclass(frozen=True, kw_only=True)
class Prox3Axiom(Axiom[Query, Document]):
    text_contents: TextContents[Union[Query, Document]]
    term_tokenizer: TermTokenizer

    def preference(
        self,
        input: Query,
        output1: Document,
        output2: Document,
    ) -> Preference:
        query_terms = self.term_tokenizer.terms(
            self.text_contents.contents(input),
        )
        query_unique_terms = set(query_terms)
        document1_terms = self.term_tokenizer.terms(
            self.text_contents.contents(output1),
        )
        document1_unique_terms = set(document1_terms)
        document2_terms = self.term_tokenizer.terms(
            self.text_contents.contents(output2),
        )
        document2_unique_terms = set(document2_terms)

        if not _same_query_term_subset(
            query_terms=query_unique_terms,
            document1_terms=document1_unique_terms,
            document2_terms=document2_unique_terms,
        ):
            return 0

        return strictly_less(
            _find_index(
                query_terms=query_terms,
                document_terms=document1_terms,
            ),
            _find_index(
                query_terms=query_terms,
                document_terms=document2_terms,
            ),
        )

    # ADDITION: Come up with a better way to batch-compute preference-matrices.


PROX3: Final = lazy_inject(Prox3Axiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class Prox4Axiom(Axiom[Query, Document]):
    text_contents: TextContents[Union[Query, Document]]
    term_tokenizer: TermTokenizer

    def preference(
        self,
        input: Query,
        output1: Document,
        output2: Document,
    ) -> Preference:
        query_unique_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(input),
        )
        document1_terms = self.term_tokenizer.terms(
            self.text_contents.contents(output1),
        )
        document1_unique_terms = set(document1_terms)
        document2_terms = self.term_tokenizer.terms(
            self.text_contents.contents(output2),
        )
        document2_unique_terms = set(document2_terms)

        if not _all_query_terms_in_documents(
            query_terms=query_unique_terms,
            document1_terms=document1_unique_terms,
            document2_terms=document2_unique_terms,
        ):
            return 0

        occurrences1, count1 = _closest_grouping_size_and_count(
            query_terms=query_unique_terms,
            document_terms=document1_terms,
        )
        occurrences2, count2 = _closest_grouping_size_and_count(
            query_terms=query_unique_terms,
            document_terms=document2_terms,
        )

        if occurrences1 != occurrences2:
            return strictly_less(occurrences1, occurrences2)
        else:
            return strictly_greater(count1, count2)


PROX4: Final = lazy_inject(Prox4Axiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class Prox5Axiom(Axiom[Query, Document]):
    text_contents: TextContents[Union[Query, Document]]
    term_tokenizer: TermTokenizer

    def preference(
        self,
        input: Query,
        output1: Document,
        output2: Document,
    ) -> Preference:
        query_unique_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(input),
        )
        document1_terms = self.term_tokenizer.terms(
            self.text_contents.contents(output1),
        )
        document1_unique_terms = set(document1_terms)
        document2_terms = self.term_tokenizer.terms(
            self.text_contents.contents(output2),
        )
        document2_unique_terms = set(document2_terms)

        if not _all_query_terms_in_documents(
            query_terms=query_unique_terms,
            document1_terms=document1_unique_terms,
            document2_terms=document2_unique_terms,
        ):
            return 0

        smallest_span1 = _average_smallest_span(
            query_terms=query_unique_terms,
            document_terms=document1_terms,
        )
        smallest_span2 = _average_smallest_span(
            query_terms=query_unique_terms,
            document_terms=document2_terms,
        )

        return strictly_less(smallest_span1, smallest_span2)

    # ADDITION: Come up with a better way to batch-compute preference-matrices.


PROX5: Final = lazy_inject(Prox5Axiom, injector)


# ADDITION: QPHRA axiom:
#  For queries with highlighted phrases (e.g., via double quotes),
#  prefer documents containing all the query phrases over
#  documents not containing all phrases. [hagen:2016d]
