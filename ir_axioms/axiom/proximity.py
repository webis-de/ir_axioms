from math import inf
from typing import List

from ir_axioms.axiom import Axiom
from ir_axioms.axiom.utils import (
    strictly_less, strictly_greater, same_query_term_subset,
    average_between_query_terms, all_query_terms_in_documents,
    closest_grouping_size_and_count, average_smallest_span
)
from ir_axioms.model import Query, RankedDocument
from ir_axioms.model.context import RerankingContext


class PROX1(Axiom):
    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        if not same_query_term_subset(context, query, document1, document2):
            return 0

        query_terms = context.term_set(query)
        document1_terms = context.terms(document1)
        document2_terms = context.terms(document2)

        overlapping_terms = (
                query_terms &
                set(document1_terms) &
                set(document2_terms)
        )

        average1 = average_between_query_terms(
            overlapping_terms,
            document1_terms
        )
        average2 = average_between_query_terms(
            overlapping_terms,
            document2_terms
        )

        return strictly_greater(average2, average1)


class PROX2(Axiom):
    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        if not same_query_term_subset(context, query, document1, document2):
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


class PROX3(Axiom):
    @staticmethod
    def find_index(query_terms: List[str], document_terms: List[str]):
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

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        if not same_query_term_subset(context, query, document1, document2):
            return 0
        query_terms = context.terms(query)
        document1_terms = context.terms(document1)
        document2_terms = context.terms(document2)
        return strictly_less(
            self.find_index(query_terms, document1_terms),
            self.find_index(query_terms, document2_terms)
        )


class PROX4(Axiom):
    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        if not all_query_terms_in_documents(
                context,
                query,
                document1,
                document2
        ):
            return 0

        query_terms = context.term_set(query)
        document1_terms = context.terms(document1)
        document2_terms = context.terms(document2)

        occurrences1, count1 = closest_grouping_size_and_count(
            query_terms,
            document1_terms
        )
        occurrences2, count2 = closest_grouping_size_and_count(
            query_terms,
            document2_terms
        )

        if occurrences1 != occurrences2:
            return strictly_less(occurrences1, occurrences2)
        else:
            return strictly_greater(count1, count2)


class PROX5(Axiom):
    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        if not all_query_terms_in_documents(
                context,
                query,
                document1,
                document2
        ):
            return 0

        query_terms = context.term_set(query)
        document1_terms = context.terms(document1)
        document2_terms = context.terms(document2)

        smallest_span1 = average_smallest_span(query_terms, document1_terms)
        smallest_span2 = average_smallest_span(query_terms, document2_terms)

        return strictly_less(smallest_span1, smallest_span2)
