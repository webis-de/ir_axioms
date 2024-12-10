from itertools import product, combinations
from statistics import mean
from typing import (
    Iterable,
    Dict,
    Collection,
    Tuple,
    Sequence,
    Protocol,
    runtime_checkable,
    AbstractSet,
)


@runtime_checkable
class TermSimilarity(Protocol):
    def similarity(self, term1: str, term2: str) -> float:
        pass

    def similarity_sums(self, terms: Iterable[str]) -> Dict[str, float]:
        similarity_sums: Dict[str, float] = {term: 0 for term in terms}
        for term1, term2 in combinations(similarity_sums.keys(), 2):
            similarity = self.similarity(term1, term2)
            similarity_sums[term1] += similarity
            similarity_sums[term2] += similarity
        return similarity_sums

    def average_similarity(
        self, terms1: Collection[str], terms2: Collection[str]
    ) -> float:
        if len(terms1) == 0 or len(terms2) == 0:
            return 0
        return mean(
            self.similarity(term1, term2) for term1 in terms1 for term2 in terms2
        )

    def _pair_similarity(self, terms: Tuple[str, str]) -> float:
        term1, term2 = terms
        return self.similarity(term1, term2)

    def max_similarity_pairs(
        self, terms1: Collection[str], terms2: Collection[str]
    ) -> AbstractSet[Tuple[str, str]]:
        if len(terms1) == 0 or len(terms2) == 0:
            return set()
        most_similar_pairs: Sequence[Tuple[str, str]] = sorted(
            product(set(terms1), set(terms2)),
            key=self._pair_similarity,
            reverse=True,
        )
        max_similarity = self._pair_similarity(most_similar_pairs[0])
        print("max_similarity", max_similarity)
        return {
            pair
            for pair in most_similar_pairs
            if self._pair_similarity(pair) >= max_similarity
        }

    def max_average_similarity_terms(self, terms: Collection[str]) -> AbstractSet[str]:
        if len(terms) == 0:
            return set()
        similarity_sums = self.similarity_sums(set(terms))
        print("similarity_sums", similarity_sums)
        max_similarity_sum = max(similarity_sums.values())
        return {term for term in terms if similarity_sums[term] >= max_similarity_sum}

    def min_average_similarity_terms(self, terms: Collection[str]) -> AbstractSet[str]:
        if len(terms) == 0:
            return set()
        similarity_sums = self.similarity_sums(set(terms))
        print("similarity_sums", similarity_sums)
        min_similarity_sum = min(similarity_sums.values())
        return {term for term in terms if similarity_sums[term] <= min_similarity_sum}
