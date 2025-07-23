from itertools import product, combinations
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

from numpy import array, float_
from numpy.typing import NDArray


@runtime_checkable
class TermSimilarity(Protocol):
    def similarity(self, term1: str, term2: str) -> float:
        pass

    def similarities(self, terms: Sequence[str]) -> NDArray[float_]:
        return array(
            [
                self.similarity(
                    term1=term1,
                    term2=term2,
                )
                for term1 in terms
                for term2 in terms
            ],
            dtype=float_,
        ).reshape((len(terms), len(terms)))

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
        return array(
            [self.similarity(term1, term2) for term1 in terms1 for term2 in terms2]
        ).mean()

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
        return {
            pair
            for pair in most_similar_pairs
            if self._pair_similarity(pair) >= max_similarity
        }

    def max_average_similarity_terms(self, terms: Collection[str]) -> AbstractSet[str]:
        if len(terms) == 0:
            return set()
        similarity_sums = self.similarity_sums(set(terms))
        max_similarity_sum = max(similarity_sums.values())
        return {term for term in terms if similarity_sums[term] >= max_similarity_sum}

    def min_average_similarity_terms(self, terms: Collection[str]) -> AbstractSet[str]:
        if len(terms) == 0:
            return set()
        similarity_sums = self.similarity_sums(set(terms))
        min_similarity_sum = min(similarity_sums.values())
        return {term for term in terms if similarity_sums[term] <= min_similarity_sum}


@runtime_checkable
class SentenceSimilarity(Protocol):
    def similarity(self, sentence1: str, sentence2: str) -> float:
        pass

    def self_similarities(self, sentences: Sequence[str]) -> NDArray[float_]:
        return array(
            [
                self.similarity(
                    sentence1=sentence1,
                    sentence2=sentence2,
                )
                for sentence1 in sentences
                for sentence2 in sentences
            ],
            dtype=float_,
        ).reshape((len(sentences), len(sentences)))

    def paired_similarities(
        self, sentences1: Sequence[str], sentences2: Sequence[str]
    ) -> NDArray[float_]:
        return array(
            [
                self.similarity(
                    sentence1=sentence1,
                    sentence2=sentence2,
                )
                for sentence1 in sentences1
                for sentence2 in sentences2
            ],
            dtype=float_,
        ).reshape((len(sentences1), len(sentences2)))

    def average_similarity(
        self, sentences1: Collection[str], sentences2: Collection[str]
    ) -> float:
        return self.paired_similarities(list(sentences1), list(sentences2)).mean()
