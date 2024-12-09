from itertools import product, combinations
from statistics import mean
from typing import (
    Iterable,
    Dict,
    Collection,
    Optional,
    Tuple,
    Sequence,
    Protocol,
    runtime_checkable,
)


from axioms.logging import logger


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

    def most_similar_pair(
        self, terms1: Collection[str], terms2: Collection[str]
    ) -> Optional[Tuple[str, str]]:
        if len(terms1) == 0 or len(terms2) == 0:
            return None
        most_similar_pairs: Sequence[Tuple[str, str]] = sorted(
            product(terms1, terms2),
            key=self._pair_similarity,
            reverse=True,
        )
        most_similar_pair = most_similar_pairs[0]
        if len(most_similar_pairs) > 1 and self._pair_similarity(
            most_similar_pair
        ) == self._pair_similarity(most_similar_pairs[1]):
            # No definite winner.
            logger.debug(
                f"Cannot find most similar term pair. "
                f"The following pairs were equally similar: "
                f"{', '.join(str(pair) for pair in most_similar_pairs)}"
            )
            return None

        return most_similar_pair

    def most_similar_term(
        self,
        terms: Collection[str],
    ) -> Optional[str]:
        if len(terms) == 0:
            return None
        similarity_sums = self.similarity_sums(terms)
        most_similar_terms: Sequence[str] = sorted(
            terms,
            key=lambda term: similarity_sums[term],
            reverse=True,
        )
        most_similar_term = most_similar_terms[0]
        if (
            len(most_similar_terms) > 1
            and similarity_sums[most_similar_term]
            == similarity_sums[most_similar_terms[1]]
        ):
            # No definite winner.
            logger.debug(
                f"Cannot find most similar term. "
                f"The following terms were equally similar: "
                f"{', '.join(most_similar_terms)}"
            )
            return None

        return most_similar_term

    def least_similar_term(
        self,
        terms: Collection[str],
    ) -> Optional[str]:
        if len(terms) == 0:
            return None
        similarity_sums = self.similarity_sums(terms)
        least_similar_terms: Sequence[str] = sorted(
            terms,
            key=lambda term: similarity_sums[term],
            reverse=False,
        )
        least_similar_term = least_similar_terms[0]
        if (
            len(least_similar_terms) > 1
            and similarity_sums[least_similar_term]
            == similarity_sums[least_similar_terms[1]]
        ):
            # No definite winner.
            logger.debug(
                f"Cannot find least similar term. "
                f"The following terms were equally similar: "
                f"{', '.join(least_similar_terms)}"
            )
            return None

        return least_similar_term
