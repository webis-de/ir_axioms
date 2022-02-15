from abc import ABC, abstractmethod
from functools import lru_cache, cached_property
from itertools import product, combinations
from statistics import mean
from typing import (
    final, Final, Iterable, Dict, Collection, Optional, Tuple, Sequence
)

from nltk.corpus import wordnet
from pymagnitude import Magnitude

from ir_axioms.utils.nltk import download_nltk_dependencies


@lru_cache(None)
def synonym_set(
        term: str,
        smoothing: int = 0
) -> Sequence[str]:
    cutoff = smoothing + 1
    return wordnet.synsets(term)[:cutoff]


@lru_cache(None)
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


class TermSimilarityMixin(ABC):
    @abstractmethod
    def similarity(self, term1: str, term2: str) -> float:
        pass

    @final
    def similarity_sums(self, terms: Iterable[str]) -> Dict[str, float]:
        similarity_sums: Dict[str, float] = {
            term: 0
            for term in terms
        }
        for term1, term2 in combinations(similarity_sums.keys(), 2):
            similarity = self.similarity(term1, term2)
            similarity_sums[term1] += similarity
            similarity_sums[term2] += similarity
        return similarity_sums

    @final
    def average_similarity(
            self,
            terms1: Collection[str],
            terms2: Collection[str]
    ) -> float:
        if len(terms1) == 0 or len(terms2) == 0:
            return 0
        return mean(
            self.similarity(term1, term2)
            for term1 in terms1
            for term2 in terms2
        )

    def _pair_similarity(self, terms: Tuple[str, str]) -> float:
        term1, term2 = terms
        return self.similarity(term1, term2)

    @final
    def most_similar_pair(
            self,
            terms1: Collection[str],
            terms2: Collection[str]
    ) -> Optional[Tuple[str, str]]:
        if len(terms1) == 0 or len(terms2) == 0:
            return None
        return max(
            product(terms1, terms2),
            key=self._pair_similarity
        )


class WordNetSynonymSetTermSimilarityMixin(TermSimilarityMixin):
    smoothing: int = 0

    def __init__(self):
        self.__post_init__()

    # noinspection PyMethodMayBeStatic
    def __post_init__(self):
        download_nltk_dependencies("wordnet", "omw-1.4")

    @final
    @lru_cache(None)
    def similarity(self, term1: str, term2: str) -> float:
        return synonym_set_similarity(term1, term2, self.smoothing)


class MagnitudeTermSimilarityMixin(TermSimilarityMixin, ABC):
    embeddings_path: str = NotImplemented

    @cached_property
    def _embeddings(self):
        return Magnitude(self.embeddings_path)

    @final
    @lru_cache(None)
    def similarity(self, term1: str, term2: str):
        return float(self._embeddings.similarity(term1, term2))


class FastTextWikiNewsTermSimilarityMixin(MagnitudeTermSimilarityMixin):
    embeddings_path: Final[str] = "fasttext/medium/wiki-news-300d-1M.magnitude"
