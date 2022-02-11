from abc import ABC, abstractmethod
from functools import lru_cache, cached_property
from itertools import product, combinations
from typing import final, List, Final, Iterable, Dict

from nltk.corpus import wordnet
from pymagnitude import Magnitude

from ir_axioms.utils.nltk import download_nltk_dependencies


@lru_cache(None)
def synonym_set(
        term: str,
        smoothing: int = 0
) -> List[str]:
    download_nltk_dependencies("wordnet", "omw-1.4")
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


class WordNetSynonymSetTermSimilarityMixin(TermSimilarityMixin):
    smoothing: int = 0

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
