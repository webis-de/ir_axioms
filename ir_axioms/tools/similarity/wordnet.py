from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Sequence, final

from nltk.corpus import wordnet

from ir_axioms.utils.nltk import download_nltk_dependencies
from ir_axioms.tools.similarity.base import TermSimilarity


@lru_cache(None)
def _synonym_set(term: str, smoothing: int = 0) -> Sequence[str]:
    cutoff = smoothing + 1
    return wordnet.synsets(term)[:cutoff]


@lru_cache(None)
def _synonym_set_similarity(term1: str, term2: str, smoothing: int = 0) -> float:
    synonyms_term1 = _synonym_set(term1, smoothing)
    synonyms_term2 = _synonym_set(term2, smoothing)

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


@dataclass(frozen=True)
class WordNetSynonymSetTermSimilarity(TermSimilarity):
    smoothing: int = 0

    def __post_init__(self):
        download_nltk_dependencies("wordnet", "omw-1.4")

    @final
    @lru_cache(None)
    def similarity(self, term1: str, term2: str) -> float:
        return _synonym_set_similarity(term1, term2, self.smoothing)
