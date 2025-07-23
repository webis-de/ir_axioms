from dataclasses import dataclass
from functools import cached_property, lru_cache
from math import nan
from typing import final

from fasttext import load_model
from fasttext.FastText import _FastText
from huggingface_hub import hf_hub_download
from numpy import ndarray, dot
from numpy.linalg import norm

from ir_axioms.tools.similarity.base import TermSimilarity


def _cosine_similarity(vector1: ndarray, vector2: ndarray) -> float:
    divisor = norm(vector1) * norm(vector2)
    if divisor == 0:
        return nan
    return dot(vector1, vector2) / divisor


@dataclass(frozen=True)
class FastTextTermSimilarity(TermSimilarity):
    model_name: str = "facebook/fasttext-en-vectors"

    @cached_property
    def model(self) -> _FastText:
        model_path = hf_hub_download(  # nosec: B615
            repo_id=self.model_name,
            filename="model.bin",
        )
        return load_model(model_path)

    @final
    @lru_cache(None)
    def similarity(self, term1: str, term2: str) -> float:
        vector1 = self.model.get_word_vector(term1)
        vector2 = self.model.get_word_vector(term2)
        return _cosine_similarity(vector1, vector2)
