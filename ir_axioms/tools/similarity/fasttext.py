from typing import TYPE_CHECKING

from ir_axioms.utils.libraries import is_fasttext_installed

if is_fasttext_installed() or TYPE_CHECKING:
    from dataclasses import dataclass
    from functools import cached_property
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

        @cached_property
        def _similarities(self) -> dict[tuple[str, str], float]:
            return {}

        @final
        def similarity(self, term1: str, term2: str) -> float:
            key = (term1, term2)
            if key not in self._similarities:
                vector1 = self.model.get_word_vector(term1)
                vector2 = self.model.get_word_vector(term2)
                self._similarities[key] = _cosine_similarity(vector1, vector2)
            return self._similarities[key]

else:
    FastTextTermSimilarity = NotImplemented  # type: ignore
