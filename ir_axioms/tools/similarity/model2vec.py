from typing import TYPE_CHECKING

from ir_axioms.utils.libraries import is_model2vec_installed

if is_model2vec_installed() or TYPE_CHECKING:
    from dataclasses import dataclass
    from functools import cached_property
    from math import nan
    from typing import Sequence

    from model2vec import StaticModel
    from numpy import array, float_, ndarray, dot
    from numpy.linalg import norm
    from numpy.typing import NDArray

    from ir_axioms.tools.similarity.base import SentenceSimilarity

    def _cosine_similarity(vector1: ndarray, vector2: ndarray) -> float:
        divisor = norm(vector1) * norm(vector2)
        if divisor == 0:
            return nan
        return dot(vector1, vector2) / divisor

    @dataclass(frozen=True)
    class Model2VecSentenceSimilarity(SentenceSimilarity):
        model_name: str = "minishlab/potion-base-32M"

        @cached_property
        def model(self) -> StaticModel:
            return StaticModel.from_pretrained(
                path=self.model_name,
            )

        def similarity(self, sentence1: str, sentence2: str) -> float:
            return self.self_similarities([sentence1, sentence2])[0, 1]

        def self_similarities(self, sentences: Sequence[str]) -> NDArray[float_]:
            vectors = self.model.encode(
                sentences=list(sentences),
                show_progress_bar=False,
            )
            return array(
                [
                    _cosine_similarity(vectors[i1], vectors[i2])
                    for i1 in range(len(sentences))
                    for i2 in range(len(sentences))
                ],
                dtype=float_,
            ).reshape((len(sentences), len(sentences)))

        def paired_similarities(
            self, sentences1: Sequence[str], sentences2: Sequence[str]
        ) -> NDArray[float_]:
            vectors1 = self.model.encode(
                sentences=list(sentences1),
                show_progress_bar=False,
            )
            vectors2 = self.model.encode(
                sentences=list(sentences2),
                show_progress_bar=False,
            )
            return array(
                [
                    _cosine_similarity(vector1, vector2)
                    for vector1 in vectors1
                    for vector2 in vectors2
                ],
                dtype=float_,
            ).reshape((len(sentences1), len(sentences2)))

else:
    Model2VecSentenceSimilarity = NotImplemented  # type: ignore
