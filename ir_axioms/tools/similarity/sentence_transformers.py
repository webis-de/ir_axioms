from typing import TYPE_CHECKING

from ir_axioms.utils.libraries import is_sentence_transformers_installed

if is_sentence_transformers_installed() or TYPE_CHECKING:
    from dataclasses import dataclass
    from functools import cached_property
    from math import nan
    from typing import Sequence

    from numpy import array, float_, ndarray, dot
    from numpy.linalg import norm
    from numpy.typing import NDArray
    from sentence_transformers import SentenceTransformer

    from ir_axioms.tools.similarity.base import SentenceSimilarity

    def _cosine_similarity(vector1: ndarray, vector2: ndarray) -> float:
        divisor = norm(vector1) * norm(vector2)
        if divisor == 0:
            return nan
        return dot(vector1, vector2) / divisor

    @dataclass(frozen=True)
    class SentenceTransformersSentenceSimilarity(SentenceSimilarity):
        model_name: str = "sentence-transformers/all-mpnet-base-v2"

        @cached_property
        def model(self) -> SentenceTransformer:
            return SentenceTransformer(
                model_name_or_path=self.model_name,
            )

        def similarity(self, sentence1: str, sentence2: str) -> float:
            return self.self_similarities([sentence1, sentence2])[0, 1]

        def self_similarities(self, sentences: Sequence[str]) -> NDArray[float_]:
            vectors = self.model.encode(
                sentences=list(sentences),
                convert_to_numpy=True,
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
                convert_to_numpy=True,
            )
            vectors2 = self.model.encode(
                sentences=list(sentences2),
                convert_to_numpy=True,
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
    SentenceTransformersSentenceSimilarity = NotImplemented
