from injector import Module, Binder, singleton

from ir_axioms.utils.libraries import is_sentence_transformers_installed

# Re-export from sub-modules.

from ir_axioms.tools.similarity.base import (
    TermSimilarity,
    SentenceSimilarity,
)

from ir_axioms.tools.similarity.fasttext import (
    FastTextTermSimilarity,
)

from ir_axioms.tools.similarity.wordnet import (
    WordNetSynonymSetTermSimilarity,
)

from ir_axioms.tools.similarity.simple import (
    AverageTermSimilaritySentenceSimilarity,
)

from ir_axioms.tools.similarity.sentence_transformers import (
    SentenceTransformersSentenceSimilarity,
)


class SimilarityModule(Module):
    def configure(self, binder: Binder) -> None:
        binder.bind(
            interface=TermSimilarity,
            to=WordNetSynonymSetTermSimilarity,
            scope=singleton,
        )
        binder.bind(
            interface=TermSimilarity,
            to=FastTextTermSimilarity,
            scope=singleton,
        )

        binder.bind(
            interface=SentenceSimilarity,
            to=AverageTermSimilaritySentenceSimilarity,
            scope=singleton,
        )
        if is_sentence_transformers_installed():
            binder.bind(
                interface=SentenceSimilarity,
                to=SentenceTransformersSentenceSimilarity,
                scope=singleton,
            )


__all__ = [
    "TermSimilarity",
    "SentenceSimilarity",
    "FastTextTermSimilarity",
    "WordNetSynonymSetTermSimilarity",
    "AverageTermSimilaritySentenceSimilarity",
    "SentenceTransformersSentenceSimilarity",
    "SimilarityModule",
]
