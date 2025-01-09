from injector import Module, Binder, singleton

# Re-export from sub-modules.

from axioms.tools.similarity.base import (  # noqa: F401
    TermSimilarity,
    SentenceSimilarity,
)

from axioms.tools.similarity.fasttext import (  # noqa: F401
    FastTextTermSimilarity,
)

from axioms.tools.similarity.wordnet import (  # noqa: F401
    WordNetSynonymSetTermSimilarity,
)

from axioms.tools.similarity.sentence_transformers import (  # noqa: F401
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
            to=SentenceTransformersSentenceSimilarity,
            scope=singleton,
        )
