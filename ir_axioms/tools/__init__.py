from injector import Module, Binder

# Re-export from sub-modules.

from ir_axioms.tools.aspects import (  # noqa: F401
    AspectExtraction,
    KeyBertAspectExtraction,
    YakeAspectExtraction,
    SpacyNounChunksAspectExtraction,
    SpacyEntitiesAspectExtraction,
    AspectsModule,
)

from ir_axioms.tools.contents import (  # noqa: F401
    TextContents,
    DocumentQueryTextContents,
    IrdsDocumentTextContents,
    IrdsQueryTextContents,
    AnseriniDocumentTextContents,
    TerrierDocumentTextContents,
    HasText,
    SimpleTextContents,
    ContentsModule,
)

from ir_axioms.tools.index_statistics import (  # noqa: F401
    IndexStatistics,
    AnseriniIndexStatistics,
    TerrierIndexStatistics,
)

from ir_axioms.tools.pivot import (  # noqa: F401
    PivotSelection,
    RandomPivotSelection,
    FirstPivotSelection,
    LastPivotSelection,
    MiddlePivotSelection,
    PivotModule,
)

from ir_axioms.tools.similarity import (  # noqa: F401
    TermSimilarity,
    SentenceSimilarity,
    FastTextTermSimilarity,
    WordNetSynonymSetTermSimilarity,
    SentenceTransformersSentenceSimilarity,
    SimilarityModule,
)

from ir_axioms.tools.text_statistics import (  # noqa: F401
    TextStatistics,
    DocumentQueryTextStatistics,
    AnseriniTextStatistics,
    TerrierTextStatistics,
    SimpleTextStatistics,
    TextStatisticsModule,
)

from ir_axioms.tools.tokenizer import (  # noqa: F401
    TermTokenizer,
    SentenceTokenizer,
    NltkTermTokenizer,
    NltkSentenceTokenizer,
    AnseriniTermTokenizer,
    TerrierTermTokenizer,
    TokenizerModule,
)


class ToolsModule(Module):
    def configure(self, binder: Binder) -> None:
        binder.install(AspectsModule)
        binder.install(ContentsModule)
        binder.install(PivotModule)
        binder.install(SimilarityModule)
        binder.install(TokenizerModule)
        # Need to be loaded after the tokenizer module because it needs it.
        binder.install(TextStatisticsModule)
