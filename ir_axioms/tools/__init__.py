from injector import Module, Binder

# Re-export from sub-modules.

from ir_axioms.tools.aspects import (
    AspectExtraction,
    KeyBertAspectExtraction,
    YakeAspectExtraction,
    SpacyNounChunksAspectExtraction,
    SpacyEntitiesAspectExtraction,
    AspectsModule,
)

from ir_axioms.tools.contents import (
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

from ir_axioms.tools.index_statistics import (
    IndexStatistics,
    AnseriniIndexStatistics,
    TerrierIndexStatistics,
)

from ir_axioms.tools.pivot import (
    PivotSelection,
    RandomPivotSelection,
    FirstPivotSelection,
    LastPivotSelection,
    MiddlePivotSelection,
    PivotModule,
)

from ir_axioms.tools.similarity import (
    TermSimilarity,
    SentenceSimilarity,
    FastTextTermSimilarity,
    WordNetSynonymSetTermSimilarity,
    SentenceTransformersSentenceSimilarity,
    SimilarityModule,
)

from ir_axioms.tools.text_statistics import (
    TextStatistics,
    DocumentQueryTextStatistics,
    AnseriniTextStatistics,
    TerrierDocumentTextStatistics,
    SimpleTextStatistics,
    TextStatisticsModule,
)

from ir_axioms.tools.tokenizer import (
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


__all__ = [
    "AspectExtraction",
    "KeyBertAspectExtraction",
    "YakeAspectExtraction",
    "SpacyNounChunksAspectExtraction",
    "SpacyEntitiesAspectExtraction",
    "AspectsModule",
    "TextContents",
    "DocumentQueryTextContents",
    "IrdsDocumentTextContents",
    "IrdsQueryTextContents",
    "AnseriniDocumentTextContents",
    "TerrierDocumentTextContents",
    "HasText",
    "SimpleTextContents",
    "ContentsModule",
    "IndexStatistics",
    "AnseriniIndexStatistics",
    "TerrierIndexStatistics",
    "PivotSelection",
    "RandomPivotSelection",
    "FirstPivotSelection",
    "LastPivotSelection",
    "MiddlePivotSelection",
    "PivotModule",
    "TermSimilarity",
    "SentenceSimilarity",
    "FastTextTermSimilarity",
    "WordNetSynonymSetTermSimilarity",
    "SentenceTransformersSentenceSimilarity",
    "SimilarityModule",
    "TextStatistics",
    "DocumentQueryTextStatistics",
    "AnseriniTextStatistics",
    "TerrierDocumentTextStatistics",
    "SimpleTextStatistics",
    "TextStatisticsModule",
    "TermTokenizer",
    "SentenceTokenizer",
    "NltkTermTokenizer",
    "NltkSentenceTokenizer",
    "AnseriniTermTokenizer",
    "TerrierTermTokenizer",
    "TokenizerModule",
    "ToolsModule",
]
