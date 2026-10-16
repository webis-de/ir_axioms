from injector import Module, Binder, singleton, inject

from ir_axioms.model import Document, Query, GenerationRequest, GenerationResponse

# Re-export from sub-modules.
from ir_axioms.tools.text_statistics.base import (
    TextStatistics,
)

from ir_axioms.tools.text_statistics.combine import (
    DocumentQueryTextStatistics,
    GenerationInputOutputTextStatistics,
)

from ir_axioms.tools.text_statistics.pyserini import (
    AnseriniTextStatistics,
)

from ir_axioms.tools.text_statistics.pyterrier import (
    TerrierDocumentTextStatistics,
)

from ir_axioms.tools.text_statistics.simple import (
    SimpleTextStatistics,
)


class TextStatisticsModule(Module):
    def configure(self, binder: Binder) -> None:
        from ir_axioms.tools import TextContents, TermTokenizer

        @inject
        def _make_simple_text_statistics_query(
            text_contents: TextContents[Query],
            term_tokenizer: TermTokenizer,
        ) -> TextStatistics[Query]:
            return SimpleTextStatistics(
                text_contents=text_contents,
                term_tokenizer=term_tokenizer,
            )

        @inject
        def _make_simple_text_statistics_document(
            text_contents: TextContents[Document],
            term_tokenizer: TermTokenizer,
        ) -> TextStatistics[Document]:
            return SimpleTextStatistics(
                text_contents=text_contents,
                term_tokenizer=term_tokenizer,
            )

        @inject
        def _make_simple_text_statistics_generation_input(
            text_contents: TextContents[GenerationRequest],
            term_tokenizer: TermTokenizer,
        ) -> TextStatistics[GenerationRequest]:
            return SimpleTextStatistics(
                text_contents=text_contents,
                term_tokenizer=term_tokenizer,
            )

        @inject
        def _make_simple_text_statistics_generation_output(
            text_contents: TextContents[GenerationResponse],
            term_tokenizer: TermTokenizer,
        ) -> TextStatistics[GenerationResponse]:
            return SimpleTextStatistics(
                text_contents=text_contents,
                term_tokenizer=term_tokenizer,
            )

        binder.bind(
            interface=TextStatistics[Query],
            to=_make_simple_text_statistics_query,
            scope=singleton,
        )
        binder.bind(
            interface=TextStatistics[Document],
            to=_make_simple_text_statistics_document,
            scope=singleton,
        )
        binder.bind(
            interface=TextStatistics[GenerationRequest],
            to=_make_simple_text_statistics_generation_input,
            scope=singleton,
        )
        binder.bind(
            interface=TextStatistics[GenerationResponse],
            to=_make_simple_text_statistics_generation_output,
            scope=singleton,
        )
        binder.bind(
            interface=TextStatistics[Query | Document],
            to=DocumentQueryTextStatistics,
            scope=singleton,
        )
        binder.bind(
            interface=TextStatistics[GenerationRequest | GenerationResponse],
            to=GenerationInputOutputTextStatistics,
            scope=singleton,
        )


__all__ = [
    "TextStatistics",
    "DocumentQueryTextStatistics",
    "GenerationInputOutputTextStatistics",
    "AnseriniTextStatistics",
    "TerrierDocumentTextStatistics",
    "SimpleTextStatistics",
    "TextStatisticsModule",
]
