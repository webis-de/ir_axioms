from typing import Union

from injector import Module, Binder, singleton, inject

from ir_axioms.model import Document, Query, GenerationInput, GenerationOutput

# Re-export from sub-modules.
from ir_axioms.tools.text_statistics.base import (  # noqa: F401
    TextStatistics,
)

from ir_axioms.tools.text_statistics.combine import (  # noqa: F401
    DocumentQueryTextStatistics,
    GenerationInputOutputTextStatistics,
)

from ir_axioms.tools.text_statistics.pyserini import (  # noqa: F401
    AnseriniTextStatistics,
)

from ir_axioms.tools.text_statistics.pyterrier import (  # noqa: F401
    TerrierTextStatistics,
)

from ir_axioms.tools.text_statistics.simple import (  # noqa: F401
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
            text_contents: TextContents[GenerationInput],
            term_tokenizer: TermTokenizer,
        ) -> TextStatistics[GenerationInput]:
            return SimpleTextStatistics(
                text_contents=text_contents,
                term_tokenizer=term_tokenizer,
            )
        
        @inject
        def _make_simple_text_statistics_generation_output(
            text_contents: TextContents[GenerationOutput],
            term_tokenizer: TermTokenizer,
        ) -> TextStatistics[GenerationOutput]:
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
            interface=TextStatistics[GenerationInput],
            to=_make_simple_text_statistics_generation_input,
            scope=singleton,
        )
        binder.bind(
            interface=TextStatistics[GenerationOutput],
            to=_make_simple_text_statistics_generation_output,
            scope=singleton,
        )
        binder.bind(
            interface=TextStatistics[Union[Query, Document]],
            to=DocumentQueryTextStatistics,
            scope=singleton,
        )
        binder.bind(
            interface=TextStatistics[Union[GenerationInput, GenerationOutput]],
            to=GenerationInputOutputTextStatistics,
            scope=singleton,
        )
