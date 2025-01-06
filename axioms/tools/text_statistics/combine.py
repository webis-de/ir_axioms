from dataclasses import dataclass
from typing import Mapping, Union

from injector import inject

from axioms.model import Document, Query, GenerationInput, GenerationOutput
from axioms.tools.text_statistics.base import TextStatistics


@inject
@dataclass(frozen=True, kw_only=True)
class DocumentQueryTextStatistics(TextStatistics[Union[Document, Query]]):
    document_text_contents: TextStatistics[Document]
    query_text_contents: TextStatistics[Query]

    def term_counts(self, document: Union[Document, Query]) -> Mapping[str, int]:
        if isinstance(document, Document):
            return self.document_text_contents.term_counts(document)
        elif isinstance(document, Query):
            return self.query_text_contents.term_counts(document)

    def term_frequencies(self, document: Union[Document, Query]) -> Mapping[str, float]:
        if isinstance(document, Document):
            return self.document_text_contents.term_frequencies(document)
        elif isinstance(document, Query):
            return self.query_text_contents.term_frequencies(document)


@inject
@dataclass(frozen=True, kw_only=True)
class GenerationInputOutputTextStatistics(
    TextStatistics[Union[GenerationInput, GenerationOutput]]
):
    generation_input_text_statistics: TextStatistics[GenerationInput]
    generation_output_text_statistics: TextStatistics[GenerationOutput]

    def term_counts(
        self, document: Union[GenerationInput, GenerationOutput]
    ) -> Mapping[str, int]:
        if isinstance(document, GenerationInput):
            return self.generation_input_text_statistics.term_counts(document)
        elif isinstance(document, GenerationOutput):
            return self.generation_output_text_statistics.term_counts(document)

    def term_frequencies(
        self, document: Union[GenerationInput, GenerationOutput]
    ) -> Mapping[str, float]:
        if isinstance(document, GenerationInput):
            return self.generation_input_text_statistics.term_frequencies(document)
        elif isinstance(document, GenerationOutput):
            return self.generation_output_text_statistics.term_frequencies(document)
