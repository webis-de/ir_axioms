from collections.abc import Mapping
from dataclasses import dataclass
from typing import Union

from injector import inject

from ir_axioms.model import Document, Query, GenerationRequest, GenerationResponse
from ir_axioms.tools.text_statistics.base import TextStatistics


@inject
@dataclass(frozen=True, kw_only=True)
class DocumentQueryTextStatistics(TextStatistics[Union[Document, Query]]):
    document_text_contents: TextStatistics[Document]
    query_text_contents: TextStatistics[Query]

    def term_counts(self, document: Document | Query) -> Mapping[str, int]:
        if isinstance(document, Document):
            return self.document_text_contents.term_counts(document)
        elif isinstance(document, Query):
            return self.query_text_contents.term_counts(document)

    def term_frequencies(self, document: Document | Query) -> Mapping[str, float]:
        if isinstance(document, Document):
            return self.document_text_contents.term_frequencies(document)
        elif isinstance(document, Query):
            return self.query_text_contents.term_frequencies(document)


@inject
@dataclass(frozen=True, kw_only=True)
class GenerationInputOutputTextStatistics(
    TextStatistics[Union[GenerationRequest, GenerationResponse]]
):
    generation_input_text_statistics: TextStatistics[GenerationRequest]
    generation_output_text_statistics: TextStatistics[GenerationResponse]

    def term_counts(
        self, document: GenerationRequest | GenerationResponse
    ) -> Mapping[str, int]:
        if isinstance(document, GenerationRequest):
            return self.generation_input_text_statistics.term_counts(document)
        elif isinstance(document, GenerationResponse):
            return self.generation_output_text_statistics.term_counts(document)

    def term_frequencies(
        self, document: GenerationRequest | GenerationResponse
    ) -> Mapping[str, float]:
        if isinstance(document, GenerationRequest):
            return self.generation_input_text_statistics.term_frequencies(document)
        elif isinstance(document, GenerationResponse):
            return self.generation_output_text_statistics.term_frequencies(document)
