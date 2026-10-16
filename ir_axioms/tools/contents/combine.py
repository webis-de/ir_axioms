from dataclasses import dataclass

from injector import inject

from ir_axioms.model import Document, Query, GenerationRequest, GenerationResponse
from ir_axioms.tools.contents.base import TextContents


@inject
@dataclass(frozen=True, kw_only=True)
class DocumentQueryTextContents(TextContents[Document | Query]):
    document_text_contents: TextContents[Document]
    query_text_contents: TextContents[Query]

    def contents(self, input: Document | Query) -> str:
        if isinstance(input, Document):
            return self.document_text_contents.contents(input)
        elif isinstance(input, Query):
            return self.query_text_contents.contents(input)


@inject
@dataclass(frozen=True, kw_only=True)
class GenerationInputOutputTextContents(
    TextContents[GenerationRequest | GenerationResponse]
):
    generation_input_text_contents: TextContents[GenerationRequest]
    generation_output_text_contents: TextContents[GenerationResponse]

    def contents(self, input: GenerationRequest | GenerationResponse) -> str:
        if isinstance(input, GenerationRequest):
            return self.generation_input_text_contents.contents(input)
        elif isinstance(input, GenerationResponse):
            return self.generation_output_text_contents.contents(input)
