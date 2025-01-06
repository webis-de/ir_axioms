from dataclasses import dataclass
from typing import Union

from injector import inject

from axioms.model import Document, Query, GenerationInput, GenerationOutput
from axioms.tools.contents.base import TextContents


@inject
@dataclass(frozen=True, kw_only=True)
class DocumentQueryTextContents(TextContents[Union[Document, Query]]):
    document_text_contents: TextContents[Document]
    query_text_contents: TextContents[Query]

    def contents(self, input: Union[Document, Query]) -> str:
        if isinstance(input, Document):
            return self.document_text_contents.contents(input)
        elif isinstance(input, Query):
            return self.query_text_contents.contents(input)


@inject
@dataclass(frozen=True, kw_only=True)
class GenerationInputOutputTextContents(
    TextContents[Union[GenerationInput, GenerationOutput]]
):
    generation_input_text_contents: TextContents[GenerationInput]
    generation_output_text_contents: TextContents[GenerationOutput]

    def contents(self, input: Union[GenerationInput, GenerationOutput]) -> str:
        if isinstance(input, GenerationInput):
            return self.generation_input_text_contents.contents(input)
        elif isinstance(input, GenerationOutput):
            return self.generation_output_text_contents.contents(input)
