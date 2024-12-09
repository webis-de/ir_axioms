from dataclasses import dataclass
from typing import Union

from injector import inject

from axioms.model.retrieval import Document, Query
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
