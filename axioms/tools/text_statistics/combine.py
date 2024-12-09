from dataclasses import dataclass
from typing import Mapping, Union

from injector import inject

from axioms.model.retrieval import Document, Query
from axioms.tools.text_statistics.base import TextStatistics


@inject
@dataclass(frozen=True, kw_only=True)
class DocumentQueryTextStatistics(TextStatistics[Union[Document, Query]]):
    document_text_contents: TextStatistics[Document]
    query_text_contents: TextStatistics[Query]

    def term_frequencies(self, document: Union[Document, Query]) -> Mapping[str, int]:
        if isinstance(document, Document):
            return self.document_text_contents.term_frequencies(document)
        elif isinstance(document, Query):
            return self.query_text_contents.term_frequencies(document)
