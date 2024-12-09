from dataclasses import dataclass
from typing import Collection
from injector import inject

from axioms.dependency_injection import injector
from axioms.model import Document, TextDocument
from axioms.tools import IndexStatistics, TextContents, TermTokenizer


@dataclass(frozen=True)
class InMemoryDocumentCollection:
    documents: Collection[Document]


@inject
@dataclass(frozen=True)
class InMemoryIndexStatistics(IndexStatistics):
    document_collection: InMemoryDocumentCollection
    text_contents: TextContents[Document]
    term_tokenizer: TermTokenizer

    @property
    def document_count(self) -> int:
        return len(self.document_collection.documents)

    def document_frequency(self, term: str) -> int:
        return sum(
            1
            for document in self.document_collection.documents
            if term
            in self.term_tokenizer.terms(
                text=self.text_contents.contents(input=document),
            )
        )


def inject_documents(documents: Collection[TextDocument]) -> None:
    injector.binder.bind(
        interface=IndexStatistics,
        to=InMemoryIndexStatistics,
    )
    injector.binder.bind(
        interface=InMemoryDocumentCollection,
        to=InMemoryDocumentCollection(documents),
    )
