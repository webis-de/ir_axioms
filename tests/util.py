from dataclasses import dataclass
from typing import Collection
from injector import inject, Injector

from ir_axioms.dependency_injection import injector as _default_injector
from ir_axioms.model import Document
from ir_axioms.tools import IndexStatistics, TextContents, TermTokenizer


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
            in self.term_tokenizer.unique_terms(
                text=self.text_contents.contents(input=document),
            )
        )


def inject_documents(
    documents: Collection[Document],
    injector: Injector = _default_injector,
) -> None:
    injector.binder.bind(
        interface=IndexStatistics,
        to=InMemoryIndexStatistics,
    )
    injector.binder.bind(
        interface=InMemoryDocumentCollection,
        to=InMemoryDocumentCollection(documents),
    )
