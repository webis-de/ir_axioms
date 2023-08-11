from dataclasses import dataclass
from functools import reduce
from typing import Union, Set, Sequence

from nltk import word_tokenize

from ir_axioms.model import Query, Document, RankedTextDocument
from ir_axioms.model.context import IndexContext
from ir_axioms.model.retrieval_model import RetrievalModel
from ir_axioms.utils.nltk import download_nltk_dependencies


@dataclass(frozen=True)
class MemoryIndexContext(IndexContext):
    documents: Set[RankedTextDocument]

    # noinspection PyMethodMayBeStatic
    def __post_init__(self):
        download_nltk_dependencies("punkt")

    def __hash__(self):
        return reduce(
            lambda acc, document: acc * hash(document),
            self.documents,
            1,
        )

    @property
    def document_count(self) -> int:
        return len(self.documents)

    def document_frequency(self, term: str) -> int:
        return sum(
            1
            for document in self.documents
            if term in self.terms(document)
        )

    def document_contents(self, document: Document) -> str:
        text_document = next(
            text_document
            for text_document in self.documents
            if text_document.id == document.id
        )
        return text_document.contents

    def terms(
            self,
            query_or_document: Union[Query, Document]
    ) -> Sequence[str]:
        text = self.contents(query_or_document)
        return tuple(word_tokenize(text))

    def retrieval_score(
            self,
            query: Query,
            document: Document,
            model: RetrievalModel
    ) -> float:
        raise NotImplementedError()
