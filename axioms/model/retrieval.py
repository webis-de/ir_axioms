from abc import abstractmethod, ABC
from dataclasses import dataclass
from math import log
from typing import Union, FrozenSet, Sequence


@dataclass(frozen=True)
class Query:
    id: str


@dataclass(frozen=True)
class TextQuery(Query):
    text: str


@dataclass(frozen=True)
class Document:
    id: str


@dataclass(frozen=True)
class TextDocument(Document):
    text: str


@dataclass(frozen=True)
class ScoredDocument(Document):
    score: float


@dataclass(frozen=True)
class RankedDocument(Document):
    rank: int


@dataclass(frozen=True)
class ScoredTextDocument(ScoredDocument, TextDocument):
    pass


@dataclass(frozen=True)
class JudgedDocument(Document):
    relevance: float


class IndexContext(ABC):
    @property
    @abstractmethod
    def document_count(self) -> int:
        pass

    @abstractmethod
    def document_frequency(self, term: str) -> int:
        pass

    def inverse_document_frequency(self, term: str) -> float:
        document_frequency = self.document_frequency(term)
        if document_frequency == 0:
            return 0
        return log(self.document_count / document_frequency)

    @abstractmethod
    def document_contents(self, document: Document) -> str:
        pass

    def contents(self, query_or_document: Union[Query, Document]) -> str:
        if isinstance(query_or_document, TextQuery):
            return query_or_document.text
        elif isinstance(query_or_document, Document):
            return self.document_contents(query_or_document)
        else:
            raise ValueError(
                f"Expected Query or Document " f"but got {type(query_or_document)}."
            )

    @abstractmethod
    def terms(self, query_or_document: Union[Query, Document]) -> Sequence[str]:
        pass

    def term_set(self, query_or_document: Union[Query, Document]) -> FrozenSet[str]:
        return frozenset(self.terms(query_or_document))

    def document_length(self, document: Document):
        return len(self.terms(document))

    def term_frequency(
        self, query_or_document: Union[Query, Document], term: str
    ) -> float:
        terms = self.terms(query_or_document)
        terms_len = len(terms)
        if terms_len == 0:
            return 0
        term_count = sum(1 for other in terms if other == term)
        return term_count / terms_len
