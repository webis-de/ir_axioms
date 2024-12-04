from abc import abstractmethod, ABC
from dataclasses import dataclass
from math import log
from typing import Union, FrozenSet, Sequence, Optional


@dataclass(frozen=True)
class Query:
    title: str


@dataclass(frozen=True)
class Document:
    id: str


@dataclass(frozen=True)
class TextDocument(Document):
    contents: str


@dataclass(frozen=True)
class ScoredDocument(Document):
    score: float

@dataclass(frozen=True)
class RankedDocument(Document):
    score: float
    rank: int


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
        if isinstance(query_or_document, Query):
            return query_or_document.title
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


@dataclass(unsafe_hash=True)
class _DefaultIndexContext(IndexContext):
    """
    Default lazy wrapper implementation to use as the global index context for all retrieval axioms (and preconditions).
    """

    context: Optional[IndexContext] = None

    @property
    def _context(self) -> IndexContext:
        if self.context is None:
            raise RuntimeError(
                "Must set an index context before using the default index context."
            )
        return self.context

    @property
    def document_count(self) -> int:
        return self._context.document_count

    def document_frequency(self, term: str) -> int:
        return self._context.document_frequency(term)

    def inverse_document_frequency(self, term: str) -> float:
        return self._context.inverse_document_frequency(term)

    def document_contents(self, document: Document) -> str:
        return self._context.document_contents(document)

    def contents(self, query_or_document: Union[Query, Document]) -> str:
        return self._context.contents(query_or_document)

    def terms(self, query_or_document: Union[Query, Document]) -> Sequence[str]:
        return self._context.terms(query_or_document)

    def term_set(self, query_or_document: Union[Query, Document]) -> FrozenSet[str]:
        return self._context.term_set(query_or_document)

    def document_length(self, document: Document):
        return self._context.document_length(document)

    def term_frequency(
        self, query_or_document: Union[Query, Document], term: str
    ) -> float:
        return self._context.term_frequency(query_or_document, term)


_DEFAULT_INDEX_CONTEXT = _DefaultIndexContext()


def get_index_context() -> IndexContext:
    return _DEFAULT_INDEX_CONTEXT


def set_index_context(context: IndexContext) -> None:
    _DEFAULT_INDEX_CONTEXT.context = context
