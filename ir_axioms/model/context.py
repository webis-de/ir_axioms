from abc import abstractmethod, ABC
from functools import lru_cache, cached_property
from math import log
from pathlib import Path
from typing import Optional, Union, FrozenSet, Sequence

from diskcache import Cache

from ir_axioms.model.base import Query, Document
from ir_axioms.model.retrieval_model import RetrievalModel


class IndexContext(ABC):
    cache_dir: Optional[Path] = None

    @cached_property
    def cache(self) -> Optional[Cache]:
        if self.cache_dir is None:
            return None
        return Cache(str(self.cache_dir.absolute()))

    @property
    @abstractmethod
    def document_count(self) -> int:
        pass

    @abstractmethod
    def document_frequency(self, term: str) -> int:
        pass

    @lru_cache(None)
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
                f"Expected Query or Document "
                f"but got {type(query_or_document)}."
            )

    @abstractmethod
    def terms(
            self,
            query_or_document: Union[Query, Document]
    ) -> Sequence[str]:
        pass

    def term_set(
            self,
            query_or_document: Union[Query, Document]
    ) -> FrozenSet[str]:
        return frozenset(self.terms(query_or_document))

    @lru_cache(None)
    def term_frequency(
            self,
            query_or_document: Union[Query, Document],
            term: str
    ) -> float:
        terms = self.terms(query_or_document)
        terms_len = len(terms)
        if terms_len == 0:
            return 0
        term_count = sum(1 for other in terms if other == term)
        return term_count / terms_len

    @abstractmethod
    def retrieval_score(
            self,
            query: Query,
            document: Document,
            model: RetrievalModel
    ) -> float:
        pass
