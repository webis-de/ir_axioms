from abc import abstractmethod, ABC
from functools import lru_cache
from math import log
from pathlib import Path
from typing import Set, List, Optional, Union

from ir_axioms.model import Query, Document


class RerankingContext(ABC):
    cache_dir: Optional[Path] = None

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
    def terms(self, query_or_document: Union[Query, Document]) -> List[str]:
        pass

    def term_set(self, query_or_document: Union[Query, Document]) -> Set[str]:
        return set(self.terms(query_or_document))

    @lru_cache
    def term_frequency(
            self,
            query_or_document: Union[Query, Document],
            term: str
    ) -> float:
        terms = self.terms(query_or_document)
        term_count = sum(1 for other in terms if other == term)
        return term_count / len(terms)

    @abstractmethod
    def tf_idf_score(
            self,
            query: Query,
            document: Document
    ) -> float:
        pass

    @abstractmethod
    def bm25_score(
            self,
            query: Query,
            document: Document,
            k1: float = 1.2,
            b: float = 0.75
    ) -> float:
        pass

    @abstractmethod
    def pl2_score(
            self,
            query: Query,
            document: Document,
            c: float = 0.1
    ) -> float:
        pass

    @abstractmethod
    def ql_score(
            self,
            query: Query,
            document: Document,
            mu: float = 1000
    ) -> float:
        pass
