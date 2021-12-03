from abc import abstractmethod, ABC
from pathlib import Path
from typing import Set, List, Optional

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

    @abstractmethod
    def inverse_document_frequency(self, term: str) -> float:
        pass

    @abstractmethod
    def terms(self, text: str) -> List[str]:
        pass

    @abstractmethod
    def term_set(self, text: str) -> Set[str]:
        pass

    @abstractmethod
    def term_frequency(self, text: str, term: str) -> float:
        pass

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
