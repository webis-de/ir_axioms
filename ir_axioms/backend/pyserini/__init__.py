from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import List, Optional, Union

from ir_axioms.backend import PyseriniBackendContext
from ir_axioms.model import Query, Document
from ir_axioms.model.context import RerankingContext

with PyseriniBackendContext():
    from pyserini.index import IndexReader
    from ir_axioms.backend.pyserini.util import (
        Similarity, ClassicSimilarity, BM25Similarity, DFRSimilarity,
        BasicModelIn, AfterEffectL, NormalizationH2, LMDirichletSimilarity
    )


@dataclass(unsafe_hash=True, frozen=True)
class IndexRerankingContext(RerankingContext):
    index_dir: Path
    cache_dir: Optional[Path] = None

    @cached_property
    def _index_reader(self) -> IndexReader:
        return IndexReader(str(self.index_dir.absolute()))

    @cached_property
    def document_count(self) -> int:
        return self._index_reader.stats()["documents"]

    def document_frequency(self, term: str) -> int:
        return self._index_reader.object.getDF(
            self._index_reader.reader,
            term
        )

    @staticmethod
    def _text(query_or_document: Union[Query, Document]) -> str:
        if isinstance(query_or_document, Query):
            return query_or_document.title
        elif isinstance(query_or_document, Document):
            return query_or_document.content
        else:
            raise ValueError(
                f"Expected Query or Document "
                f"but got {type(query_or_document)}."
            )

    @lru_cache
    def terms(self, query_or_document: Union[Query, Document]) -> List[str]:
        text = self._text(query_or_document)
        return self._index_reader.analyze(text)

    @staticmethod
    @lru_cache
    def _tf_idf_similarity() -> Similarity:
        return ClassicSimilarity()

    def tf_idf_score(
            self,
            query: Query,
            document: Document
    ) -> float:
        return self._index_reader.compute_query_document_score(
            document.id,
            query.title,
            self._tf_idf_similarity()
        )

    @staticmethod
    @lru_cache
    def _bm25_similarity(k1: float = 1.2, b: float = 0.75) -> Similarity:
        return BM25Similarity(k1, b)

    def bm25_score(
            self,
            query: Query,
            document: Document,
            k1: float = 1.2,
            b: float = 0.75
    ) -> float:
        return self._index_reader.compute_query_document_score(
            document.id,
            query.title,
            self._bm25_similarity(k1, b)
        )

    @staticmethod
    @lru_cache
    def _pl2_similarity(c: float = 0.1) -> Similarity:
        return DFRSimilarity(
            BasicModelIn(),
            AfterEffectL(),
            NormalizationH2(c)
        )

    def pl2_score(
            self,
            query: Query,
            document: Document,
            c: float = 0.1
    ) -> float:
        return self._index_reader.compute_query_document_score(
            document.id,
            query.title,
            self._pl2_similarity(c)
        )

    @staticmethod
    @lru_cache
    def _ql_similarity(mu: float = 1000) -> Similarity:
        return LMDirichletSimilarity(mu)

    def ql_score(
            self,
            query: Query,
            document: Document,
            mu: float = 1000
    ) -> float:
        return self._index_reader.compute_query_document_score(
            document.id,
            query.title,
            self._ql_similarity(mu)
        )
