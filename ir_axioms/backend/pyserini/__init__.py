from dataclasses import dataclass
from functools import cached_property, lru_cache
from math import log
from pathlib import Path
from typing import List, Set, Optional

from ir_axioms.backend import PyseriniBackend
from ir_axioms.model import Query, Document
from ir_axioms.model.context import RerankingContext

with PyseriniBackend():
    from pyserini.index import IndexReader
    from ir_axioms.backend.pyserini.util import (
        JSimilarity, JClassicSimilarity, JBM25Similarity, JDFRSimilarity,
        JBasicModelIn, JAfterEffectL, JNormalizationH2, JLMDirichletSimilarity
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
        return self._index_reader.object.getDF(self._index_reader.reader,
                                               term)

    def inverse_document_frequency(self, term: str) -> float:
        document_frequency = self.document_frequency(term)
        if document_frequency == 0:
            return 0
        return log(self.document_count / document_frequency)

    @lru_cache
    def terms(self, text: str) -> List[str]:
        return self._index_reader.analyze(text)

    def term_set(self, text: str) -> Set[str]:
        return set(self.terms(text))

    @lru_cache
    def term_frequency(self, text: str, term: str) -> float:
        # TODO: Is this correctly implemented?
        terms = self.terms(text)
        term_count = sum(1 for other in terms if other == term)
        return term_count / len(terms)

    @staticmethod
    @lru_cache
    def _tf_idf_similarity() -> JSimilarity:
        return JClassicSimilarity()

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
    def _bm25_similarity(k1: float = 1.2, b: float = 0.75) -> JSimilarity:
        return JBM25Similarity(k1, b)

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
    def _pl2_similarity(c: float = 0.1) -> JSimilarity:
        return JDFRSimilarity(
            JBasicModelIn(),
            JAfterEffectL(),
            JNormalizationH2(c)
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
    def _ql_similarity(mu: float = 1000) -> JSimilarity:
        return JLMDirichletSimilarity(mu)

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
