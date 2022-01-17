from dataclasses import dataclass
from functools import cached_property, lru_cache
from json import loads
from logging import warning
from pathlib import Path
from typing import List, Optional, Union, Callable, NamedTuple

from ir_datasets import load, Dataset
from ir_datasets.indices import Docstore
from pyserini.search import SimpleSearcher

from ir_axioms.backend import PyseriniBackendContext
from ir_axioms.model import Query, Document, TextDocument
from ir_axioms.model.context import RerankingContext
from ir_axioms.model.retrieval_model import (
    RetrievalModel, TfIdf, BM25, DirichletLM, PL2
)

with PyseriniBackendContext():
    from pyserini.index import IndexReader
    from ir_axioms.backend.pyserini.util import (
        Similarity, ClassicSimilarity, BM25Similarity, DFRSimilarity,
        BasicModelIn, AfterEffectL, NormalizationH2, LMDirichletSimilarity
    )


    @dataclass(unsafe_hash=True, frozen=True)
    class IndexRerankingContext(RerankingContext):
        index_dir: Path
        dataset: Optional[Union[Dataset, str]] = None
        contents_accessor: Optional[Union[
            str,
            Callable[[NamedTuple], str]
        ]] = "text"
        cache_dir: Optional[Path] = None

        @cached_property
        def _index_reader(self) -> IndexReader:
            return IndexReader(str(self.index_dir.absolute()))

        @cached_property
        def _dataset(self) -> Optional[Dataset]:
            if self.dataset is None:
                return None
            elif isinstance(self.dataset, Dataset):
                return self.dataset
            else:
                return load(self.dataset)

        @cached_property
        def document_count(self) -> int:
            return self._index_reader.stats()["documents"]

        def document_frequency(self, term: str) -> int:
            return self._index_reader.object.getDF(
                self._index_reader.reader,
                term
            )

        @cached_property
        def _searcher(self) -> SimpleSearcher:
            return SimpleSearcher(str(self.index_dir.absolute()))

        def document_contents(self, document: Document) -> str:
            # Shortcut when text is given in the document.
            if isinstance(document, TextDocument):
                return document.contents

            # Shortcut when ir_dataset is specified.
            if self._dataset is not None:
                documents_store: Docstore = self._dataset.docs_store()
                document = documents_store.get(document.id)
                if self.contents_accessor is None:
                    return document.text
                elif isinstance(self.contents_accessor, str):
                    return getattr(document, self.contents_accessor)
                else:
                    return self.contents_accessor(document)

            document = self._searcher.doc(document.id)
            json_document = loads(document.raw())
            return json_document["contents"]

        @lru_cache
        def terms(
                self,
                query_or_document: Union[Query, Document]
        ) -> List[str]:
            text = self.contents(query_or_document)
            return self._index_reader.analyze(text)

        @staticmethod
        @lru_cache
        def _similarity(model: RetrievalModel) -> Similarity:
            if isinstance(model, TfIdf):
                return ClassicSimilarity()
            elif isinstance(model, BM25):
                if model.k_3 != 8:
                    warning(
                        "The Pyserini backend doesn't support setting "
                        "the k_3 parameter for the BM25 retrieval model. "
                        "It will be ignored."
                    )
                return BM25Similarity(model.k_1, model.b)
            elif isinstance(model, PL2):
                return DFRSimilarity(
                    BasicModelIn(),
                    AfterEffectL(),
                    NormalizationH2(model.c)
                )
            elif isinstance(model, DirichletLM):
                return LMDirichletSimilarity(model.mu)
            else:
                raise NotImplementedError(
                    f"The Pyserini backend doesn't support "
                    f"the {type(model)} retrieval model."
                )

        def retrieval_score(
                self,
                query: Query,
                document: Document,
                model: RetrievalModel
        ) -> float:
            return self._index_reader.compute_query_document_score(
                document.id,
                query.title,
                self._similarity(model)
            )
