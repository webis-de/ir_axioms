from dataclasses import dataclass
from functools import cached_property, lru_cache
from json import loads
from logging import warning
from pathlib import Path
from typing import Optional, Union, Callable, NamedTuple, Sequence

from ir_datasets import load, Dataset
from ir_datasets.indices import Docstore

from ir_axioms.backend.pyserini.safe import IndexReader, SimpleSearcher
from ir_axioms.backend.pyserini.util import (
    Similarity, ClassicSimilarity, BM25Similarity, DFRSimilarity,
    BasicModelIn, AfterEffectL, NormalizationH2, LMDirichletSimilarity
)
from ir_axioms.model import Query, Document, TextDocument, IndexContext
from ir_axioms.model.retrieval_model import (
    RetrievalModel, TfIdf, BM25, DirichletLM, PL2
)


@lru_cache(None)
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


@dataclass(frozen=True)
class PyseriniIndexContext(IndexContext):
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

    @lru_cache(None)
    def document_frequency(self, term: str) -> int:
        return self._index_reader.object.getDF(
            self._index_reader.reader,
            term
        )

    @cached_property
    def _searcher(self) -> SimpleSearcher:
        return SimpleSearcher(str(self.index_dir.absolute()))

    @lru_cache(None)
    def document_contents(self, document: Document) -> str:
        # Shortcut when text is given in the document.
        if isinstance(document, TextDocument):
            return document.contents

        # Shortcut when ir_dataset is specified.
        if self._dataset is not None:
            documents_store: Docstore = self._dataset.docs_store()
            try:
                document = documents_store.get(document.id)
                if self.contents_accessor is None:
                    return document.text
                elif isinstance(self.contents_accessor, str):
                    return getattr(document, self.contents_accessor)
                else:
                    return self.contents_accessor(document)
            except KeyError:
                # Document not found. Assume empty content.
                return ""

        document = self._searcher.doc(document.id)
        json_document = loads(document.raw())
        return json_document["contents"]

    @lru_cache(None)
    def terms(
            self,
            query_or_document: Union[Query, Document]
    ) -> Sequence[str]:
        text = self.contents(query_or_document)
        return tuple(str(term) for term in self._index_reader.analyze(text))

    @lru_cache(None)
    def retrieval_score(
            self,
            query: Query,
            document: Document,
            model: RetrievalModel
    ) -> float:
        return self._index_reader.compute_query_document_score(
            document.id,
            query.title,
            _similarity(model)
        )
