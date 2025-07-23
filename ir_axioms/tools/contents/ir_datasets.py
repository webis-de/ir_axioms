from dataclasses import dataclass
from functools import cached_property
from typing import Iterable, Mapping, Union

from ir_datasets import Dataset, load as irds_load
from ir_datasets.formats import GenericDoc, GenericQuery
from ir_datasets.indices import Docstore

from ir_axioms.model.retrieval import Document, Query
from ir_axioms.tools.contents.base import TextContents
from ir_axioms.tools.contents.simple import FallbackTextContentsMixin


@dataclass(frozen=True, kw_only=True)
class IrdsDocumentTextContents(
    FallbackTextContentsMixin[Document], TextContents[Document]
):
    dataset: Union[Dataset, str]

    @cached_property
    def _dataset(self) -> Dataset:
        if isinstance(self.dataset, Dataset):
            return self.dataset
        else:
            return irds_load(self.dataset)

    @cached_property
    def _documents_store(self) -> Docstore:
        return self._dataset.docs_store()

    def contents(self, input: Document) -> str:
        irds_document: GenericDoc = self._documents_store.get(input.id)
        return irds_document.default_text()


@dataclass(frozen=True, kw_only=True)
class IrdsQueryTextContents(FallbackTextContentsMixin[Query], TextContents[Query]):
    dataset: Union[Dataset, str]

    @cached_property
    def _dataset(self) -> Dataset:
        if isinstance(self.dataset, Dataset):
            return self.dataset
        else:
            return irds_load(self.dataset)

    @cached_property
    def _queries(self) -> Mapping[str, str]:
        queries: Iterable[GenericQuery] = self._dataset.queries_iter()
        return {query.query_id: query.default_text() for query in queries}

    def contents(self, input: Query) -> str:
        return self._queries[input.id]
