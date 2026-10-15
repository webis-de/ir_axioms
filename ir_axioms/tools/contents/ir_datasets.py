from dataclasses import dataclass
from functools import cached_property
from collections.abc import Iterator, Iterable, Mapping

from ir_datasets import Dataset, load as irds_load
from ir_datasets.formats import GenericDoc, GenericQuery
from ir_datasets.indices import Docstore

from ir_axioms.model.retrieval import Document, Query
from ir_axioms.tools.contents.base import TextContents


@dataclass(frozen=True, kw_only=True)
class IrdsDocumentTextContents(TextContents[Document]):
    dataset: Dataset | str

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
        if input.text is not None:
            return input.text
        irds_document: GenericDoc = self._documents_store.get(input.id)
        return irds_document.default_text()

    def iter_contents(self, inputs: Iterable[Document]) -> Iterator[str]:
        inputs = list(inputs)
        missing_ids = {input.id for input in inputs if input.text is None}
        irds_documents: Mapping[str, GenericDoc] = (
            self._documents_store.get_many(missing_ids) if len(missing_ids) > 0 else {}
        )
        for input in inputs:
            if input.text is not None:
                yield input.text
            elif input.id in irds_documents:
                yield irds_documents[input.id].default_text()
            else:
                # Raise the same error as for a single missing document.
                yield self.contents(input)


@dataclass(frozen=True, kw_only=True)
class IrdsQueryTextContents(TextContents[Query]):
    dataset: Dataset | str

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
        if input.text is not None:
            return input.text
        return self._queries[input.id]
