# Check if Pyserini is installed.
try:
    import pyserini  # noqa: F401
except ImportError as error:
    raise ImportError(
        "The Pyserini backend requires that 'pyserini' is installed."
    ) from error

from dataclasses import dataclass
from functools import lru_cache, cached_property
from json import loads
from pathlib import Path
from typing import Optional, Union, Callable, NamedTuple, Sequence

from ir_datasets import load, Dataset
from ir_datasets.formats import GenericDoc
from ir_datasets.indices import Docstore
from pyserini.index import IndexReader
from pyserini.search import SimpleSearcher

from axioms.model import Query, Document, TextDocument, IndexContext


@dataclass(frozen=True, kw_only=True)
class AnseriniIndexContext(IndexContext):
    index_dir: Union[Path, str]
    dataset: Optional[Union[Dataset, str]] = None
    contents_accessor: Optional[Union[str, Callable[[NamedTuple], str]]] = "text"
    cache_dir: Optional[Path] = None

    @cached_property
    def _index_reader(self) -> IndexReader:
        if isinstance(self.index_dir, Path):
            return IndexReader(str(self.index_dir.absolute()))
        elif isinstance(self.index_dir, str):
            return IndexReader(self.index_dir)
        else:
            raise ValueError(f"Cannot load index from location {self.index_dir}.")

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
        return self._index_reader.object.getDF(self._index_reader.reader, term)

    @cached_property
    def _searcher(self) -> SimpleSearcher:
        if isinstance(self.index_dir, Path):
            return SimpleSearcher(str(self.index_dir.absolute()))
        elif isinstance(self.index_dir, str):
            return SimpleSearcher(self.index_dir)
        else:
            raise ValueError(f"Cannot load index from location {self.index_dir}.")

    @lru_cache(None)
    def document_contents(self, document: Document) -> str:
        # Shortcut when text is given in the document.
        if isinstance(document, TextDocument):
            return document.contents

        # Shortcut when ir_dataset is specified.
        if self._dataset is not None:
            documents_store: Docstore = self._dataset.docs_store()
            try:
                irds_document: GenericDoc = documents_store.get(document.id)
                if self.contents_accessor is None:
                    return irds_document.default_text()
                elif isinstance(self.contents_accessor, str):
                    return getattr(irds_document, self.contents_accessor)
                else:
                    return self.contents_accessor(irds_document)
            except KeyError:
                # Document not found. Assume empty content.
                return ""

        pynserini_document = self._searcher.doc(document.id)
        json_document = loads(pynserini_document.raw())
        return json_document["contents"]

    @lru_cache(None)
    def terms(self, query_or_document: Union[Query, Document]) -> Sequence[str]:
        text = self.contents(query_or_document)
        return tuple(str(term) for term in self._index_reader.analyze(text))
