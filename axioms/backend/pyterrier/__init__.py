# Check if PyTerrier is installed.
try:
    import pyterrier # noqa: F401
except ImportError as error:
    raise ImportError(
        "The PyTerrier backend requires that 'python-terrier' is installed."
    ) from error

from dataclasses import dataclass
from functools import lru_cache, cached_property
from pathlib import Path
from re import split
from typing import Optional, Union, Callable, NamedTuple, Sequence, TYPE_CHECKING, TypeAlias, Any

from ir_datasets import Dataset, load
from ir_datasets.indices import Docstore
from jnius import autoclass
from pyterrier.datasets import IRDSDataset
from pyterrier.java import init

from axioms.model import Query, Document, TextDocument, IndexContext


# Ensure that PyTerrier's JVM is started
init()

# Load Java classes.
if TYPE_CHECKING:
    StringReader: TypeAlias = Any
    Index: TypeAlias = Any
    IndexRef: TypeAlias = Any
    IndexFactory: TypeAlias = Any
    MetaIndex: TypeAlias = Any
    Lexicon: TypeAlias = Any
    CollectionStatistics: TypeAlias = Any
    Tokeniser: TypeAlias = Any
    EnglishTokeniser: TypeAlias = Any
    TermPipelineAccessor: TypeAlias = Any
    BaseTermPipelineAccessor: TypeAlias = Any
    ApplicationSetup: TypeAlias = Any
else:
    StringReader = autoclass("java.io.StringReader")
    Index = autoclass("org.terrier.structures.Index")
    IndexRef = autoclass('org.terrier.querying.IndexRef')
    IndexFactory = autoclass('org.terrier.structures.IndexFactory')
    MetaIndex = autoclass("org.terrier.structures.MetaIndex")
    Lexicon = autoclass("org.terrier.structures.Lexicon")
    CollectionStatistics = autoclass("org.terrier.structures.CollectionStatistics")
    Tokeniser = autoclass("org.terrier.indexing.tokenisation.Tokeniser")
    EnglishTokeniser = autoclass(
        "org.terrier.indexing.tokenisation.EnglishTokeniser"
    )
    TermPipelineAccessor = autoclass("org.terrier.terms.TermPipelineAccessor")
    BaseTermPipelineAccessor = autoclass(
        "org.terrier.terms.BaseTermPipelineAccessor"
    )
    ApplicationSetup = autoclass('org.terrier.utility.ApplicationSetup')

ContentsAccessor = Union[str, Callable[[NamedTuple], str]]


@dataclass(frozen=True, kw_only=True)
class TerrierIndexContext(IndexContext):
    index_location: Union[Index, IndexRef, Path, str]
    dataset: Optional[Union[Dataset, str, IRDSDataset]] = None
    contents_accessor: Optional[ContentsAccessor] = "text"
    tokeniser: Optional[Tokeniser] = None
    cache_dir: Optional[Path] = None

    @cached_property
    def _index(self) -> Index:
        if not TYPE_CHECKING and isinstance(self.index_location, Index):
            return self.index_location
        elif not TYPE_CHECKING and isinstance(self.index_location, IndexRef):
            return IndexFactory.of(self.index_location)
        elif isinstance(self.index_location, str):
            return IndexFactory.of(self.index_location)
        elif isinstance(self.index_location, Path):
            return IndexFactory.of(str(self.index_location.absolute()))
        else:
            raise ValueError(
                f"Cannot load index from location {self.index_location}."
            )

    def __str__(self):
        ret = str(self.index_location)
        ret = ret.split("/")[-1].split(" ")[0]
        return f'TerrierIndexContext({ret})'

    @cached_property
    def _meta_index(self) -> MetaIndex:
        meta_index = self._index.getMetaIndex()
        if meta_index is None:
            raise ValueError(
                f"Index {self.index_location} does not have a metaindex."
            )
        return meta_index

    @cached_property
    def _meta_index_keys(self) -> Sequence[str]:
        return tuple(str(key) for key in self._meta_index.getKeys())

    @cached_property
    def _lexicon(self) -> Lexicon:
        return self._index.getLexicon()

    @cached_property
    def _collection_statistics(self) -> CollectionStatistics:
        return self._index.getCollectionStatistics()

    @cached_property
    def _dataset(self) -> Optional[Dataset]:
        if self.dataset is None:
            return None
        elif isinstance(self.dataset, Dataset):
            return self.dataset
        elif isinstance(self.dataset, str):
            return load(self.dataset)
        elif isinstance(self.dataset, IRDSDataset):
            return self.dataset.irds_ref()
        else:
            raise ValueError(f"Cannot load dataset {self.dataset}.")

    @cached_property
    def _dataset_doc_store(self) -> Optional[Docstore]:
        if self._dataset is None:
            return None
        else:
            return self._dataset.docs_store()

    @cached_property
    def document_count(self) -> int:
        return int(self._collection_statistics.numberOfDocuments)

    @lru_cache(None)
    def document_frequency(self, term: str) -> int:
        entry = self._lexicon.getLexiconEntry(term)
        if entry is None or entry.getNumberOfEntries() == 0:
            del entry
            return 0
        else:
            document_frequency = int(entry.getDocumentFrequency())
            del entry
            return document_frequency

    @lru_cache(None)
    def _document_contents(self, document_id: str) -> str:
        # Shortcut when ir_dataset is specified.
        documents_store = self._dataset_doc_store
        if documents_store is not None:
            try:
                store_document = documents_store.get(document_id)
                if self.contents_accessor is None:
                    return store_document.text
                elif isinstance(self.contents_accessor, str):
                    return getattr(store_document, self.contents_accessor)
                else:
                    return self.contents_accessor(store_document)
            except KeyError:
                # Document not found. Assume empty content.
                return ""

        if (
                self.contents_accessor is None or
                not isinstance(self.contents_accessor, str)
        ):
            raise ValueError(
                f"Cannot load contents "
                f"from metaindex field {self.contents_accessor}."
            )

        if self.contents_accessor not in self._meta_index_keys:
            raise ValueError(
                f"Index {self.index_location} did not have "
                f"requested metaindex key {self.contents_accessor}. "
                f"Keys present in metaindex "
                f"are {self._meta_index_keys}."
            )

        doc_id = int(self._meta_index.getDocument("docno", document_id))
        contents = str(self._meta_index.getItem(
            self.contents_accessor,
            doc_id,
        ))
        return contents

    def document_contents(self, document: Document) -> str:
        # Shortcut when text is given in the document.
        if isinstance(document, TextDocument):
            return document.contents

        return self._document_contents(document.id)

    @cached_property
    def _tokeniser(self) -> Tokeniser:
        if self.tokeniser is None:
            return EnglishTokeniser()
        return self.tokeniser

    @cached_property
    def _term_pipelines(self) -> Sequence[TermPipelineAccessor]:
        term_pipelines = str(ApplicationSetup.getProperty(
            "termpipelines",
            "Stopwords,PorterStemmer"
        ))
        return tuple(
            BaseTermPipelineAccessor(pipeline)
            for pipeline in split(r"\s*,\s*", term_pipelines.strip())
        )

    @lru_cache(None)
    def _terms(self, text: str) -> Sequence[str]:
        reader = StringReader(text)
        terms = tuple(
            str(term)
            for term in self._tokeniser.tokenise(reader)
            if term is not None
        )
        del reader

        for pipeline in self._term_pipelines:
            terms = tuple(
                str(term)
                for term in map(pipeline.pipelineTerm, terms)
                if term is not None
            )
        return terms

    def terms(
            self,
            query_or_document: Union[Query, Document]
    ) -> Sequence[str]:
        text = self.contents(query_or_document)
        return self._terms(text)
