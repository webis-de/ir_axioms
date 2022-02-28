from dataclasses import dataclass
from functools import cached_property, lru_cache
from logging import warning
from pathlib import Path
from re import split
from typing import Optional, Union, Callable, NamedTuple, Sequence

from ir_datasets import Dataset, load
from ir_datasets.indices import Docstore

from ir_axioms.backend.pyterrier.config import (
    RETRIEVAL_SCORE_APPLICATION_PROPERTIES
)
from ir_axioms.backend.pyterrier.util import (
    EnglishTokeniser, Lexicon, CollectionStatistics, BaseTermPipelineAccessor,
    WeightingModel, TfModel, TfIdfModel, BM25Model, PL2Model, DirichletLMModel,
    Index, TermPipelineAccessor, Manager, ManagerFactory, SearchRequest,
    ScoredDocList, ScoredDoc, RequestContextMatching, MetaIndex, IndexRef,
    Tokeniser, IndexFactory, ApplicationSetup, StringReader
)
from ir_axioms.model import Query, Document, TextDocument, IndexContext
from ir_axioms.model.retrieval_model import (
    RetrievalModel, Tf, TfIdf, BM25, PL2, DirichletLM
)


@lru_cache(None)
def _weighting_model(model: RetrievalModel) -> WeightingModel:
    if isinstance(model, Tf):
        return TfModel()
    if isinstance(model, TfIdf):
        return TfIdfModel()
    elif isinstance(model, BM25):
        if model.k_1 != 1.2:
            warning(
                "The PyTerrier backend doesn't support setting "
                "the k_1 parameter for the BM25 retrieval model. "
                "It will be ignored."
            )
        if model.k_3 != 8:
            warning(
                "The Pyserini backend doesn't support setting "
                "the k_3 parameter for the BM25 retrieval model. "
                "It will be ignored."
            )
        weighting_model = BM25Model()
        weighting_model.setParameter(model.b)
        return weighting_model
    elif isinstance(model, PL2):
        return PL2Model(model.c)
    elif isinstance(model, DirichletLM):
        weighting_model = DirichletLMModel()
        weighting_model.setParameter(model.mu)
        return weighting_model
    else:
        raise NotImplementedError(
            f"The Pyserini backend doesn't support "
            f"the {type(model)} retrieval model."
        )


ContentsAccessor = Union[str, Callable[[NamedTuple], str]]


@dataclass(frozen=True)
class TerrierIndexContext(IndexContext):
    index_location: Union[Path, IndexRef, Index]
    dataset: Optional[Union[Dataset, str]] = None
    contents_accessor: Optional[ContentsAccessor] = "text"
    tokeniser: Optional[Tokeniser] = None
    cache_dir: Optional[Path] = None

    @cached_property
    def _index(self) -> Index:
        if isinstance(self.index_location, Index):
            return self.index_location
        elif isinstance(self.index_location, IndexRef):
            return IndexFactory.of(self.index_location)
        elif isinstance(self.index_location, Path):
            return IndexFactory.of(str(self.index_location.absolute()))
        else:
            raise ValueError(
                f"Cannot load index from location {self.index_location}."
            )

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

    @cached_property
    def _manager(self) -> Manager:
        index_ref = self._index.getIndexRef()
        # noinspection PyProtectedMember
        manager = ManagerFactory._from_(index_ref)
        del index_ref
        return manager

    @lru_cache(None)
    def _retrieval_score(
            self,
            query_title: str,
            document_id: str,
            weighting_model: WeightingModel
    ) -> float:
        if len(query_title) == 0:
            return 0

        # Setup retrieval as per BatchRetrieve.
        for key, value in RETRIEVAL_SCORE_APPLICATION_PROPERTIES.items():
            ApplicationSetup.setProperty(key, value)

        manager = self._manager

        # Build search request.
        request: SearchRequest = manager.newSearchRequestFromQuery(
            query_title
        )

        # Set weighting model.
        request.setControl("context_wmodel", "on")
        request.setContextObject("context_wmodel", weighting_model)

        # Filter documents matching the original ID.
        matching = RequestContextMatching.of(request)
        matching.fromDocnos([document_id])
        matching.build()

        # Return search score from weighting model.
        request.setControl(
            "matching",
            ",".join([
                "org.terrier.matching.ScoringMatching",
                request.getControl("matching")
            ])
        )

        # Limit to 1 result.
        request.setControl("end", str(1 - 1))

        # Execute search request.
        manager.runSearchRequest(request)

        # Parse result document.
        result_docs: ScoredDocList = request.getResults()

        if len(result_docs) == 0:
            # Document was not received.
            return 0

        assert len(result_docs) == 1
        first_result_doc: ScoredDoc = result_docs[0]
        assert str(first_result_doc.getMetadata("docno")) == document_id

        # Get retrieval score.
        retrieval_score = float(first_result_doc.getScore())

        # Cleanup.
        del first_result_doc
        del result_docs
        del request
        del matching

        return retrieval_score

    def retrieval_score(
            self,
            query: Query,
            document: Document,
            model: RetrievalModel
    ) -> float:
        return self._retrieval_score(
            query.title,
            document.id,
            _weighting_model(model)
        )
