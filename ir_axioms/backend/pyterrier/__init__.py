from dataclasses import dataclass
from functools import cached_property, lru_cache
from logging import warning
from pathlib import Path
from re import split
from typing import List, Optional, Union, Callable, NamedTuple

from ir_datasets import Dataset, load
from ir_datasets.indices import Docstore

from ir_axioms.backend.pyterrier.util import (
    EnglishTokeniser, Lexicon, CollectionStatistics, BaseTermPipelineAccessor,
    WeightingModel, TfModel, TfIdfModel, BM25Model, PL2Model, DirichletLMModel,
    Index, TermPipelineAccessor, Manager, ManagerFactory, SearchRequest,
    ScoredDocList, ScoredDoc, RequestContextMatching, MetaIndex, IndexRef,
    Tokeniser, IndexFactory, ApplicationSetup, StringReader
)
from ir_axioms.model import Query, Document, TextDocument
from ir_axioms.model.context import RerankingContext
from ir_axioms.model.retrieval_model import (
    RetrievalModel, Tf, TfIdf, BM25, PL2, DirichletLM
)

_retrieval_score_application_properties = {
    "querying.processes": ",".join([
        "terrierql:TerrierQLParser",
        "parsecontrols:TerrierQLToControls",
        "parseql:TerrierQLToMatchingQueryTerms",
        "applypipeline:ApplyTermPipeline",
        "context_wmodel:org.terrier.python.WmodelFromContextProcess",
        "localmatching:LocalManager$ApplyLocalMatching",
        "filters:LocalManager$PostFilterProcess"
    ]),
    "querying.postfilters": "decorate:SimpleDecorate",
    "querying.default.controls": ",".join([
        "parsecontrols:on",
        "parseql:on",
        "applypipeline:on",
        "terrierql:on",
        "localmatching:on",
        "filters:on",
        "decorate:on"
    ]),
}


@dataclass(unsafe_hash=True, frozen=True)
class IndexRerankingContext(RerankingContext):
    index_location: Union[Path, IndexRef, Index]
    dataset: Optional[Union[Dataset, str]] = None
    contents_accessor: Optional[Union[
        str,
        Callable[[NamedTuple], str]
    ]] = "text"
    tokeniser: Tokeniser = EnglishTokeniser()
    cache_dir: Optional[Path] = None

    @cached_property
    def _index_ref(self) -> IndexRef:
        if isinstance(self.index_location, IndexRef):
            return self.index_location
        elif isinstance(self.index_location, Index):
            return self.index_location.getIndexRef()
        else:
            return IndexRef.of(str(self.index_location.absolute()))

    @cached_property
    def _index(self) -> Index:
        if isinstance(self.index_location, Index):
            return self.index_location
        else:
            return IndexFactory.of(self._index_ref)

    @cached_property
    def _meta_index(self) -> MetaIndex:
        meta_index = self._index.getMetaIndex()
        if meta_index is None:
            raise ValueError(
                f"Index {self.index_location} does not have a metaindex."
            )
        return meta_index

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
        else:
            return load(self.dataset)

    @cached_property
    def document_count(self) -> int:
        return self._collection_statistics.numberOfDocuments

    @lru_cache
    def document_frequency(self, term: str) -> int:
        entry = self._lexicon.getLexiconEntry(term)
        if entry is None or entry.getNumberOfEntries() == 0:
            return 0
        return entry.getDocumentFrequency()

    @lru_cache
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

        metaindex_keys: List[str] = self._meta_index.getKeys()
        if self.contents_accessor not in metaindex_keys:
            raise ValueError(
                f"Index {self.index_location} did not have "
                f"requested metaindex key {self.contents_accessor}. "
                f"Keys present in metaindex "
                f"are {metaindex_keys}."
            )
        doc_id = self._meta_index.getDocument("docno", document.id)
        return self._meta_index.getItem(
            self.contents_accessor,
            doc_id
        )

    @cached_property
    def _term_pipelines(self) -> List[TermPipelineAccessor]:
        term_pipelines = ApplicationSetup.getProperty(
            "termpipelines",
            "Stopwords,PorterStemmer"
        )
        return [
            BaseTermPipelineAccessor(pipeline)
            for pipeline in split(r"\s*,\s*", term_pipelines.strip())
        ]

    @lru_cache
    def terms(
            self,
            query_or_document: Union[Query, Document]
    ) -> List[str]:
        text = self.contents(query_or_document)
        reader = StringReader(text)
        terms = list(self.tokeniser.tokenise(reader))
        terms = [term for term in terms if term is not None]
        for pipeline in self._term_pipelines:
            terms = [
                pipeline.pipelineTerm(term)
                for term in terms
            ]
        terms = [term for term in terms if term is not None]
        return terms

    @staticmethod
    @lru_cache
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

    @cached_property
    def _manager(self) -> Manager:
        # noinspection PyProtectedMember
        return ManagerFactory._from_(self._index_ref)

    @lru_cache
    def retrieval_score(
            self,
            query: Query,
            document: Document,
            model: RetrievalModel
    ) -> float:
        if len(query.title) == 0:
            return 0

        # Setup retrieval as per BatchRetrieve.
        for key, value in _retrieval_score_application_properties.items():
            ApplicationSetup.setProperty(key, value)

        # Build search request.
        request: SearchRequest = self._manager.newSearchRequestFromQuery(
            query.title
        )

        # Set weighting model.
        request.setControl("context_wmodel", "on")
        request.setContextObject(
            "context_wmodel",
            self._weighting_model(model)
        )

        # Filter documents matching the original ID.
        matching = RequestContextMatching.of(request)
        matching.fromDocnos([document.id])
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
        self._manager.runSearchRequest(request)

        # Parse result document.
        result_docs: ScoredDocList = request.getResults()
        assert len(result_docs) == 1
        first_result_doc: ScoredDoc = result_docs[0]
        assert first_result_doc.getMetadata("docno") == document.id

        # Get retrieval score.
        score = first_result_doc.getScore()
        return score
