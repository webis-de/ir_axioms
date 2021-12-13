from dataclasses import dataclass
from functools import cached_property, lru_cache
from logging import warning
from pathlib import Path
from re import split
from typing import List, Optional, Union

from ir_axioms.backend import PyTerrierBackendContext
from ir_axioms.model import Query, Document
from ir_axioms.model.context import RerankingContext
from ir_axioms.model.retrieval_model import (
    RetrievalModel, Tf, TfIdf, BM25, PL2, DirichletLM
)
from ir_axioms.utils import text_content

with PyTerrierBackendContext():
    from pandas import DataFrame
    from pyterrier import IndexRef, IndexFactory
    from pyterrier.batchretrieve import TextScorer
    from ir_axioms.backend.pyterrier.util import (
        StringReader, Tokeniser, EnglishTokeniser, PropertiesIndex, Lexicon,
        CollectionStatistics, ApplicationSetup, BaseTermPipelineAccessor,
        WeightingModel, TfModel, TfIdfModel, BM25Model, PL2Model,
        DirichletLMModel, with_properties, Index, TermPipelineAccessor,
    )


    @dataclass(unsafe_hash=True, frozen=True)
    class IndexRerankingContext(RerankingContext):
        index_location: Path
        tokeniser: Tokeniser = EnglishTokeniser()
        cache_dir: Optional[Path] = None

        @cached_property
        def _index_ref(self) -> IndexRef:
            return IndexRef.of(str(self.index_location.absolute()))

        @cached_property
        def _index(self) -> Union[PropertiesIndex, Index]:
            index = (IndexFactory.of(self._index_ref))
            return with_properties(index)

        @cached_property
        def _lexicon(self) -> Lexicon:
            return self._index.getLexicon()

        @cached_property
        def _collection_statistics(self) -> CollectionStatistics:
            return self._index.getCollectionStatistics()

        @cached_property
        def document_count(self) -> int:
            return self._collection_statistics.numberOfDocuments

        def document_frequency(self, term: str) -> int:
            print(self._lexicon.numberOfEntries())
            print(term)
            entry = self._lexicon.getLexiconEntry(term)
            print(entry)
            if entry is None:
                return 0
            return entry.getDocumentFrequency()

        @cached_property
        def _term_pipelines(self) -> List[TermPipelineAccessor]:
            print(self._index)
            if isinstance(self._index, PropertiesIndex):
                term_pipelines: str = self._index.getIndexProperty(
                    "termpipelines",
                    None
                )
            else:
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
            text = text_content(query_or_document)
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

        def retrieval_score(
                self,
                query: Query,
                document: Document,
                model: RetrievalModel
        ) -> float:
            weighting_model = self._weighting_model(model)
            scorer = TextScorer(
                wmodel=weighting_model,
                background_index=self._index
            )

            documents = DataFrame(
                data=[["q1", query.title, "d1", document.content]],
                columns=["qid", "query", "docno", "body"]
            )

            retrieved: DataFrame = scorer.transform(documents)
            score: float = retrieved.head(1)["score"]

            return score
