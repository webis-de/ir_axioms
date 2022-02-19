from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Sequence, Union, Optional, Callable

from ir_datasets import Dataset
from pandas import DataFrame
from pyterrier.transformer import EstimatorBase

from ir_axioms.axiom import Axiom, AxiomLike, EstimatorAxiom, to_axioms
from ir_axioms.axiom.estimator import (
    ScikitEstimatorType, ScikitLearnEstimatorAxiom
)
from ir_axioms.backend.pyterrier import (
    IndexRef, Index, ContentsAccessor, Tokeniser, TerrierIndexContext
)
from ir_axioms.backend.pyterrier.safe import Transformer
from ir_axioms.backend.pyterrier.transformer_utils import (
    require_columns, load_documents, load_queries
)
from ir_axioms.backend.pyterrier.transformers import AxiomaticReranker
from ir_axioms.model import IndexContext
from ir_axioms.model.base import RankedDocument, JudgedRankedDocument
from ir_axioms.modules.ranking import PivotSelection, RandomPivotSelection


@dataclass(frozen=True)
class EstimatorKwikSortReranker(EstimatorBase):
    name = "EstimatorKwikSortReranker"

    axioms: Sequence[AxiomLike]
    estimator: ScikitEstimatorType
    index: Union[Path, IndexRef, Index]
    dataset: Optional[Union[Dataset, str]] = None
    contents_accessor: Optional[ContentsAccessor] = "text"
    pivot_selection: PivotSelection = RandomPivotSelection(),
    filter_pairs: Optional[Callable[
        [JudgedRankedDocument, JudgedRankedDocument],
        bool
    ]] = None
    tokeniser: Optional[Tokeniser] = None
    cache_dir: Optional[Path] = None
    verbose: bool = False

    @cached_property
    def _axioms(self) -> Sequence[Axiom]:
        return to_axioms(self.axioms)

    @cached_property
    def _estimator_axiom(self) -> EstimatorAxiom:
        return ScikitLearnEstimatorAxiom(
            self._axioms,
            self.estimator,
            verbose=self.verbose
        )

    @cached_property
    def _context(self) -> IndexContext:
        return TerrierIndexContext(
            index_location=self.index,
            dataset=self.dataset,
            contents_accessor=self.contents_accessor,
            tokeniser=self.tokeniser,
            cache_dir=self.cache_dir,
        )

    @cached_property
    def _reranker(self) -> Transformer:
        return AxiomaticReranker(
            axiom=self._estimator_axiom,
            index=self.index,
            dataset=self.dataset,
            contents_accessor=self.contents_accessor,
            pivot_selection=self.pivot_selection,
            tokeniser=self.tokeniser,
            cache_dir=self.cache_dir,
            verbose=self.verbose,
        )

    def fit(
            self,
            results_train: DataFrame,
            qrels_train: DataFrame,
            _results_validation: Optional[DataFrame] = None,
            _qrels_validation: Optional[DataFrame] = None
    ) -> Transformer:
        """
        Train the model with the given ranking.
        """
        require_columns(
            results_train,
            {"qid", "query", "docno", "rank", "score"}
        )
        require_columns(qrels_train, {"qid", "docno", "label"})

        results_train = results_train.merge(
            qrels_train,
            on=["qid", "docno"],
            how="inner"
        )

        if len(results_train.index) == 0:
            raise ValueError("No results to fit to")

        queries = load_queries(results_train)
        documents: Sequence[RankedDocument] = load_documents(
            results_train,
            self.contents_accessor
        )

        # Because we joined with qrels before, it is safe to assume
        # that each document is judged.
        documents: Sequence[JudgedRankedDocument]

        query_document_pairs = list(zip(queries, documents))

        self._estimator_axiom.fit_oracle(
            self._context,
            query_document_pairs,
            self.filter_pairs
        )

        return self

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        return self._reranker.transform(topics_or_res)
