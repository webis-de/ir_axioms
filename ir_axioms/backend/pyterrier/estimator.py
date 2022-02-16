from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Sequence, Union, Optional, Tuple, Iterable

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
    require_columns, JoinQrelsTransformer, load_documents, load_queries
)
from ir_axioms.backend.pyterrier.transformers import AxiomaticReranker
from ir_axioms.model import IndexContext
from ir_axioms.model.base import Query, RankedDocument, JudgedRankedDocument


@dataclass(frozen=True)
class EstimatorKwikSortReranker(EstimatorBase):
    name = "EstimatorKwikSortReranker"

    axioms: Sequence[AxiomLike]
    estimator: ScikitEstimatorType
    index: Union[Path, IndexRef, Index]
    dataset: Optional[Union[Dataset, str]] = None
    contents_accessor: Optional[ContentsAccessor] = "text"
    tokeniser: Optional[Tokeniser] = None
    cache_dir: Optional[Path] = None
    verbose: bool = False

    @cached_property
    def _axioms(self) -> Sequence[Axiom]:
        return to_axioms(self.axioms)

    @cached_property
    def _estimator_axiom(self) -> EstimatorAxiom:
        return ScikitLearnEstimatorAxiom(self._axioms, self.estimator)

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

        join_qrels = JoinQrelsTransformer(qrels_train)
        results_train = join_qrels.transform(results_train)

        if len(results_train.index) == 0:
            raise ValueError("No results to fit to")

        queries = load_queries(results_train)
        documents: Sequence[RankedDocument] = load_documents(results_train,
                                                             self.contents_accessor)

        # Because we joined with qrels before, it is safe to assume
        # that each document is judged.
        documents: Sequence[JudgedRankedDocument]

        query_document_pairs: Iterable[Tuple[Query, JudgedRankedDocument]] = (
            zip(queries, documents)
        )

        self._estimator_axiom.fit_oracle(self._context, query_document_pairs)

        return self

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        return self._reranker.transform(topics_or_res)
