from ir_axioms.utils.libraries import is_pyterrier_installed

if is_pyterrier_installed():
    from dataclasses import dataclass, field
    from functools import cached_property, cache
    from pathlib import Path
    from typing import Sequence, Union, Optional, Any, cast

    from ir_datasets import Dataset
    from pandas import DataFrame
    from pyterrier import Transformer, Estimator
    from pyterrier.datasets import IRDSDataset
    from pyterrier.java import (
        required as pt_java_required,
        autoclass as pt_java_autoclass,
    )

    from ir_axioms.axiom import Axiom, EstimatorAxiom, ORACLE
    from ir_axioms.axiom.estimator import ScikitLearnEstimator, ScikitLearnEstimatorAxiom
    from ir_axioms.integrations.pyterrier.utils import (
        inject_pyterrier,
        require_columns,
        load_documents,
        load_queries,
    )
    from ir_axioms.integrations.pyterrier.transformers import KwikSortReranker
    from ir_axioms.model import JudgedRankedDocument, Document, JudgedDocument, Query
    from ir_axioms.tools import PivotSelection, RandomPivotSelection

    @pt_java_required
    def autoclass(*args, **kwargs) -> Any:
        return pt_java_autoclass(*args, **kwargs)

    Index = autoclass("org.terrier.structures.Index")
    IndexRef = autoclass("org.terrier.querying.IndexRef")
    Tokeniser = autoclass("org.terrier.indexing.tokenisation.Tokeniser")
    EnglishTokeniser = autoclass("org.terrier.indexing.tokenisation.EnglishTokeniser")

    @dataclass(frozen=True, kw_only=True)
    class EstimatorKwikSortReranker(Estimator):
        name = "EstimatorKwikSortReranker"

        axioms: Sequence[Axiom[Query, Document]]
        oracle: Axiom[Query, JudgedDocument] = ORACLE()
        estimator: ScikitLearnEstimator
        pivot_selection: PivotSelection = RandomPivotSelection()
        index: Union[Index, IndexRef, Path, str]  # type: ignore
        dataset: Optional[Union[Dataset, str, IRDSDataset]] = None
        text_field: Optional[str] = "text"
        tokeniser: Tokeniser = field(  # type: ignore
            default_factory=lambda: EnglishTokeniser()
        )
        verbose: bool = False

        @cache
        def _inject(self) -> None:
            inject_pyterrier(
                index_location=self.index,
                text_field=self.text_field,
                tokeniser=self.tokeniser,
                dataset=self.dataset,
            )

        @cached_property
        def _estimator_axiom(self) -> EstimatorAxiom:
            return ScikitLearnEstimatorAxiom(
                axioms=self.axioms,
                estimator=self.estimator,
            )

        @cached_property
        def _reranker(self) -> Transformer:
            return KwikSortReranker(
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
            _qrels_validation: Optional[DataFrame] = None,
        ) -> Transformer:
            """
            Train the model with the given ranking.
            """
            require_columns(results_train, {"qid", "query", "docno", "rank", "score"})
            require_columns(qrels_train, {"qid", "docno", "label"})

            results_train = results_train.merge(
                qrels_train, on=["qid", "docno"], how="inner"
            )

            if len(results_train.index) == 0:
                raise ValueError("No results to fit to")

            queries = load_queries(results_train)
            # Because we joined with qrels before, it is safe to assume
            # that each document is judged.
            documents: Sequence[JudgedRankedDocument] = cast(
                Sequence[JudgedRankedDocument],
                load_documents(results_train, self.contents_accessor),
            )

            query_document_pairs = list(zip(queries, documents))

            # Inject the Terrier tooling.
            self._inject()

            self._estimator_axiom.fit(ORACLE(), query_document_pairs, self.filter_pairs)

            return self

        def transform(self, topics_or_res: DataFrame) -> DataFrame:
            return self._reranker.transform(topics_or_res)

else:
    EstimatorKwikSortReranker = NotImplemented  # type: ignore
