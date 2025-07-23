from typing import TYPE_CHECKING

from ir_axioms.utils.libraries import is_pyterrier_installed

if is_pyterrier_installed() or TYPE_CHECKING:
    from dataclasses import dataclass, field
    from functools import cached_property
    from itertools import groupby
    from pathlib import Path
    from typing import Sequence, Union, Optional, cast

    from ir_datasets import Dataset
    from pandas import DataFrame
    from pyterrier import Transformer, Estimator

    from ir_axioms.axiom.base import Axiom
    from ir_axioms.axiom.retrieval.simple import ORACLE
    from ir_axioms.axiom.estimator import (
        EstimatorAxiom,
        ScikitLearnEstimator,
        ScikitLearnEstimatorAxiom,
    )
    from ir_axioms.integrations.pyterrier.transformers import KwikSortReranker
    from ir_axioms.integrations.pyterrier.utils import (
        inject_pyterrier,
        require_columns,
        load_documents,
        load_queries,
    )
    from ir_axioms.model import JudgedRankedDocument, Document, JudgedDocument, Query
    from ir_axioms.tools import PivotSelection, RandomPivotSelection
    from ir_axioms.utils.pyterrier import (
        Index,
        IndexRef,
        Tokeniser,
        EnglishTokeniser,
    )

    def _get_oracle() -> Axiom[Query, JudgedDocument]:
        """
        Returns the oracle axiom for the estimator.
        """
        from ir_axioms.axiom.retrieval.simple import ORACLE

        return ORACLE()

    @dataclass(frozen=True, kw_only=True)
    class EstimatorKwikSortReranker(Estimator):
        name = "EstimatorKwikSortReranker"

        axioms: Sequence[Axiom[Query, Document]]
        oracle: Axiom[Query, JudgedDocument] = field(default_factory=_get_oracle)
        estimator: ScikitLearnEstimator
        pivot_selection: PivotSelection = RandomPivotSelection()
        index: Union[Index, IndexRef, Path, str]  # type: ignore
        dataset: Optional[Union[Dataset, str]] = None
        text_field: Optional[str] = "text"
        tokeniser: Tokeniser = field(  # type: ignore
            default_factory=lambda: EnglishTokeniser()
        )
        verbose: bool = False

        def _inject(self) -> None:
            inject_pyterrier(
                index_location=self.index,
                text_field=self.text_field,
                tokeniser=self.tokeniser,
                dataset=self.dataset,
            )

        @cached_property
        def _estimator_axiom(self) -> EstimatorAxiom[Query, Document]:
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
                pivot_selection=self.pivot_selection,
                tokeniser=self.tokeniser,
                verbose=self.verbose,
            )

        def fit(
            self,
            topics_or_res_tr: DataFrame,
            qrels_tr: DataFrame,
            topics_or_res_va: Optional[DataFrame] = None,
            qrels_va: Optional[DataFrame] = None,
        ) -> Transformer:
            """
            Train the model with the given ranking.
            """
            require_columns(
                topics_or_res_tr, {"qid", "query", "docno", "rank", "score"}
            )
            require_columns(qrels_tr, {"qid", "docno", "label"})

            topics_or_res_tr = topics_or_res_tr.merge(
                qrels_tr, on=["qid", "docno"], how="inner"
            )

            if len(topics_or_res_tr.index) == 0:
                raise ValueError("No results to fit to")

            queries = load_queries(topics_or_res_tr)
            # Because we joined with qrels before, it is safe to assume
            # that each document is judged.
            documents: Sequence[JudgedRankedDocument] = cast(
                Sequence[JudgedRankedDocument],
                load_documents(topics_or_res_tr, text_column=self.text_field),
            )

            query_document_pairs = list(zip(queries, documents))
            query_document_batches = [
                (
                    query,
                    [query_document[1] for query_document in query_documents],
                )
                for query, query_documents in groupby(
                    query_document_pairs, key=lambda x: x[0]
                )
            ]

            # Inject the Terrier tooling.
            self._inject()

            self._estimator_axiom.fit(ORACLE(), query_document_batches)

            return self

        def transform(self, inp: DataFrame) -> DataFrame:
            return self._reranker.transform(inp)

else:
    EstimatorKwikSortReranker = NotImplemented  # type: ignore
