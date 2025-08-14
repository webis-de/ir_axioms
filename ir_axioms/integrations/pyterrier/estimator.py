from typing import TYPE_CHECKING

from ir_axioms.utils.libraries import is_pyterrier_installed

if is_pyterrier_installed() or TYPE_CHECKING:
    from dataclasses import dataclass, field
    from functools import cached_property
    from itertools import groupby
    from typing import Sequence, Optional
    from warnings import warn

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
        require_columns,
        load_documents,
        load_queries,
    )
    from ir_axioms.model import Document, Query
    from ir_axioms.tools import PivotSelection, RandomPivotSelection

    @dataclass(frozen=True, kw_only=True)
    class EstimatorKwikSortReranker(Estimator):
        """
        A KwikSort re-ranker that uses a scikit-learn estimator to estimate the target preferences (e.g., ORACLE preferences) from the preferences of multiple axioms.
        This class extends from PyTerrier's ``Estimator`` so that it can be fitted/trained like any other PyTerrier estimator.
        """

        name = "EstimatorKwikSortReranker"

        axioms: Sequence[Axiom[Query, Document]]
        target: Axiom[Query, Document] = field(default_factory=ORACLE)
        estimator: ScikitLearnEstimator
        pivot_selection: PivotSelection = RandomPivotSelection()
        text_field: Optional[str] = "text"
        verbose: bool = False

        @cached_property
        def _estimator_axiom(self) -> EstimatorAxiom[Query, Document]:
            """
            Create an estimator axiom that uses the provided axioms and estimator.
            This is used to fit the estimator to the preferences of the axioms.
            """
            return ScikitLearnEstimatorAxiom(
                axioms=self.axioms,
                estimator=self.estimator,
            )

        @cached_property
        def _reranker(self) -> Transformer:
            """
            Create a KwikSort re-ranker using the internal estimator axiom.
            """
            return KwikSortReranker(
                axiom=self._estimator_axiom,
                pivot_selection=self.pivot_selection,
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
            Train the model with the given rankings and qrels.
            """

            # Check if validation data is provided.
            if topics_or_res_va is not None or qrels_va is not None:
                warn("Validation data will be ignored in this estimator.", UserWarning)

            # Check if the required columns are present.
            require_columns(topics_or_res_tr, {"qid", "docno"})
            require_columns(qrels_tr, {"qid", "docno", "label"})

            # Check if there are any results to fit to.
            if len(topics_or_res_tr.index) == 0:
                raise ValueError("No results to fit to.")

            # Merge the rankings with the qrels to have the labels available.
            topics_or_res_tr = topics_or_res_tr.merge(
                qrels_tr,
                on=["qid", "docno"],
                how="left",
                suffixes=(
                    "#old#",
                    "",
                ),  # Keep the old columns to warn about overwriting.
            )
            # If the columns were overwritten, warn the user.
            overwritten_columns = [
                col.removesuffix("#old#")
                for col in topics_or_res_tr.columns
                if col.endswith("#old#")
            ]
            if len(overwritten_columns) > 0:
                warn(
                    f"Columns were overwritten by the qrels: {', '.join(overwritten_columns)}.",
                    UserWarning,
                )
                topics_or_res_tr = topics_or_res_tr.drop(
                    columns=[f"{col}#old#" for col in overwritten_columns]
                )

            # Extract the queries and documents from the updated ranking.
            queries = load_queries(topics_or_res_tr)
            documents = load_documents(
                topics_or_res_tr,
                text_column=self.text_field,
            )

            # Group into batches of common queries and documents for each query.
            query_document_pairs = zip(queries, documents)
            query_document_batches = [
                (
                    query,
                    [document for _, document in query_documents],
                )
                for query, query_documents in groupby(
                    query_document_pairs, key=lambda x: x[0]
                )
            ]

            # Using the grouped queries and documents, fit the estimator axiom to the given target axiom.
            self._estimator_axiom.fit(
                target=self.target,
                inputs_outputs=query_document_batches,
            )

            # Return the fitted re-ranker (i.e., this object).
            return self

        def transform(self, inp: DataFrame) -> DataFrame:
            # Apply KwikSort re-ranking with the internal re-ranker.
            # The internal re-ranker uses an estimator axiom, which is assumed to be fitted at this point.
            return self._reranker.transform(inp)

else:
    EstimatorKwikSortReranker = NotImplemented
