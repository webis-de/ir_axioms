from typing import TYPE_CHECKING

from ir_axioms.utils.libraries import is_pyterrier_installed

if is_pyterrier_installed() or TYPE_CHECKING:
    from abc import abstractmethod, ABC
    from dataclasses import dataclass, field
    from itertools import product
    from logging import DEBUG
    from pathlib import Path
    from typing import (
        Union,
        Optional,
        Set,
        Sequence,
        Callable,
        final,
        Any,
        cast,
        ClassVar,
        Iterable,
    )

    from ir_datasets import Dataset
    from numpy import apply_along_axis, stack, ndarray, array
    from pandas import DataFrame
    from pandas.core.groupby import DataFrameGroupBy
    from pyterrier import Transformer
    from tqdm.auto import tqdm

    from ir_axioms.logging import logger
    from ir_axioms.axiom.base import Axiom
    from ir_axioms.integrations.pyterrier.utils import (
        inject_pyterrier,
        require_columns,
        load_documents,
    )
    from ir_axioms.model import Query, RankedDocument, ScoredDocument, Document
    from ir_axioms.tools import PivotSelection, RandomPivotSelection
    from ir_axioms.utils.pyterrier import (
        Index,
        IndexRef,
        Tokeniser,
        EnglishTokeniser,
    )

    @dataclass(frozen=True, kw_only=True)
    class _PerGroupTransformer(Transformer, ABC):
        group_columns: ClassVar[Set[str]]
        optional_group_columns: ClassVar[Set[str]] = set()
        description: ClassVar[Optional[str]] = None
        unit: ClassVar[str] = "it"

        verbose: bool = False

        @abstractmethod
        def transform_group(self, topics_or_res: DataFrame) -> DataFrame:
            pass

        def _all_group_columns(self, topics_or_res: DataFrame) -> Set[str]:
            return self.group_columns | {
                column
                for column in self.optional_group_columns
                if column in topics_or_res.columns
            }

        @final
        def transform(self, inp: DataFrame) -> DataFrame:
            require_columns(inp, self.group_columns)

            query_rankings: DataFrameGroupBy = inp.groupby(
                by=list(self._all_group_columns(inp)),
                as_index=False,
                sort=False,
            )
            if self.verbose:
                # Show progress during reranking queries.
                tqdm.pandas(
                    desc=self.description,
                    unit=self.unit,
                )
                inp = query_rankings.progress_apply(self.transform_group)  # type: ignore
            else:
                inp = query_rankings.apply(self.transform_group)
            return inp.reset_index(drop=True)

    @dataclass(frozen=True, kw_only=True)
    class _AxiomTransformer(_PerGroupTransformer, ABC):
        group_columns = {"query"}
        optional_group_columns = {"qid", "name"}
        unit = "query"

        index: Union[Index, IndexRef, Path, str]  # type: ignore
        dataset: Optional[Union[Dataset, str]] = None
        text_field: Optional[str] = "text"
        tokeniser: Tokeniser = field(  # type: ignore
            default_factory=lambda: EnglishTokeniser()
        )

        def _inject(self) -> None:
            inject_pyterrier(
                index_location=self.index,
                text_field=self.text_field,
                tokeniser=self.tokeniser,
                dataset=self.dataset,
            )

        @final
        def transform_group(self, topics_or_res: DataFrame) -> DataFrame:
            require_columns(topics_or_res, {"query", "docno", "rank", "score"})

            if len(topics_or_res.index) == 0:
                # Empty ranking, skip reranking.
                return topics_or_res

            # Convert query.
            # As we grouped per query, we don't expect multiple queries here.
            if topics_or_res["query"].nunique() > 1:
                raise RuntimeError("Expected only one query in this data frame.")
            query = Query(topics_or_res.iloc[0]["query"])

            # Load document list.
            documents = load_documents(topics_or_res, text_column=self.text_field)

            # Inject the Terrier tooling.
            self._inject()

            return self.transform_query_ranking(query, documents, topics_or_res)

        @abstractmethod
        def transform_query_ranking(
            self,
            query: Query,
            documents: Sequence[RankedDocument],
            topics_or_res: DataFrame,
        ) -> DataFrame:
            pass

    @dataclass(frozen=True, kw_only=True)
    class KwikSortReranker(_AxiomTransformer):
        name = "KwikSortReranker"
        description = "Rerank with KwikSort"

        axiom: Axiom[Query, Document]
        pivot_selection: PivotSelection = RandomPivotSelection()

        def transform_query_ranking(
            self,
            query: Query,
            documents: Sequence[Document],
            topics_or_res: DataFrame,
        ) -> DataFrame:
            # Rerank documents.
            reranked_documents = self.axiom.rerank_kwiksort(
                input=query,
                ranking=documents,
                pivot_selection=self.pivot_selection,
            )

            # Convert reranked documents back to data frame.
            reranked_data: dict[str, Any] = {
                "docno": [doc.id for doc in reranked_documents],
            }
            if all(isinstance(doc, RankedDocument) for doc in reranked_documents):
                reranked_data["rank"] = [
                    cast(RankedDocument, doc).rank for doc in reranked_documents
                ]
            if all(isinstance(doc, ScoredDocument) for doc in reranked_documents):
                reranked_data["score"] = [
                    cast(ScoredDocument, doc).score for doc in reranked_documents
                ]
            reranked = DataFrame(reranked_data)

            # Remove original scores and ranks.
            original_ranking = topics_or_res.copy()
            del original_ranking["rank"]
            del original_ranking["score"]

            # Merge with new scores.
            reranked = reranked.merge(original_ranking, on="docno")
            return reranked

    @dataclass(frozen=True, kw_only=True)
    class AggregatedAxiomaticPreferences(_AxiomTransformer):
        name = "AggregatedAxiomaticPreferences"
        description = "Aggregate axiom preferences"

        axioms: Sequence[Axiom[Query, Document]]
        aggregations: Sequence[Callable[[Sequence[float]], float]]

        def transform_query_ranking(
            self,
            query: Query,
            documents: Sequence[Document],
            topics_or_res: DataFrame,
        ) -> DataFrame:
            aggregations = self.aggregations

            # Shape: |documents| x |documents| x |axioms|
            features: ndarray = stack(
                tuple(
                    # Shape: |documents| x |documents|
                    axiom.preferences(
                        input=query,
                        outputs=documents,
                    )
                    for axiom in self.axioms
                ),
                -1,
            )

            # Shape: |documents| x |axioms| x |aggregations|
            features = stack(
                tuple(
                    # Shape: |documents| x |axioms|
                    apply_along_axis(
                        lambda preferences: aggregation(preferences.tolist()),
                        0,
                        features,
                    )
                    for aggregation in aggregations
                ),
                -1,
            )

            # Shape: |documents| x (|aggregations| * |axioms|)
            features = features.reshape((features.shape[0], -1))

            topics_or_res["features"] = list(map(array, features))
            return topics_or_res

    @dataclass(frozen=True, kw_only=True)
    class AxiomaticPreferences(_AxiomTransformer):
        name = "AxiomaticPreferences"
        description = "Compute axiom preferences"

        axioms: Sequence[Axiom]
        axiom_names: Optional[Sequence[str]] = None
        verbose: bool = False

        def transform_query_ranking(
            self,
            query: Query,
            documents: Sequence[RankedDocument],
            topics_or_res: DataFrame,
        ) -> DataFrame:
            # Result cross product.
            results_pairs = topics_or_res.merge(
                topics_or_res,
                on=list(self._all_group_columns(topics_or_res)),
                suffixes=("_a", "_b"),
            )

            # Document pairs.
            document_pairs: Iterable[tuple[Document, Document]] = list(
                product(documents, documents)
            )

            # Compute axiom preferences.
            axioms: Iterable[Axiom[Query, Document]] = self.axioms
            if self.verbose and 0 < logger.level <= DEBUG:
                axioms = tqdm(
                    axioms,
                    desc="Axiom preferences",
                    unit="axiom",
                )

            names: Sequence[str]
            if self.axiom_names is not None:
                if len(self.axiom_names) != len(self.axioms):
                    raise ValueError("Number of axioms and names must match.")
                names = self.axiom_names
            else:
                names = [str(axiom) for axiom in axioms]

            columns = [f"{name}_preference" for name in names]

            for column, axiom in zip(columns, axioms):
                if self.verbose and 0 < logger.level <= DEBUG:
                    # Very verbose progress bars.
                    document_pairs = tqdm(
                        document_pairs,
                        desc="Axiom preference",
                        unit="pair",
                    )
                results_pairs[column] = [
                    axiom.preference(query, document1, document2)
                    for document1, document2 in document_pairs
                ]

            return results_pairs

else:
    KwikSortReranker = NotImplemented  # type: ignore
    AggregatedAxiomaticPreferences = NotImplemented  # type: ignore
    AxiomaticPreferences = NotImplemented  # type: ignore
